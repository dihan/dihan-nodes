"""Lightweight mobile status page for ComfyUI.

Serves a small, self-contained dashboard at ``/status`` on ComfyUI's own web
server, so it is reachable at whatever port ComfyUI was launched with
(``:8188`` by default, ``:8189`` if you passed ``--port 8189``).

The page is a few KB of HTML/CSS/JS with no dependencies -- it is meant to be
opened on a phone to check on a run without loading the full ComfyUI frontend.

Progress is captured by teeing ``PromptServer.send_sync``: every event the
server emits is recorded into a small in-process snapshot before being passed
through untouched. This is why the page sees live progress even though those
events are addressed only to the client that submitted the prompt.

Failures are recorded two ways, because they fail in two very different ways:

* A node raising (including a CUDA OOM) emits ``execution_error``. That is
  caught, enriched with the traceback, the memory state at the moment it blew
  up, and the tail of the terminal log, then appended to ``failures.jsonl``.

* The process being killed outright -- the Linux OOM killer taking ComfyUI
  when system RAM is exhausted -- emits nothing at all, because there is no
  longer a process to emit it. For that case a marker file tracks the run in
  flight; if it is still there at the next startup, the previous run died and
  the kernel log is searched for the evidence.

Persisting to disk is what makes the second case work: the page has to be able
to explain a crash that took the page down with it.
"""

import json
import os
import subprocess
import threading
import time
from collections import deque

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - not running inside ComfyUI
    web = None
    PromptServer = None

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(HERE, "web", "status.html")

STATE_DIR = os.path.join(HERE, ".state")
FAILURE_PATH = os.path.join(STATE_DIR, "failures.jsonl")
MARKER_PATH = os.path.join(STATE_DIR, "current_run.json")

HISTORY_LEN = 12
LOG_TAIL_DEFAULT = 200

# Bounds on what gets persisted per failure. Enough to diagnose, small enough
# that the file stays readable and the /api/failures response stays sane.
MAX_FAILURES = 50
FAIL_LOG_LINES = 40
FAIL_LOG_LINE_CHARS = 400
FAIL_TRACE_FRAMES = 20

# How long after an execution_error the task_done fallback assumes the failure
# has already been recorded in detail and stays quiet.
DEDUPE_WINDOW = 30.0

_BOOT_TIME = time.time()
_STATE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Failure persistence
# ---------------------------------------------------------------------------

# Substrings that mark an exception as a memory exhaustion rather than an
# ordinary bug. Matched case-insensitively against the type and message.
_OOM_MARKERS = (
    "outofmemoryerror",
    "cuda out of memory",
    "out of memory",
    "can't allocate memory",
    "cannot allocate memory",
    "defaultcpuallocator: not enough memory",
    "no memory available",
)

# Kernel log lines that indicate the *process* was killed for memory reasons.
# The dxgk entry is WSL specific: it is the GPU paravirtualisation layer
# reporting ENOMEM (-12) when it cannot make an allocation resident in VRAM.
_KERNEL_OOM_PATTERNS = (
    "out of memory: killed process",
    "oom-kill:",
    "oom_reaper",
    "killed process",
    "dxgkio_make_resident: ioctl failed: -12",
)

_failure_seq = 0


def _ensure_state_dir():
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        return True
    except OSError:
        return False


def _classify_oom(exception_type, message):
    """Return True when the text looks like memory exhaustion."""
    blob = ("%s %s" % (exception_type or "", message or "")).lower()
    return any(marker in blob for marker in _OOM_MARKERS)


def _log_tail(lines=FAIL_LOG_LINES):
    """Last few lines of ComfyUI's own terminal log, trimmed for storage."""
    try:
        from app.logger import get_logs

        entries = list(get_logs())[-lines:]
    except Exception:
        return []
    out = []
    for entry in entries:
        try:
            text = str(entry.get("m", "")).rstrip("\n")
        except Exception:
            continue
        if not text:
            continue
        out.append(text[:FAIL_LOG_LINE_CHARS])
    return out


def _kernel_evidence(since=None, pid=None, limit=12):
    """Best-effort scan of the kernel ring buffer for OOM-kill evidence.

    Runs only after a suspected crash, never on the hot path. Both readers are
    tried because either can be unavailable depending on how the distro is
    started; failure to read is not itself interesting, so it stays quiet.
    """
    commands = (
        ["dmesg", "--ctime"],
        ["journalctl", "-k", "--no-pager", "-n", "2000"],
    )
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0 or not proc.stdout:
            continue

        text = proc.stdout.decode("utf-8", "replace")
        hits = []
        for line in text.splitlines():
            low = line.lower()
            if not any(pat in low for pat in _KERNEL_OOM_PATTERNS):
                continue
            # A pid match is strong evidence; without one the line still counts,
            # since the ring buffer may have wrapped past the kill message.
            hits.append(line.strip()[:FAIL_LOG_LINE_CHARS])
        if hits:
            if pid is not None:
                exact = [h for h in hits if str(pid) in h]
                if exact:
                    return exact[-limit:]
            return hits[-limit:]
    return []


def _next_failure_id():
    global _failure_seq
    _failure_seq += 1
    return "%d-%d" % (int(time.time() * 1000), _failure_seq)


def _append_failure(record):
    """Append one failure to the log, trimming it back to MAX_FAILURES."""
    if not _ensure_state_dir():
        return None
    record.setdefault("id", _next_failure_id())
    record.setdefault("at", time.time())
    line = json.dumps(record, default=str)

    with _STATE_LOCK:
        try:
            with open(FAILURE_PATH, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            print("[dihan-nodes] could not write failure log: %s" % exc)
            return None

        try:
            with open(FAILURE_PATH, "r", encoding="utf-8") as handle:
                kept = handle.readlines()
            if len(kept) > MAX_FAILURES:
                with open(FAILURE_PATH, "w", encoding="utf-8") as handle:
                    handle.writelines(kept[-MAX_FAILURES:])
        except OSError:
            pass
    return record


def _read_failures(limit=10):
    """Most recent failures first."""
    with _STATE_LOCK:
        try:
            with open(FAILURE_PATH, "r", encoding="utf-8") as handle:
                raw = handle.readlines()
        except OSError:
            return []

    out = []
    for line in reversed(raw):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


def _summarise(record):
    """Compact form carried in every snapshot; detail is fetched on demand."""
    message = (record.get("message") or "").strip()
    first = message.splitlines()[0] if message else ""
    return {
        "id": record.get("id"),
        "at": record.get("at"),
        "kind": record.get("kind"),
        "oom": bool(record.get("oom")),
        "node_label": record.get("node_label") or record.get("node_type"),
        "node_id": record.get("node_id"),
        "exception_type": record.get("exception_type"),
        "message": first[:200],
    }


# -- crash marker ------------------------------------------------------------
#
# Written while a run is in flight and removed when it ends. Surviving the
# process is the whole point: a marker present at startup means the previous
# run never reached an end state.

def _write_marker(payload):
    if not _ensure_state_dir():
        return
    try:
        with open(MARKER_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except (OSError, TypeError):
        pass


def _read_marker():
    try:
        with open(MARKER_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _clear_marker():
    try:
        os.remove(MARKER_PATH)
    except OSError:
        pass


class StatusTracker:
    """Rolling snapshot of what the executor is currently doing."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset_run()
        self.queue_remaining = 0
        self.history = deque(maxlen=HISTORY_LEN)
        self.last_event = None
        self.last_event_at = None
        self.failures = deque(_summarise(r) for r in _read_failures(8))
        self._recent_error_at = 0.0

    def reset_run(self):
        self.prompt_id = None
        self.started = None
        self.node_id = None
        self.node_started = None
        self.node_value = 0.0
        self.node_max = 0.0
        self.nodes = {}
        self.cached = set()
        self.error = None
        self.graph = None
        self._marker_at = 0.0

    # -- event intake ----------------------------------------------------

    def handle(self, event, data):
        if not isinstance(data, dict):
            data = {}
        with self._lock:
            self.last_event = event
            self.last_event_at = time.time()

            if event == "status":
                info = (data.get("status") or {}).get("exec_info") or {}
                self.queue_remaining = info.get("queue_remaining", 0)

            elif event == "execution_start":
                self.reset_run()
                self.prompt_id = data.get("prompt_id")
                self.started = time.time()
                self._mark(force=True)

            elif event == "execution_cached":
                for nid in data.get("nodes") or []:
                    self.cached.add(str(nid))

            elif event == "executing":
                node = data.get("node")
                if node is None:
                    self._finish("done")
                else:
                    self._enter_node(str(node))

            elif event == "progress":
                node = data.get("node")
                if node is not None:
                    self._enter_node(str(node))
                self.node_value = _num(data.get("value"))
                self.node_max = _num(data.get("max"))

            elif event == "progress_state":
                self._apply_progress_state(data)

            elif event == "executed":
                node = data.get("node")
                if node is not None:
                    entry = self.nodes.setdefault(str(node), {})
                    entry["state"] = "finished"

            elif event == "execution_error":
                self.error = {
                    "node_id": str(data.get("node_id") or ""),
                    "node_type": data.get("node_type") or "",
                    "message": data.get("exception_message") or "",
                }
                self._record_error(data)
                self._finish("error")

            elif event == "execution_interrupted":
                self._finish("interrupted")

            elif event == "execution_success":
                self._finish("success")

    def _enter_node(self, node_id):
        if node_id != self.node_id:
            self.node_id = node_id
            self.node_started = time.time()
            self.node_value = 0.0
            self.node_max = 0.0
            self._mark()

    # -- crash marker ----------------------------------------------------

    def _mark(self, force=False):
        """Record what is in flight, so a kill leaves a trail.

        Called on node transitions rather than on every progress tick, and
        throttled on top of that: this sits on the executor's event path, so
        it must stay cheap.
        """
        now = time.time()
        if not force and now - self._marker_at < 1.0:
            return
        self._marker_at = now
        _write_marker(
            {
                "pid": os.getpid(),
                "prompt_id": self.prompt_id,
                "started": self.started,
                "updated": now,
                "node_id": self.node_id,
                "node_label": self._label_for(self.node_id),
                "value": self.node_value,
                "max": self.node_max,
            }
        )

    def _label_for(self, node_id):
        """Resolve a node id to a human label, caching the graph per run."""
        if node_id is None:
            return None
        if self.graph is None:
            try:
                queue = PromptServer.instance.prompt_queue
                getter = (getattr(queue, "get_current_queue_volatile", None)
                          or queue.get_current_queue)
                current_running, _ = getter()
                for item in current_running or []:
                    if len(item) > 2 and isinstance(item[2], dict):
                        if not self.prompt_id or item[1] == self.prompt_id:
                            self.graph = item[2]
                            break
            except Exception:
                pass
            if self.graph is None:
                self.graph = {}
        return _node_label(self.graph, node_id)

    # -- failure recording -----------------------------------------------

    def _record_error(self, data):
        """Turn an execution_error event into a durable failure record."""
        message = data.get("exception_message") or ""
        exception_type = data.get("exception_type") or ""
        node_id = str(data.get("node_id") or "") or None
        oom = _classify_oom(exception_type, message)

        traceback_lines = data.get("traceback")
        if not isinstance(traceback_lines, list):
            traceback_lines = []
        traceback_lines = [
            str(frame)[:FAIL_LOG_LINE_CHARS]
            for frame in traceback_lines[-FAIL_TRACE_FRAMES:]
        ]

        record = {
            "kind": "node_error",
            "prompt_id": data.get("prompt_id") or self.prompt_id,
            "node_id": node_id,
            "node_type": data.get("node_type") or "",
            "node_label": self._label_for(node_id) or data.get("node_type") or "",
            "exception_type": exception_type,
            "message": message.strip(),
            "traceback": traceback_lines,
            "oom": oom,
            "elapsed": (time.time() - self.started) if self.started else None,
            "memory": _system_info(),
            "log_tail": _log_tail(),
        }
        # A CUDA OOM often has a kernel-side counterpart (the WSL dxgk ENOMEM
        # lines), which distinguishes "this allocation failed" from "the whole
        # GPU allocator is wedged".
        if oom:
            record["kernel"] = _kernel_evidence(since=self.started, pid=os.getpid())

        self._recent_error_at = time.time()
        self._push_failure(record)

    def note_unreported_failure(self, messages):
        """Fallback for a failed prompt that never emitted execution_error.

        ``add_message`` only reaches ``send_sync`` when a client is attached,
        so an API-submitted prompt can fail silently as far as the hook is
        concerned. ``task_done`` always runs, so it backstops that case.
        """
        with self._lock:
            if time.time() - self._recent_error_at < DEDUPE_WINDOW:
                return  # already captured in full detail
            text = "\n".join(str(m) for m in (messages or []))[:4000]
            self._push_failure(
                {
                    "kind": "queue_error",
                    "prompt_id": self.prompt_id,
                    "node_label": self._label_for(self.node_id),
                    "node_id": self.node_id,
                    "exception_type": "",
                    "message": text or "Prompt reported an error with no detail.",
                    "traceback": [],
                    "oom": _classify_oom("", text),
                    "memory": _system_info(),
                    "log_tail": _log_tail(),
                }
            )

    def _push_failure(self, record):
        stored = _append_failure(record)
        if stored:
            self.failures.appendleft(_summarise(stored))
            while len(self.failures) > 8:
                self.failures.pop()

    def add_recovered_failure(self, record):
        """Record a crash reconstructed at startup (see _recover_crash)."""
        with self._lock:
            self._push_failure(record)

    def _apply_progress_state(self, data):
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return
        if self.started is None:
            # Progress arrived without a preceding execution_start.
            self.started = time.time()
            self.prompt_id = data.get("prompt_id")
        self.nodes = nodes
        running = [
            (nid, n)
            for nid, n in nodes.items()
            if isinstance(n, dict) and n.get("state") == "running"
        ]
        if running:
            nid, node = running[-1]
            self._enter_node(str(nid))
            self.node_value = _num(node.get("value"))
            self.node_max = _num(node.get("max"))

    def _finish(self, status):
        # Clear unconditionally: the run reached an end state, so a marker left
        # from it would be misread as a crash by the next startup.
        _clear_marker()
        if self.prompt_id is None and self.started is None:
            return
        self.history.appendleft(
            {
                "prompt_id": self.prompt_id,
                "status": status if not self.error else "error",
                "duration": (time.time() - self.started) if self.started else None,
                "ended": time.time(),
                "error": self.error,
            }
        )
        self.reset_run()

    # -- read side -------------------------------------------------------

    def snapshot(self):
        with self._lock:
            now = time.time()
            run = None
            if self.started is not None:
                done, total = self._node_counts()
                run = {
                    "prompt_id": self.prompt_id,
                    "elapsed": now - self.started,
                    "overall": self._overall(),
                    "nodes_done": done,
                    "nodes_total": total,
                    "node": self._node_info(now),
                }
            return {
                "state": "running" if run else "idle",
                "run": run,
                "queue_remaining": self.queue_remaining,
                "history": list(self.history),
                "failures": list(self.failures),
                "last_event": self.last_event,
                "last_event_at": self.last_event_at,
            }

    def _node_counts(self):
        if self.nodes:
            done = sum(
                1
                for n in self.nodes.values()
                if isinstance(n, dict) and n.get("state") == "finished"
            )
            return done, len(self.nodes)
        return len(self.cached), 0

    def _overall(self):
        if not self.nodes:
            return None
        fracs = []
        for node in self.nodes.values():
            if not isinstance(node, dict):
                continue
            if node.get("state") == "finished":
                fracs.append(1.0)
                continue
            mx = _num(node.get("max"))
            val = _num(node.get("value"))
            fracs.append(min(val / mx, 1.0) if mx > 0 else 0.0)
        if not fracs:
            return None
        return sum(fracs) / len(fracs)

    def _node_info(self, now):
        if self.node_id is None:
            return None
        elapsed = (now - self.node_started) if self.node_started else 0.0
        eta = None
        if self.node_value > 0 and self.node_max > self.node_value and elapsed > 0:
            per_step = elapsed / self.node_value
            eta = (self.node_max - self.node_value) * per_step
        return {
            "id": self.node_id,
            "value": self.node_value,
            "max": self.node_max,
            "elapsed": elapsed,
            "eta": eta,
        }


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


TRACKER = StatusTracker()


# ---------------------------------------------------------------------------
# Hook into the server's event stream
# ---------------------------------------------------------------------------

def _install_hook():
    """Tee every send_sync event into TRACKER, then pass it through."""
    original = PromptServer.send_sync
    if getattr(original, "_dihan_status_hook", False):
        return

    def send_sync(self, event, data, sid=None):
        try:
            TRACKER.handle(event, data)
        except Exception:
            pass  # never let the status page break execution
        return original(self, event, data, sid)

    send_sync._dihan_status_hook = True
    PromptServer.send_sync = send_sync


def _install_queue_hook():
    """Backstop for failures that never reach send_sync.

    ``PromptQueue.task_done`` runs for every prompt, attached client or not, so
    it catches the errors the event tee structurally cannot see.
    """
    try:
        from execution import PromptQueue
    except Exception:
        return

    original = PromptQueue.task_done
    if getattr(original, "_dihan_status_hook", False):
        return

    def task_done(self, *args, **kwargs):
        try:
            status = kwargs.get("status")
            if status is None and len(args) > 2:
                status = args[2]
            if status is not None and getattr(status, "status_str", None) == "error":
                TRACKER.note_unreported_failure(getattr(status, "messages", None))
        except Exception:
            pass  # never let bookkeeping break the queue
        return original(self, *args, **kwargs)

    task_done._dihan_status_hook = True
    PromptQueue.task_done = task_done


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

def _recover_crash():
    """Reconstruct a failure for a run that died without reporting.

    A marker still on disk at startup means the previous process vanished
    mid-run: OOM-killed, segfaulted, or stopped. The kernel log is the only
    place the reason survives, so it is searched here -- in a background
    thread, because the readers can take a moment and startup should not wait.
    """
    marker = _read_marker()
    if not marker:
        return
    _clear_marker()

    def work():
        kernel = _kernel_evidence(since=marker.get("started"), pid=marker.get("pid"))
        oom = bool(kernel)
        label = marker.get("node_label") or (
            "node %s" % marker.get("node_id") if marker.get("node_id") else "unknown node"
        )
        if oom:
            message = (
                "ComfyUI exited during this run and the kernel log shows memory "
                "pressure around that time. It was most likely killed for running "
                "out of memory while executing %s." % label
            )
        else:
            message = (
                "ComfyUI exited during this run without reporting an error, while "
                "executing %s. No kernel memory evidence was found, so this was "
                "more likely a manual stop, a restart, or a hard crash." % label
            )

        started = marker.get("started")
        updated = marker.get("updated")
        TRACKER.add_recovered_failure(
            {
                "kind": "process_died",
                "at": updated or time.time(),
                "prompt_id": marker.get("prompt_id"),
                "node_id": marker.get("node_id"),
                "node_label": marker.get("node_label"),
                "exception_type": "ProcessExited",
                "message": message,
                "traceback": [],
                "oom": oom,
                "elapsed": (updated - started) if (started and updated) else None,
                "progress": {"value": marker.get("value"), "max": marker.get("max")},
                "dead_pid": marker.get("pid"),
                "kernel": kernel,
                # The log tail here is the *current* process's log, which is not
                # the dead one's. Left out deliberately to avoid implying it is.
                "log_tail": [],
            }
        )
        print(
            "[dihan-nodes] recovered an unfinished run from the previous process "
            "(%s)" % ("suspected OOM kill" if oom else "cause unknown")
        )

    threading.Thread(target=work, name="dihan-crash-recovery", daemon=True).start()


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def _queue_items():
    """Return (running, pending) as light dicts, plus the running prompt graph."""
    running, pending, graphs = [], [], {}
    try:
        queue = PromptServer.instance.prompt_queue
        getter = getattr(queue, "get_current_queue_volatile", None) or queue.get_current_queue
        cur_running, cur_pending = getter()
    except Exception:
        return running, pending, graphs

    def describe(item):
        number = item[0] if len(item) > 0 else None
        prompt_id = item[1] if len(item) > 1 else None
        prompt = item[2] if len(item) > 2 else None
        if isinstance(prompt, dict) and prompt_id:
            graphs[prompt_id] = prompt
        return {
            "number": number,
            "prompt_id": prompt_id,
            "nodes": len(prompt) if isinstance(prompt, dict) else None,
        }

    for item in cur_running or []:
        try:
            running.append(describe(item))
        except Exception:
            pass
    for item in cur_pending or []:
        try:
            pending.append(describe(item))
        except Exception:
            pass
    pending.sort(key=lambda d: (d["number"] is None, d["number"]))
    return running, pending, graphs


def _node_label(graph, node_id):
    """Resolve a node id to a human label using the running prompt graph."""
    if not isinstance(graph, dict):
        return None
    node = graph.get(str(node_id))
    if not isinstance(node, dict):
        return None
    title = (node.get("_meta") or {}).get("title")
    class_type = node.get("class_type")
    if title and class_type and title != class_type:
        return "%s (%s)" % (title, class_type)
    return title or class_type


def _system_info():
    info = {"devices": [], "ram_total": None, "ram_free": None}
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total"] = mem.total
        info["ram_free"] = mem.available
    except Exception:
        pass

    try:
        import comfy.model_management as mm

        device = mm.get_torch_device()
        info["devices"].append(
            {
                "name": mm.get_torch_device_name(device),
                "type": device.type,
                "index": getattr(device, "index", None),
                "vram_total": mm.get_total_memory(device),
                "vram_free": mm.get_free_memory(device),
                "torch_vram_total": mm.get_total_memory(device, torch_total_too=True)[1],
                "torch_vram_free": mm.get_free_memory(device, torch_free_too=True)[1],
            }
        )
    except Exception:
        pass
    return info


def _version_info():
    import sys

    out = {
        "python_version": sys.version.split()[0],
        "comfyui_version": None,
        "pytorch_version": None,
        "uptime": time.time() - _BOOT_TIME,
    }
    try:
        from comfyui_version import __version__ as comfy_version

        out["comfyui_version"] = comfy_version
    except Exception:
        pass
    try:
        import torch

        out["pytorch_version"] = torch.version.__version__
    except Exception:
        pass
    return out


def _build_snapshot():
    snap = TRACKER.snapshot()
    running, pending, graphs = _queue_items()

    run = snap.get("run")
    if run:
        graph = graphs.get(run.get("prompt_id"))
        node = run.get("node")
        if node and node.get("id"):
            node["label"] = _node_label(graph, node["id"])
        if not run.get("nodes_total") and isinstance(graph, dict):
            run["nodes_total"] = len(graph)

    snap["queue"] = {
        "running": running,
        "pending": pending,
        "remaining": snap.pop("queue_remaining", 0),
    }
    snap["system"] = _system_info()
    snap["server"] = _version_info()
    snap["now"] = time.time()
    return snap


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _taken_paths():
    """Paths already declared on the shared RouteTableDef by core or other nodes."""
    try:
        items = getattr(PromptServer.instance.routes, "_items", [])
        return {getattr(i, "path", None) for i in items if getattr(i, "method", "") == "GET"}
    except Exception:
        return set()


def _pick_base():
    taken = _taken_paths()
    for base in ("/status", "/dihan-status"):
        if base not in taken:
            return base
    return None


def _server_url(base):
    try:
        from comfy.cli_args import args

        host = args.listen
        if host in ("0.0.0.0", "::", ""):
            host = "<your-ip>"
        return "http://%s:%s%s" % (host, args.port, base)
    except Exception:
        return base


def _register_routes():
    routes = PromptServer.instance.routes
    base = _pick_base()
    if base is None:
        print("[dihan-nodes] status page disabled: /status already in use")
        return

    @routes.get(base)
    async def status_page(request):
        try:
            with open(PAGE_PATH, "r", encoding="utf-8") as handle:
                body = handle.read()
        except OSError as exc:
            return web.Response(status=500, text="status.html missing: %s" % exc)
        return web.Response(
            text=body,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @routes.get(base + "/api/snapshot")
    async def status_snapshot(request):
        return web.json_response(_build_snapshot(), headers={"Cache-Control": "no-store"})

    @routes.get(base + "/api/failures")
    async def status_failures(request):
        """Full failure records. Fetched on demand -- these carry tracebacks
        and log tails, so they are far too heavy for the polling snapshot."""
        try:
            limit = int(request.rel_url.query.get("limit", 10))
        except ValueError:
            limit = 10
        limit = max(1, min(limit, MAX_FAILURES))
        return web.json_response(
            {"failures": _read_failures(limit)}, headers={"Cache-Control": "no-store"}
        )

    @routes.get(base + "/api/logs")
    async def status_logs(request):
        try:
            tail = int(request.rel_url.query.get("tail", LOG_TAIL_DEFAULT))
        except ValueError:
            tail = LOG_TAIL_DEFAULT
        tail = max(1, min(tail, 1000))
        entries = []
        try:
            from app.logger import get_logs

            entries = [
                {"t": entry.get("t"), "m": entry.get("m")}
                for entry in list(get_logs())[-tail:]
            ]
        except Exception as exc:
            return web.json_response({"entries": [], "error": str(exc)})
        return web.json_response(
            {"entries": entries}, headers={"Cache-Control": "no-store"}
        )

    print("[dihan-nodes] status page: %s" % _server_url(base))
    return base


def setup():
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        _install_hook()
        _install_queue_hook()
        _register_routes()
    except Exception as exc:  # pragma: no cover
        print("[dihan-nodes] status page disabled: %s" % exc)

    # Independent of the page itself: even if route registration failed above,
    # an unfinished run from the previous process is still worth recording.
    try:
        _recover_crash()
    except Exception as exc:  # pragma: no cover
        print("[dihan-nodes] crash recovery skipped: %s" % exc)
