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
"""

import os
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

HISTORY_LEN = 12
LOG_TAIL_DEFAULT = 200

_BOOT_TIME = time.time()


class StatusTracker:
    """Rolling snapshot of what the executor is currently doing."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset_run()
        self.queue_remaining = 0
        self.history = deque(maxlen=HISTORY_LEN)
        self.last_event = None
        self.last_event_at = None

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
        _register_routes()
    except Exception as exc:  # pragma: no cover
        print("[dihan-nodes] status page disabled: %s" % exc)
