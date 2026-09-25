"""Deterministic checker for MiniMax H3 prompts (Ref2VA and the base modes).

Everything here is mechanical: it catches rule breaks that can be detected without
understanding the scene, so a prompt never fails a render on format. Scene quality
is the writer model's job.

Errors   -> a hard rule is broken; the writer node sends these back to the model.
Warnings -> probably a problem; sent back only in strict mode.

Ref2VA follows MiniMax's full-reference format (skills/h3-prompt-writing/references/
ref-en.txt): subject_definitions, summary, retention_analysis, detailed_description,
overall_soundscape, non_diegetic_music. The base modes (T2VA / I2VA / FL2VA / L2VA)
use integrated_multimodal_description, overall_soundscape, non_diegetic_music.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REF_MODE = "Ref2VA"
BASE_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")
MODES = (REF_MODE,) + BASE_MODES
BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REF_FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
              "overall_soundscape", "non_diegetic_music")
DASH = "—"

I2VA_LINE = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."

TASK_TYPES = ("reference generation", "keyframe completion", "video editing", "video continuation",
              "audio reuse", "audio reference")
VISUAL_MARKERS = ("fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference")
AUDIO_MARKERS = ("fully_copy", "partially_copy", "reference", "weak_reference")


def alignment_line(mode: str, duration: float, final_shot: int) -> str | None:
    if mode == "I2VA":
        return I2VA_LINE
    if mode == "FL2VA":
        return (
            f"How the reference pictures align with the target video {DASH} Picture 1 (from Shot 1) aligns with the "
            f"0.00-second mark of the target video; Picture 2 (from Shot {final_shot}) aligns with the "
            f"{duration:.2f}-second mark of the target video."
        )
    if mode == "L2VA":
        return (
            f"How the reference pictures align with the target video {DASH} <Picture 1> (from [Shot {final_shot}]) "
            f"aligns with the {duration:.2f}-second mark of the target video."
        )
    return None


# ---------------------------------------------------------------- extraction

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\n(.*?)```", re.S)
_START_RE = re.compile(
    r"^[ \t]*(How the reference pictures|For the target video|integrated_multimodal_description\s*:|subject_definitions\s*:)",
    re.M,
)
_PROMPT_MARKERS = ("integrated_multimodal_description", "subject_definitions", "detailed_description")


def extract_prompt(raw: str) -> str:
    """Pull the prompt out of a model reply: drop reasoning, fences and chatter."""
    text = _THINK_RE.sub("", raw or "").strip()
    fences = _FENCE_RE.findall(text)
    if fences:
        for block in fences:
            if any(m in block for m in _PROMPT_MARKERS):
                text = block.strip()
                break
        else:
            text = fences[0].strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = _START_RE.search(text)
    if m:
        text = text[m.start():]
    # drop chatter after the music section (its value may sit on the header line or the next line)
    mm = re.search(r"^[ \t]*non_diegetic_music\s*:[ \t]*(.*)$", text, re.M)
    if mm:
        if mm.group(1).strip():
            text = text[: mm.end()]
        else:
            rest = text[mm.end():]
            nxt = re.match(r"\s*\n[ \t]*(\S[^\n]*)", rest)
            if nxt:
                text = text[: mm.end() + nxt.end()]
    return text.strip().strip("`").strip()


def looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and not any(m in t for m in _PROMPT_MARKERS) and t.endswith("?") and len(t) < 400


def infer_mode_and_duration(text: str) -> tuple[str, float]:
    """Best-effort read of the mode (and a base-mode duration) from an existing prompt."""
    t = (text or "").strip()
    if re.search(r"^\s*subject_definitions\s*:", t, re.M) or re.search(r"^\s*detailed_description\s*:", t, re.M):
        return REF_MODE, 0.0
    if t.startswith("For the target video"):
        return "I2VA", 0.0
    first = t.split("\n")[0]
    if t.startswith("How the reference pictures"):
        m = re.search(r"aligns with the (\d+(?:\.\d+)?)-second mark of the target video\.\s*$", first)
        return ("FL2VA" if "Picture 2" in first else "L2VA"), (float(m.group(1)) if m else 0.0)
    return "T2VA", 0.0


# ---------------------------------------------------------------- parsing

@dataclass
class Sections:
    pre: str                      # text before the first section header
    values: dict                  # name -> raw value (may span lines)
    order: list
    problems: list = field(default_factory=list)


def split_sections(text: str, names: tuple) -> Sections:
    rx = re.compile(r"^[ \t]*(" + "|".join(names) + r")[ \t]*:", re.M)
    hits = list(rx.finditer(text))
    order = [h.group(1) for h in hits]
    problems = []
    for n in names:
        c = order.count(n)
        if c == 0:
            problems.append(f"Missing section `{n}:`.")
        elif c > 1:
            problems.append(f"Section `{n}:` appears {c} times; it must appear once.")
    values = {}
    for i, h in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        values.setdefault(h.group(1), text[h.end(): end])
    pre = text[: hits[0].start()] if hits else text
    return Sections(pre.strip(), values, order, problems)


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def lines_of(s: str) -> list[str]:
    return [squash(x) for x in (s or "").split("\n") if x.strip()]


# ---------------------------------------------------------------- helpers

STOP = set(
    """a an the and or but so to of in on at by for with from into onto over under up down out off as is are was were be been
    being it its this that these those i you he she we they me him her us them my your his our their mine yours hers ours
    theirs what which who whom whose when where why how not no do does did done have has had will would can could should
    shall may might must just then than there here very too also all any some one two get got go going gonna let lets
    let's i'm you're it's don't can't won't im youre dont cant wont yes oh okay ok hey""".split()
)


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", (s or "").lower())


def content_words(s: str) -> set[str]:
    return {w for w in words(s) if w not in STOP and len(w) >= 3}


def sentences(s: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", s or "") if x.strip()]


D_RE = re.compile(r"<d>(.*?)</d>", re.S)
SHOT_RE = re.compile(r"\[Shot (\d+)\]")
TS_RE = re.compile(r"^At (\d{2}):(\d{2})\.(\d{3}),")
ANY_TS_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\.\d{1,3})?\b")
LABEL_RE = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")
SPEAKER_RE = re.compile(r"\(S\d+(?:\s*,\s*S\d+)*\)")
BASE_TAGS = {"<d>", "</d>", "<scenetrans>", "<cutoff>", "<Picture 1>", "<Picture 2>"}
REF_TAGS = {"<d>", "</d>", "<scenetrans>", "<cutoff>"}

STYLE_WORDS = (
    "cinematic", "live-action", "live action", "animated", "animation", "3d", "cg", "cgi", "claymation", "watercolor",
    "watercolour", "vintage", "film", "commercial", "documentary", "music-video", "music video", "anime", "stop-motion",
    "photoreal", "realistic", "illustrated", "pixel", "noir", "style", "painterly", "hand-drawn", "cel-shaded", "sitcom",
)

PLANNING_RE = re.compile(
    r"\b(plausible (earlier|preceding|starting)|inferred (starting|earlier|preceding)|compatible (preceding|earlier)|"
    r"reference image shows|reference picture shows|as described above|as mentioned above|the user (requested|asked|wants|wanted)|"
    r"last-frame state|first-frame state|first-frame anchor|action onset|narrowing differences|"
    r"gradual convergence|last-frame landing)\b",
    re.I,
)
STILLNESS_RE = re.compile(
    r"(no (facial|face|expression|mouth|lip) (changes?|movement)|no changes? (to|in) (her|his|their|the) (face|expression|mouth)|"
    r"remains? (in the same position|completely still|perfectly still|motionless|frozen)|stays? (completely |perfectly )?still|"
    r"(face|expression) (stays|remains) (unchanged|fixed|the same|frozen)|without any (movement|changes)|\bno changes\b)",
    re.I,
)
SOUND_SILENCE_RE = re.compile(
    r"(no background (noise|sound)|no (echo|reverb|ambien\w+)|clean audio|nothing distracting|complete silence|"
    r"total silence|dead silence|pure silence|digital silence|silent background|absolute silence|no other sounds?)",
    re.I,
)
SOUND_RECORDING_RE = re.compile(
    r"(close-?mik(ed|e)|close-?mic(ed|rophone)?|well[- ]recorded|studio[- ]quality|studio recording|crisp (voice|audio|vocals?)|"
    r"clean (voice|vocals?)|voice is (clear|clean|intimate|crisp)|free of echo|broadcast[- ]quality)",
    re.I,
)
SOUND_MUSIC_RE = re.compile(r"\b(music|musical|melody|lyrics?|song|singing|soundtrack|score|beat drops?)\b", re.I)
MUSIC_VAGUE_RE = re.compile(r"\b(emotional|epic|sad|happy|uplifting|inspiring|inspirational|dramatic|beautiful|moody|heartfelt|feel-?good)\b", re.I)
REPORTED_SPEECH_RE = re.compile(
    r"\b(tells|told|explains|explained|greets|greeted|introduces|introduced|announces|announced|discuss(es|ed)?|"
    r"mentions|mentioned|replies that|says that|asks (him|her|them|if|whether|where|what|why|how)|asked (him|her|them))\b",
    re.I,
)
CAPS_MOTION_RE = re.compile(
    r"\b(Push In|Pull Out|Zoom In|Zoom Out|Pan Left|Pan Right|Truck Left|Truck Right|Tilt Up|Tilt Down|Pedestal Up|"
    r"Pedestal Down|Arc Shot|Tracking Shot|Static Shot|Shake Slightly|Shake Strongly|Roll Clockwise|Roll Counterclockwise)\b"
)
ALLOWED_CAM_PHRASES = re.compile(
    r"(with small amplitude|with large amplitude|at slow speed|at fast speed|shakes? slightly|shakes? strongly)", re.I
)
CAM_VERBS = {
    "camera", "pan", "pans", "panning", "tilt", "tilts", "tilting", "zoom", "zooms", "zooming", "push", "pushes", "pushing",
    "pull", "pulls", "pulling", "truck", "trucks", "trucking", "pedestal", "pedestals", "arc", "arcs", "arcing", "orbit",
    "orbits", "orbiting", "dolly", "dollies", "dollying", "track", "tracks", "roll", "rolls", "rolling", "crane", "cranes",
}
AMBIGUOUS_CAM = {"track", "tracks", "roll", "rolls", "arc", "arcs", "push", "pushes", "pull", "pulls"}
CAM_BANNED = {
    "slow", "slowly", "slower", "gentle", "gently", "subtle", "subtly", "slight", "slightly", "quick", "quickly", "fast",
    "faster", "rapid", "rapidly", "swift", "swiftly", "smooth", "smoothly", "steady", "steadily", "gradual", "gradually",
    "restrained", "leisurely", "amplitude", "speed", "very", "barely", "imperceptibly", "lazily", "briskly", "sharply",
}
MOUTH_RE = re.compile(r"\b(lips?|mouth|jaw)\b", re.I)
CLOSE_RE = re.compile(r"\b(closes?|closed|closing|presses|pressed|press|falls silent|stops speaking|settles|lips?|mouth)\b", re.I)
CITATION_RE = re.compile(r"(\[\d+\]|【|\bcite(turn|:)|\(source[: ]|\[source)", re.I)
BASE_ALIGN_RE = re.compile(r"(aligns with the [\d.]+-second mark|is fully referenced\.|How the reference pictures align)", re.I)


# ---------------------------------------------------------------- result

@dataclass
class LintResult:
    text: str
    mode: str
    duration: float
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    fixes: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def report(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        dur = f"{self.duration:.2f}s" if self.duration else "no duration"
        lines = [f"{head} - {self.mode}, {dur} - {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        if self.stats:
            lines.append("stats: " + ", ".join(f"{k}={v}" for k, v in self.stats.items()))
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"warn:  {w}" for w in self.warnings]
        lines += [f"auto-fixed: {f}" for f in self.fixes]
        return "\n".join(lines)


# ---------------------------------------------------------------- shared body checks

def _shot_checks(desc: str, duration: float, E: list, W: list):
    """desc: squashed description text containing the [Shot N] markers."""
    shots = [int(n) for n in SHOT_RE.findall(desc)]
    if not shots:
        E.append("The description has no [Shot 1] marker.")
    elif shots != list(range(1, len(shots) + 1)):
        E.append(f"Shot numbers must run 1, 2, 3... in order; found {shots}.")
    segs = list(SHOT_RE.finditer(desc))
    cut_times, last_t = [], 0.0
    for i, m in enumerate(segs):
        seg = desc[m.end(): segs[i + 1].start() if i + 1 < len(segs) else len(desc)].strip()
        n = int(m.group(1))
        tm = TS_RE.match(seg)
        if n == 1:
            if re.match(r"^At\s+\d", seg):
                E.append("[Shot 1] must not carry a timestamp.")
            continue
        if not tm:
            E.append(f"[Shot {n}] must open with a cut time like `At 00:03.500,`.")
            continue
        t = int(tm.group(1)) * 60 + int(tm.group(2)) + int(tm.group(3)) / 1000
        stamp = tm.group(0)[3:-1]
        if t <= last_t:
            E.append(f"[Shot {n}] cut time {stamp} must be later than the previous shot.")
        if duration and t >= duration:
            E.append(f"[Shot {n}] cut time {stamp} is at or beyond the {duration:.2f}s duration.")
        last_t = t
        cut_times.append(stamp)
    if len(shots) > 1 and not duration:
        E.append("Several shots need a duration for exact cut times; set duration or write one continuous shot.")
    stray = [t for t in ANY_TS_RE.findall(desc) if t not in cut_times]
    if stray:
        E.append(f"Timestamps inside a shot are not allowed (found {', '.join(stray[:3])}); only cut times may use them.")
    if re.search(r"\[\s*\d+(\.\d+)?\s*s?\s*[-–]\s*\d+(\.\d+)?\s*s\s*\]", desc) or re.search(
        r"\bat (one|two|three|four|five|six|seven|eight|nine|ten|\d+) seconds?\b", desc, re.I
    ):
        E.append("Time ranges like [0s-2s] or 'at five seconds' are not allowed; describe events in order.")
    return shots, segs


def _dialogue_checks(desc: str, sound: str, music: str, segs, duration: float, E: list, W: list, ref_mode: bool,
                     summary: str = ""):
    dlines = []
    for m in D_RE.finditer(desc):
        inner = m.group(1).strip()
        lm = re.match(r"^\[([A-Za-z][^\]]{0,30})\]\s*(.+)$", inner, re.S)
        if not lm:
            E.append(f"<d> block must start with a language tag like [English]: <d>{inner[:40]}</d>")
            line = inner
        else:
            line = lm.group(2).strip()
        dlines.append((m, line))
    if desc.count("<d>") != desc.count("</d>"):
        E.append("Unbalanced <d> ... </d> tags.")

    outside_desc = D_RE.sub(" §. ", desc)  # keep sentence boundaries around lines
    total_words = 0
    for m, line in dlines:
        total_words += len(words(line))
        before = desc[max(0, m.start() - 320): m.start()]
        after = desc[m.end(): m.end() + 320]
        after_first = (sentences(after) or [""])[0]
        is_vo = "off-screen voiceover" in before[-160:].lower()
        if not SPEAKER_RE.search(before[-260:]):
            W.append(f"No speaker ID (S1, S2...) just before: <d>{line[:40]}</d>")
        if is_vo:
            if not re.search(r"lips?\s+(remain|remains|stay|stays|are|is|kept)\b[^.]{0,40}closed|closed lips|mouth (remains|stays) closed", after[:220], re.I):
                E.append(f"Voiceover line must be followed by the on-screen character's lips staying closed: <d>{line[:40]}</d>")
        elif not re.search(r"off-screen", before[-160:], re.I):
            if not (MOUTH_RE.search(D_RE.sub(" ", before[-220:])) or MOUTH_RE.search(D_RE.sub(" ", after[:220]))):
                E.append(f"On-screen line has no visible mouth movement described (lips/mouth/jaw): <d>{line[:40]}</d>")
            if not (CLOSE_RE.search(after_first) or after.lstrip().startswith(("<scenetrans>", "<cutoff>"))):
                W.append(f"After </d>, describe the speaker returning to a non-speaking state (e.g. closes her lips): <d>{line[:40]}</d>")
        cw = content_words(line)
        norm_line = " ".join(words(line))
        outside = [(s, "description") for s in sentences(outside_desc)]
        outside += [(s, "overall_soundscape") for s in sentences(sound)] + [(s, "non_diegetic_music") for s in sentences(music)]
        outside += [(s, "summary") for s in sentences(summary)]
        for s, where in outside:
            overlap = cw & content_words(s)
            exact = len(norm_line.split()) >= 3 and norm_line in " ".join(words(s))
            if exact or (len(cw) >= 2 and len(overlap) >= 2 and len(overlap) / len(cw) >= 0.6):
                E.append(f"Speech restated outside its <d> block ({where}): \"{s[:110]}\" repeats <d>{line[:40]}</d>. Rewrite that sentence to describe only what is seen/heard.")
                break

    if re.search(r"\b(voice-?over|voice over)\b", desc, re.I) and "says in an off-screen voiceover" not in desc:
        E.append("Voiceover must use the exact phrase 'says in an off-screen voiceover'.")
    rs = REPORTED_SPEECH_RE.search(outside_desc + " " + summary)
    if rs and dlines:
        W.append(f"Reported-speech wording '{rs.group(0)}' may restate dialogue; describe delivery, not content.")

    ids = []
    for grp in SPEAKER_RE.findall(desc):
        for n in re.findall(r"S(\d+)", grp):
            if int(n) not in ids:
                ids.append(int(n))
    if ids and ids != list(range(1, len(ids) + 1)):
        E.append(f"Speaker IDs must be numbered by first vocal event (S1, S2, ...); first-appearance order was {['S%d' % i for i in ids]}.")

    if dlines:
        for i, m in enumerate(segs):
            seg = desc[m.start(): segs[i + 1].start() if i + 1 < len(segs) else len(desc)]
            if "<d>" in seg:
                sm = STILLNESS_RE.search(D_RE.sub(" ", seg))
                if sm:
                    E.append(f"Stillness wording in a speaking shot ('{sm.group(0)}') can freeze the mouth; keep preservation to identity, wardrobe and setting.")
        need = total_words / 3.0
        if duration and need > duration:
            E.append(f"Dialogue is {total_words} words (~{need:.1f}s at 3 words/s) but the clip is {duration:.2f}s. Shorten the line or lengthen the clip.")
        if (duration or 6.0) <= 6.0:
            first_shot = segs[0].end() if segs else 0
            n_before = len(sentences(desc[first_shot: dlines[0][0].start()]))
            if n_before > 2:
                W.append(f"On a short clip the first line should start in the first beat; {n_before} sentences come before it.")
    return outside_desc, total_words, bool(dlines)


def _camera_checks(desc: str, outside_desc: str, E: list, W: list):
    for m in CAPS_MOTION_RE.finditer(desc):
        E.append(f"Camera motion written as a capitalised keyword ('{m.group(0)}'); write it as natural lowercase English.")
    for s in sentences(outside_desc):
        toks = words(ALLOWED_CAM_PHRASES.sub(" ", s))
        if not ({"camera", "lens"} & set(toks)):
            continue  # heads tilt and people turn; only police sentences about the camera
        for i, t in enumerate(toks):
            if t in CAM_VERBS:
                bad = [w for w in toks[max(0, i - 2): i + 4] if w in CAM_BANNED]
                if bad:
                    E.append(f"Camera speed/amplitude in free wording ('{bad[0]}' in: \"{s[:90]}\"). Use only: with small amplitude, with large amplitude, at slow speed, at fast speed.")
                    break
        if re.search(r"\b(medium amplitude|normal speed|moderate speed|medium speed)\b", s, re.I):
            E.append("Medium amplitude / normal speed need no wording; remove it.")
    if re.search(r"\bdoll(y|ies|ying)\b", desc, re.I):
        W.append("'dolly' is not an H3 motion term; use push in / pull out (forward/back) or truck left/right (sideways).")


def _sound_checks(sound: str, music: str, has_dialogue: bool, silent_ok: bool, E: list, W: list):
    if not sound:
        E.append("overall_soundscape is empty.")
    elif sound.strip().upper().rstrip(".") == "N/A":
        if not silent_ok:
            W.append("overall_soundscape: N/A means total silence and exposes the opening click; use a soft ambience floor unless silence was requested.")
    else:
        for rx, msg in ((SOUND_SILENCE_RE, "asks for silence ('{}'); give a soft ambience floor instead."),
                        (SOUND_RECORDING_RE, "contains a recording note ('{}'); voice character belongs beside the speaker."),
                        (SOUND_MUSIC_RE, "mentions music/lyrics ('{}'); music goes in non_diegetic_music or the description.")):
            m = rx.search(sound)
            if m:
                E.append("overall_soundscape " + msg.format(m.group(0)))
        if "<d>" in sound:
            E.append("overall_soundscape contains dialogue.")
        if len(sentences(sound)) > 4:
            W.append("overall_soundscape should be one to four sentences.")
    if not music:
        E.append("non_diegetic_music is empty (use N/A for no score).")
    elif music.strip().upper().rstrip(".") != "N/A":
        m = MUSIC_VAGUE_RE.search(music)
        if m:
            W.append(f"non_diegetic_music uses a mood word ('{m.group(0)}'); describe instruments, tempo and dynamics instead.")
        if "<d>" in music:
            E.append("non_diegetic_music contains dialogue.")
        if len(sentences(music)) > 3:
            W.append("non_diegetic_music should be one to three sentences.")
    elif has_dialogue:
        W.append("Talking shot with non_diegetic_music: N/A; a quiet pad helps mask the opening click.")


def _common_text_checks(text: str, E: list):
    for m in PLANNING_RE.finditer(text):
        E.append(f"Planning language reached the prompt: '{m.group(0)}'. Describe the actual pose/action instead.")
    if CITATION_RE.search(text):
        E.append("Citation or reference markers are not allowed inside the prompt.")


# ---------------------------------------------------------------- Ref2VA

def _ref_rebuild(sec: Sections) -> tuple[str, dict]:
    v = sec.values
    defs = squash(v.get("subject_definitions", ""))
    defs_lines = [x.strip() for x in re.split(r"\s+(?=<(?:Subject|Picture|Video|Audio) \d+> (?:is|are)\b)", defs) if x.strip()]
    ret = squash(v.get("retention_analysis", ""))
    ret_lines = [x.strip() for x in re.split(r"\s+(?=<(?:Subject|Picture|Video|Audio) \d+>(?: \([^)]*\))?:)", ret) if x.strip()]
    desc = squash(v.get("detailed_description", ""))
    desc = re.sub(r"\s*(\[Shot \d+\])", r"\n\1", desc).strip()
    parts = {
        "subject_definitions": "\n".join(defs_lines),
        "summary": squash(v.get("summary", "")),
        "retention_analysis": "\n".join(ret_lines),
        "detailed_description": desc,
        "overall_soundscape": squash(v.get("overall_soundscape", "")),
        "non_diegetic_music": squash(v.get("non_diegetic_music", "")),
    }
    return "\n\n".join(f"{k}:\n{parts[k]}" for k in REF_FIELDS), parts


def _lint_ref(res: LintResult, text: str, autofix: bool, n_pictures: int | None, silent_ok: bool):
    E, W, duration = res.errors, res.warnings, res.duration
    if re.search(r"^\s*integrated_multimodal_description\s*:", text, re.M):
        E.append("Ref2VA uses `detailed_description:`, not the base-mode `integrated_multimodal_description:`.")
    sec = split_sections(text, REF_FIELDS)
    if sec.problems:
        E.extend(sec.problems)
        return
    if sec.order != list(REF_FIELDS):
        msg = "Sections are out of order; required: " + ", ".join(REF_FIELDS) + "."
        if autofix:
            res.fixes.append("reordered sections to the official Ref2VA order")
        else:
            E.append(msg)
    if sec.pre:
        if BASE_ALIGN_RE.search(sec.pre):
            (res.fixes.append("removed a base-mode alignment line (it makes H3 treat references as keyframes)")
             if autofix else E.append("Ref2VA must not have a base-mode alignment line; it makes H3 treat references as keyframes."))
        elif autofix:
            res.fixes.append("removed text before subject_definitions")
        else:
            E.append("Nothing may come before `subject_definitions:`.")

    rebuilt, P = _ref_rebuild(sec)
    if autofix:
        if rebuilt != text and not res.fixes:
            res.fixes.append("normalised section layout (headers on their own line, one line per label/shot)")
        res.text = text = rebuilt
    elif rebuilt != text:
        W.append("Layout differs from the official format (header on its own line, one line per label or shot, one blank line between sections).")

    defs_lines, ret_lines = lines_of(P["subject_definitions"]), lines_of(P["retention_analysis"])
    summary, desc_full = P["summary"], P["detailed_description"]
    sound, music = P["overall_soundscape"], P["non_diegetic_music"]
    desc = squash(desc_full)

    # ---- subject_definitions
    defined, cited_in_defs, kinds = [], set(), {}
    for ln in defs_lines:
        m = re.match(r"^<(Subject|Picture|Video|Audio) (\d+)> (is|are)\b", ln)
        if not m:
            E.append(f"subject_definitions line must start with a label and 'is': \"{ln[:80]}\"")
            continue
        lab = f"<{m.group(1)} {m.group(2)}>"
        if lab in defined:
            E.append(f"{lab} is defined twice.")
        defined.append(lab)
        kinds[lab] = m.group(1)
        cited_in_defs.update(f"<{a} {b}>" for a, b in LABEL_RE.findall(ln))
        if m.group(1) == "Audio" and SPEAKER_RE.search(ln) is None and re.search(r"voice|timbre|speaker", ln, re.I):
            W.append(f"{lab} is a voice reference; name the subject and speaker ID it maps to, e.g. '<Subject 1> (S1)'.")
    if not defined:
        E.append("subject_definitions defines no labels.")
    subj_nums = sorted(int(l.split()[1][:-1]) for l in defined if kinds.get(l) == "Subject")
    if subj_nums and subj_nums != list(range(1, len(subj_nums) + 1)):
        W.append(f"<Subject N> numbers should run 1, 2, 3...; found {subj_nums}.")
    known = set(defined) | cited_in_defs

    # ---- pictures vs connected references
    all_labels = {f"<{a} {b}>" for a, b in LABEL_RE.findall(text)}
    pics_used = sorted({int(l.split()[1][:-1]) for l in all_labels if l.startswith("<Picture")})
    if n_pictures is not None:
        allowed = n_pictures if isinstance(n_pictures, int) else None
        if isinstance(n_pictures, (list, tuple, set)):
            bad = [p for p in pics_used if p not in n_pictures]
            unused = [p for p in n_pictures if f"<Picture {p}>" not in cited_in_defs]
        else:
            bad = [p for p in pics_used if p > allowed]
            unused = [p for p in range(1, allowed + 1) if f"<Picture {p}>" not in cited_in_defs]
        if bad:
            E.append(f"<Picture {bad[0]}> is used but no such reference image was supplied.")
        for p in unused:
            W.append(f"Reference image <Picture {p}> is connected but never cited in subject_definitions.")

    # ---- unresolved labels
    for lab in sorted(all_labels - known):
        E.append(f"{lab} is used but never defined in subject_definitions.")

    # ---- summary
    sm = re.match(r"^\[([^\]]+)\]\s+\S", summary)
    tasks = []
    if not sm:
        E.append("summary must start with a task-type prefix such as `[reference generation] `.")
    else:
        tasks = [t.strip() for t in sm.group(1).split("+")]
        for t in tasks:
            if t not in TASK_TYPES:
                E.append(f"Unknown summary task type '[{t}]'; use: {', '.join(TASK_TYPES)}.")
        if len(set(tasks)) != len(tasks):
            E.append("summary repeats a task type.")
    for lab in sorted({f"<{a} {b}>" for a, b in LABEL_RE.findall(summary)} - known):
        E.append(f"summary introduces {lab}, which is not defined in subject_definitions.")
    if "<d>" in summary:
        E.append("summary must not contain dialogue.")
    for l in defined:
        if kinds[l] == "Picture":
            ln = next(x for x in defs_lines if x.startswith(l))
            if not re.search(r"(frame|keyframe|anchor|storyboard|composition|shot-planning|planning|layout reference|blocking)", ln, re.I):
                W.append(f"{l} has its own line but is not a frame, composition or storyboard anchor; if it only defines a person, place or style, make it a <Subject N> that cites {l}.")
    frame_pics = [l for l in defined if kinds[l] == "Picture" and re.search(
        r"(first frame|keyframe|last frame|final frame)", next(x for x in defs_lines if x.startswith(l)), re.I)]
    if frame_pics and "keyframe completion" not in tasks and tasks:
        W.append(f"{frame_pics[0]} is defined as a concrete frame; add 'keyframe completion' to the summary task types.")
    if "keyframe completion" in tasks and not frame_pics:
        W.append("summary says 'keyframe completion' but no <Picture N> is defined as a first/key/last frame.")
    if any(k == "Audio" for k in kinds.values()) and tasks and not any(t.startswith("audio") for t in tasks):
        W.append("An <Audio N> is defined; add 'audio reference' or 'audio reuse' to the summary task types.")

    # ---- retention_analysis
    seen = []
    for ln in ret_lines:
        mv = re.match(r"^<(Subject|Picture|Video) (\d+)> \(([^)]*)\): (\w+) - \S", ln)
        ma = re.match(r"^<Audio (\d+)>(?: \([^)]*\))?: (\w+) - \S", ln)
        if mv:
            lab, marker, allowed_m = f"<{mv.group(1)} {mv.group(2)}>", mv.group(4), VISUAL_MARKERS
        elif ma:
            lab, marker, allowed_m = f"<Audio {ma.group(1)}>", ma.group(2), AUDIO_MARKERS
        else:
            E.append(f"retention_analysis line format: `<Subject 1> (appears in [Shot 1]): fully_preserved - reason` / `<Audio 1>: reference - reason`. Got: \"{ln[:80]}\"")
            continue
        if marker not in allowed_m:
            E.append(f"{lab} has marker '{marker}'; allowed: {', '.join(allowed_m)}.")
        if lab not in defined:
            (W if lab in cited_in_defs else E).append(
                f"retention_analysis lists {lab}, which has no line of its own in subject_definitions.")
        seen.append(lab)
    for lab in defined:
        if lab not in seen:
            E.append(f"retention_analysis has no line for {lab}.")
    if SPEAKER_RE.search(P["retention_analysis"]):
        E.append("Speaker IDs like (S1) must not appear in retention_analysis.")

    # ---- detailed_description
    first_shot = desc.find("[Shot 1]")
    opener = desc[:first_shot].strip() if first_shot >= 0 else ""
    if first_shot >= 0 and not opener:
        E.append("detailed_description must open with one or two style sentences before [Shot 1].")
    elif opener and not any(s in opener.lower() for s in STYLE_WORDS):
        W.append("The line before [Shot 1] should set the visual style (e.g. 'The target video is live-action and cinematic, ...').")
    if len(sentences(opener)) > 3:
        W.append("The style opening should be one or two sentences.")
    shots, segs = _shot_checks(desc, duration, E, W)
    for lab in defined:
        if kinds[lab] == "Subject" and lab not in desc:
            W.append(f"{lab} is defined but never appears in detailed_description.")
    if BASE_ALIGN_RE.search(desc):
        E.append("Ref2VA must not use base-mode alignment wording; it makes H3 treat references as keyframes.")

    for tag in set(re.findall(r"<[^<>]{1,40}>", text)):
        if tag not in REF_TAGS and not LABEL_RE.fullmatch(tag):
            E.append(f"Tag {tag} is not an H3 tag.")

    outside_desc, dwords, has_d = _dialogue_checks(desc, sound, music, segs, duration, E, W, True, summary)
    _camera_checks(desc, outside_desc, E, W)
    _sound_checks(sound, music, has_d, silent_ok, E, W)
    _common_text_checks(desc + " " + sound + " " + music, E)
    if "<d>" in P["subject_definitions"] or "<d>" in P["retention_analysis"]:
        E.append("Dialogue belongs only in detailed_description.")

    n_words = len(words(D_RE.sub(" ", desc)))
    res.stats = {"shots": len(shots) or 1, "labels": len(defined), "description_words": n_words, "dialogue_words": dwords}
    if n_words < 150:
        W.append(f"detailed_description is {n_words} words; MiniMax's guide targets roughly 350-500 for generation. Add concrete visual detail (appearance, position, light, sound), not more actions.")
    elif n_words > 700:
        W.append(f"detailed_description is {n_words} words; MiniMax's guide targets roughly 350-500.")


# ---------------------------------------------------------------- base modes

def _lint_base(res: LintResult, text: str, autofix: bool, silent_ok: bool):
    E, W, mode, duration = res.errors, res.warnings, res.mode, res.duration
    sec = split_sections(text, BASE_FIELDS)
    if sec.problems:
        E.extend(sec.problems)
        return
    if sec.order != list(BASE_FIELDS):
        E.append("Fields are out of order; required order is integrated_multimodal_description, overall_soundscape, non_diegetic_music.")
        return
    raw_desc = sec.values["integrated_multimodal_description"]
    desc, sound, music = (squash(sec.values[f]) for f in BASE_FIELDS)
    shots_found = [int(n) for n in SHOT_RE.findall(desc)]
    final_shot = max(shots_found) if shots_found else 1

    expected = None
    if mode in ("FL2VA", "L2VA") and not duration:
        E.append(f"{mode} needs an exact duration; none was given.")
    else:
        expected = alignment_line(mode, duration, final_shot)

    def rebuild(align):
        parts = ([align] if align else []) + [f"integrated_multimodal_description: {desc}",
                                             f"overall_soundscape: {sound}", f"non_diegetic_music: {music}"]
        return "\n\n".join(parts)

    if autofix:
        align = None
        if mode == "T2VA":
            if sec.pre:
                res.fixes.append("removed alignment line (T2VA has none)")
        else:
            align = expected or sec.pre or None
            if expected and sec.pre != expected:
                res.fixes.append("replaced alignment line with the exact template")
        if "\n" in raw_desc.strip():
            res.fixes.append("joined line breaks inside integrated_multimodal_description")
        new = rebuild(align)
        if new != text and not res.fixes:
            res.fixes.append("normalised blank lines between fields")
        res.text = text = new
    else:
        if mode == "T2VA" and sec.pre:
            E.append("T2VA prompts have no alignment line; the prompt must start with `integrated_multimodal_description:`.")
        elif expected and sec.pre != expected:
            E.append(f"Alignment line must be exactly: {expected}")
        if "\n" in raw_desc.strip():
            E.append("integrated_multimodal_description contains line breaks; it must be a single line.")
        if rebuild(sec.pre or None) != text:
            E.append("Layout: exactly one blank line between the alignment line and each field, and nowhere else.")

    if not desc.startswith("[Shot 1]"):
        E.append("integrated_multimodal_description must begin with `[Shot 1]`.")
    shots, segs = _shot_checks(desc, duration, E, W)

    for tag in set(re.findall(r"<[^<>]{1,40}>", text)):
        if tag not in BASE_TAGS:
            E.append(f"Tag {tag} is not an H3 tag for {mode}; only <d>, </d>, <scenetrans>, <cutoff>, <Picture 1>, <Picture 2>.")
    if mode == "T2VA" and "Picture" in text:
        E.append("T2VA has no reference picture; remove mentions of Picture 1/2.")
    if mode in ("I2VA", "L2VA") and "Picture 2" in text:
        E.append(f"{mode} has only one reference picture; remove mentions of Picture 2.")
    if "<scenetrans>" in desc and desc.count("<scenetrans>") % 2 == 1:
        W.append("<scenetrans> should appear in both shots around a cut.")
    if not any(s in desc[len("[Shot 1]"):][:90].lower() for s in STYLE_WORDS):
        W.append("[Shot 1] should open with the visual style (e.g. 'Live-action, cinematic, ...'), not a summary.")

    outside_desc, dwords, has_d = _dialogue_checks(desc, sound, music, segs, duration, E, W, False)
    _camera_checks(desc, outside_desc, E, W)
    _sound_checks(sound, music, has_d, silent_ok, E, W)
    _common_text_checks(text, E)

    nsent = len(sentences(outside_desc))
    budget = duration or 6.0
    res.stats = {"shots": len(shots) or 1, "sentences": nsent, "dialogue_words": dwords}
    if nsent > budget * 1.25 + 2:
        W.append(f"About {nsent} sentences for a {'%.2fs' % duration if duration else '~6s (assumed)'} clip; more beats than seconds get dropped at random. Trim to roughly one beat per second.")
    tail = desc[-400:]
    if mode == "FL2VA" and "Picture 2" not in tail:
        W.append("FL2VA should end by settling into the pose, viewpoint, lighting and composition of Picture 2.")
    if mode == "L2VA" and "<Picture 1>" not in tail:
        W.append("L2VA should end by stating the scene reaches the exact state of <Picture 1>.")
    if mode == "I2VA" and "<Picture 1>" not in desc[:400]:
        W.append("I2VA should anchor [Shot 1] to <Picture 1> (style, subject, composition).")


# ---------------------------------------------------------------- entry point

def lint(raw: str, mode: str = "auto", duration: float = 0.0, autofix: bool = True, silent_ok: bool = False,
         pictures=None) -> LintResult:
    """Check a prompt.

    mode: Ref2VA, T2VA, I2VA, FL2VA, L2VA, or anything else to detect it from the text.
    pictures: Ref2VA only - number of reference images, or the list of picture numbers supplied.
    """
    text = extract_prompt(raw)
    fixes = []
    if text != (raw or "").strip():
        fixes.append("removed text outside the prompt (code fence, reasoning or commentary)")
    if mode not in MODES:
        mode, inferred = infer_mode_and_duration(text)
        duration = duration or inferred
    res = LintResult(text=text, mode=mode, duration=float(duration or 0.0), fixes=fixes)
    if looks_like_question(text):
        res.errors.append(f"The model asked a question instead of writing a prompt: {text}")
        return res
    if mode == REF_MODE:
        _lint_ref(res, text, autofix, pictures, silent_ok)
    else:
        _lint_base(res, text, autofix, silent_ok)
    res.errors = list(dict.fromkeys(res.errors))
    res.warnings = list(dict.fromkeys(res.warnings))
    return res
