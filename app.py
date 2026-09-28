"""Body-cam face anonymizer —  FastAPI app.

Pipeline:

  video → decode → SCRFD detect → ByteTrack → dilate box → Gaussian blur → mp4

On-demand QC ("Run QC" button on the detail page):

  Independent RetinaFace-ResNet50 pass samples the anonymized
  output 1 frame/second. Any face still detected → residual leak → the
  job is flagged.

Run:  python app.py     → http://127.0.0.1:8000
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
SAMPLES = ROOT / "samples"
TEMPLATES = ROOT / "templates"
DB_PATH = ROOT / "jobs.db"

for d in (UPLOADS, OUTPUTS):
    d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Detector — SCRFD via insightface (SOTA, ONNX, CPU-friendly)
# ---------------------------------------------------------------------------

class SCRFD:
    """Wraps insightface FaceAnalysis with the detection module only."""

    def __init__(self, det_size: tuple[int, int] = (640, 640), threshold: float = 0.5):
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self._app.prepare(ctx_id=-1, det_size=det_size, det_thresh=threshold)

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[list[float], float]]:
        return [(f.bbox.tolist(), float(f.det_score)) for f in self._app.get(frame_bgr)]


# Lazy singletons so the model loads on first use, not import.
_FAST: Optional[SCRFD] = None
_QC_RETINAFACE: Any = None    # facexlib RetinaFace-ResNet50, loaded on demand.


def fast_detector() -> SCRFD:
    global _FAST
    if _FAST is None:
        # Low internal threshold so ByteTrack's stage-2 sees weak detections.
        _FAST = SCRFD(det_size=(640, 640), threshold=0.1)
    return _FAST


def qc_retinaface() -> Any:
    """Independent QC detector: RetinaFace-ResNet50"""
    global _QC_RETINAFACE
    if _QC_RETINAFACE is None:
        import warnings
        from facexlib.detection import init_detection_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _QC_RETINAFACE = init_detection_model(
                "retinaface_resnet50", half=False, device="cpu",
            )
    return _QC_RETINAFACE


# ---------------------------------------------------------------------------
# 2. Tracker — supervision.ByteTrack (Kalman motion + two-stage cascade)
# ---------------------------------------------------------------------------



class Tracker:
    """Wraps supervision.ByteTrack, also emitting Kalman-predicted boxes on
    detector-miss frames so the anonymizer never skips a blur."""

    def __init__(self, max_missed: int = 8):
        import warnings
        import supervision as sv

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._bt = sv.ByteTrack()
        self._sv = sv
        self._max_missed = max_missed
        self._miss_start_frame: dict[int, int] = {}

    def update(self, detections: list[tuple[list[float], float]]) -> list[list[float]]:
        if detections:
            det = self._sv.Detections(
                xyxy=np.array([b for b, _ in detections], dtype=float),
                confidence=np.array([s for _, s in detections], dtype=float),
                class_id=np.zeros(len(detections), dtype=int),
            )
        else:
            det = self._sv.Detections.empty()

        tracked = self._bt.update_with_detections(det)
        out: list[list[float]] = [row.tolist() for row in tracked.xyxy]
        matched_ids: set[int] = set()
        if tracked.tracker_id is not None and tracked.tracker_id.size > 0:
            matched_ids = {int(i) for i in tracked.tracker_id}

        # Fill in Kalman-predicted boxes for tracks that were confirmed but
        # missed this frame — otherwise the anonymizer skips them = leak.
        current_frame = self._bt.frame_id
        for lt in self._bt.lost_tracks:
            tid = int(lt.external_track_id)
            if not lt.is_activated or tid in matched_ids or tid < 0:
                continue
            self._miss_start_frame.setdefault(tid, current_frame)
            if current_frame - self._miss_start_frame[tid] <= self._max_missed:
                out.append([float(v) for v in lt.tlbr])
        # Forget miss counters for tracks that came back.
        for tid in matched_ids:
            self._miss_start_frame.pop(tid, None)

        return out


# ---------------------------------------------------------------------------
# 3. Blur — elliptical mask on a dilated bbox
# ---------------------------------------------------------------------------

def blur_face(frame: np.ndarray, bbox: list[float], pad: float = 0.35) -> None:
    """In-place Gaussian blur of the padded face bounding box.

    Blurs the entire dilated rectangle — earlier we used a decorative
    elliptical mask, but the QC pass showed detectors could still find
    the face in the corners the ellipse missed.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad))
    y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(w - 1, x2 + bw * pad))
    y2 = int(min(h - 1, y2 + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    k = max(31, int(0.5 * max(rw, rh)))
    if k % 2 == 0:
        k += 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


# ---------------------------------------------------------------------------
# 4. Anonymize — streaming video processor
# ---------------------------------------------------------------------------

def anonymize_video(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Read video → detect + track + blur → write mp4. Return stats."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # H.264 so <video> can play the output in-browser; fall back to mp4v.
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open writer: {output_path}")

    det = fast_detector()
    trk = Tracker()
    frames_total = frames_with_faces = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = det.detect(frame)
        tracks = trk.update(detections)
        if tracks:
            frames_with_faces += 1
            for bb in tracks:
                blur_face(frame, bb)
        writer.write(frame)
        frames_total += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    return {
        "frames_total_source": n_frames or frames_total,
        "frames_processed": frames_total,
        "frames_with_faces": frames_with_faces,
        "input_fps": round(fps, 2),
        "elapsed_sec": round(elapsed, 2),
        "processing_fps": round(frames_total / max(elapsed, 1e-6), 2),
    }


# ---------------------------------------------------------------------------
# 5. Independent QC — scans the OUTPUT for residual faces.
# ---------------------------------------------------------------------------

def qc_scan_output(output_path: Path, sample_every_seconds: float = 1.0) -> dict[str, Any]:
    """Scan the ANONYMIZED output with an independent RetinaFace detector.

    Any face still detected here is a leak: the blur missed something.
    RetinaFace-ResNet50 (facexlib / PyTorch) is architecturally different
    from the SCRFD (insightface / ONNX) used for anonymization
    """
    import torch

    qc = qc_retinaface()
    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open output: {output_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * sample_every_seconds)))

    frames_checked = residual_frames = residual_faces = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames_checked += 1
            # facexlib forgets torch.no_grad(); wrap it ourselves.
            with torch.no_grad():
                dets = qc.detect_faces(frame, 0.9)
            n = int(dets.shape[0]) if hasattr(dets, "shape") else len(dets)
            if n:
                residual_frames += 1
                residual_faces += n
        idx += 1
    cap.release()
    return {
        "detector": "RetinaFace-ResNet50 (facexlib)",
        "frames_checked": frames_checked,
        "sample_every_seconds": sample_every_seconds,
        "residual_face_frames": residual_frames,
        "residual_faces_total": residual_faces,
        "flagged": residual_frames > 0,
    }


# ---------------------------------------------------------------------------
# 6. SQLite job store — idempotent on event_id
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    event_id      TEXT PRIMARY KEY,
    video_path    TEXT NOT NULL,
    status        TEXT NOT NULL,      -- pending / processing / completed / failed
    stats_json    TEXT,
    miss_json     TEXT,
    qc_json       TEXT,
    error         TEXT,
    created_at    REAL NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA busy_timeout=10000;")
    return c


with _conn() as _c:
    _c.executescript(_SCHEMA)
    # Idempotent migration for DBs created before qc_json existed.
    try:
        _c.execute("ALTER TABLE jobs ADD COLUMN qc_json TEXT;")
    except sqlite3.OperationalError:
        pass


def enqueue_job(event_id: str, video_path: Path) -> bool:
    """INSERT OR IGNORE — duplicate deliveries of the same event id collapse."""
    with _conn() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO jobs(event_id, video_path, status, created_at) "
            "VALUES (?, ?, 'pending', ?)",
            (event_id, str(video_path), time.time()),
        )
        return cur.rowcount == 1


def get_job(event_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        r = c.execute(
            "SELECT event_id, video_path, status, stats_json, qc_json, "
            "error, created_at FROM jobs WHERE event_id=?",
            (event_id,),
        ).fetchone()
    if not r:
        return None
    return {
        "event_id": r[0],
        "video_path": r[1],
        "status": r[2],
        "stats": json.loads(r[3]) if r[3] else None,
        "qc": json.loads(r[4]) if r[4] else None,
        "error": r[5],
        "created_at": r[6],
    }


def list_jobs() -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT event_id, status, qc_json, error, created_at "
            "FROM jobs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    out = []
    for r in rows:
        qc = json.loads(r[2]) if r[2] else None
        out.append({
            "event_id": r[0],
            "status": r[1],
            "qc_flagged": qc["flagged"] if qc else None,
            "error": r[3],
            "created_at": r[4],
        })
    return out


def update_job(event_id: str, **fields: Any) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE event_id=?",
                  (*fields.values(), event_id))


# ---------------------------------------------------------------------------
# 7. Background worker — serialize CPU-bound jobs with a threading.Lock
# ---------------------------------------------------------------------------

_worker_lock = threading.Lock()


def _worker(event_id: str) -> None:
    with _worker_lock:
        job = get_job(event_id)
        if not job or job["status"] != "pending":
            return
        update_job(event_id, status="processing")
        try:
            src = Path(job["video_path"])
            dst = OUTPUTS / f"{event_id}.mp4"
            stats = anonymize_video(src, dst)
            update_job(event_id,
                       status="completed",
                       stats_json=json.dumps(stats))
        except Exception as exc:
            update_job(event_id, status="failed", error=repr(exc))


def _qc_worker(event_id: str) -> None:
    """On-demand QC: scan the anonymized output for residual faces."""
    with _worker_lock:
        job = get_job(event_id)
        if not job:
            return
        try:
            result = qc_scan_output(OUTPUTS / f"{event_id}.mp4")
            update_job(event_id, qc_json=json.dumps(result))
        except Exception as exc:
            update_job(event_id, error=repr(exc))


# ---------------------------------------------------------------------------
# 8. FastAPI routes
# ---------------------------------------------------------------------------

app = FastAPI(title="Face Anonymizer")
templates = Jinja2Templates(directory=str(TEMPLATES))
templates.env.cache = None  # Jinja LRU-cache bug workaround on Python 3.14


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    sample_files = sorted(p.name for p in SAMPLES.glob("*.mp4")) if SAMPLES.exists() else []
    return templates.TemplateResponse(
        request, "index.html", {"samples": sample_files, "jobs": list_jobs()}
    )


@app.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "no filename")
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise HTTPException(400, f"unsupported extension: {suffix}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    # Content hash = idempotency key. Two uploads of the same file dedupe.
    event_id = hashlib.sha256(data).hexdigest()[:16]
    dest = UPLOADS / f"{event_id}{suffix}"
    dest.write_bytes(data)
    if enqueue_job(event_id, dest):
        background.add_task(_worker, event_id)
    return RedirectResponse(url=f"/jobs/{event_id}", status_code=303)


@app.post("/run-sample")
async def run_sample(background: BackgroundTasks, name: str = Form(...)):
    src = SAMPLES / name
    if not src.exists() or src.suffix.lower() != ".mp4":
        raise HTTPException(404, f"sample not found: {name}")
    event_id = "sample-" + hashlib.sha256(name.encode()).hexdigest()[:12]
    staged = UPLOADS / f"{event_id}.mp4"
    if not staged.exists():
        shutil.copyfile(src, staged)
    if enqueue_job(event_id, staged):
        background.add_task(_worker, event_id)
    return RedirectResponse(url=f"/jobs/{event_id}", status_code=303)


@app.post("/qc/{event_id}")
def qc_endpoint(event_id: str, background: BackgroundTasks):
    job = get_job(event_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "completed":
        raise HTTPException(409, "anonymization not finished yet")
    background.add_task(_qc_worker, event_id)
    return JSONResponse({"status": "scheduled"})


@app.get("/jobs")
def jobs_json():
    return JSONResponse(list_jobs())


@app.get("/jobs/{event_id}")
def job_json_or_page(request: Request, event_id: str):
    job = get_job(event_id)
    if not job:
        raise HTTPException(404, "job not found")
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(job)
    return templates.TemplateResponse(request, "detail.html", {"job": job})


@app.get("/media/original/{event_id}")
def media_original(event_id: str):
    for suffix in (".mp4", ".mov", ".mkv", ".avi"):
        p = UPLOADS / f"{event_id}{suffix}"
        if p.exists():
            return FileResponse(p, media_type="video/mp4")
    raise HTTPException(404, "original missing")


@app.get("/media/anonymized/{event_id}")
def media_anonymized(event_id: str):
    p = OUTPUTS / f"{event_id}.mp4"
    if not p.exists():
        raise HTTPException(404, "not yet available")
    return FileResponse(p, media_type="video/mp4")


# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
