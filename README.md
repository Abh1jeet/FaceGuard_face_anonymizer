# Body-cam Face Anonymizer

FastAPI application that anonymizes faces in body-camera video and
independently audits the output for residual faces.

```
video → decode → SCRFD detect → ByteTrack → dilate box → Gaussian blur → mp4
                                                                        │
                                                                        ▼
                                                        RetinaFace QC (on demand)
                                                        flags residual faces
```

---

## Run

```bash
# Python 3.11+ recommended.
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py           # http://127.0.0.1:8000
```

First run downloads the model weights:
- SCRFD (`buffalo_l`) — ~280 MB, cached under `~/.insightface/models/`
- RetinaFace-ResNet50 — ~104 MB, cached under `.venv/.../facexlib/weights/`

Open the browser at `http://127.0.0.1:8000`:
1. **Upload video** or pick from `samples/`.
2. The job appears in *Recent jobs*, transitioning
   `pending → processing → completed`.
3. On the detail page, click **Run QC** to run the independent RetinaFace
   pass; the panel shows residual leak frames, frames sampled and the
   verdict.

---

## Approach

The system processes the video as a streaming pipeline:

**Video → Decode → SCRFD Face Detection → ByteTrack → Bounding-box dilation → Gaussian Blur → Anonymized MP4.**

SCRFD is used as the primary face detector. ByteTrack provides temporal
tracking using motion prediction and helps maintain face regions during
short detection failures or occlusions. For additional privacy margin,
the detected face bounding box is expanded before applying a strong
Gaussian blur.

If the detector temporarily misses a face, the tracker can continue
using its predicted bounding box for a limited number of frames,
reducing the possibility of an unblurred face appearing due to a
transient detection failure.

## Privacy Quality Check

After anonymization, the output can be independently validated using
**RetinaFace-ResNet50**. The QC stage samples the anonymized video at
1 frame/second and runs an independent face detector. If RetinaFace
detects any residual face, the job is flagged as a potential privacy
leak.

The QC detector is intentionally architecturally different from the
anonymization detector (RetinaFace vs. SCRFD) to reduce the likelihood
of shared model failure modes.

## Evaluation

I would evaluate the system using:

- Face detection recall / missed-face rate
- Residual face detection rate after anonymization
- Blur coverage of detected face regions
- Temporal consistency / flicker rate
- Performance in low-light, motion blur, occlusion, unusual viewpoints,
  small faces, camera movement, and crowded scenes
- Processing throughput (FPS) and end-to-end processing time

For a rigorous evaluation, human-annotated face bounding boxes would be
used as ground truth rather than treating another detector as ground
truth.

## Asynchronous Processing and Idempotency

Video processing is performed asynchronously using FastAPI background
tasks. Each uploaded video is assigned a SHA-256 content hash as its
event/job ID. The job store uses SQLite with `event_id` as a primary
key and `INSERT OR IGNORE`, so duplicate deliveries of the same upload
collapse into a single processing job rather than triggering duplicate
processing.

Job states are persisted as `pending`, `processing`, `completed`, or
`failed`, allowing the processing status and errors to be tracked.

## Known Limitations

The current implementation is a prototype. The QC stage samples one
frame per second, so a residual face that appears only briefly between
sampled frames may not be detected. The current worker also serializes
processing and is not intended for high-scale deployment.

## Production Considerations

For production, I would use a durable queue such as Kafka/SQS/Pub/Sub
with a scalable GPU worker pool, object storage for input/output
videos, atomic job claiming for multi-worker idempotency, GPU-accelerated
inference, comprehensive monitoring, and a fail-closed policy where an
output that fails the privacy QC is not released downstream.

The code is runnable as a FastAPI application and includes upload,
asynchronous processing, job-status tracking, anonymized-video output,
and an independent QC workflow.

---

## Layout

```
app.py                # entire pipeline + API + worker + RetinaFace QC
requirements.txt      # pinned dependencies
templates/
  index.html          # upload form + jobs table with live status
  detail.html         # side-by-side players + Run QC button + metrics
samples/              # bundled clips for the demo
uploads/              # created at runtime — inputs
outputs/              # created at runtime — anonymized mp4s
jobs.db               # created at runtime — SQLite job queue
```

## API

| Method / path                 | Purpose                                             |
|-------------------------------|-----------------------------------------------------|
| `GET  /`                      | Landing page (upload + jobs list)                   |
| `POST /upload`                | Multipart upload → enqueue → redirect to detail     |
| `POST /run-sample`            | Enqueue one of the bundled sample clips             |
| `GET  /jobs`                  | JSON list of recent jobs                            |
| `GET  /jobs/{event_id}`       | HTML detail page (or JSON if `Accept: application/json`) |
| `POST /qc/{event_id}`         | Run the RetinaFace QC pass on the anonymized output |
| `GET  /media/original/{id}`   | Streams the input mp4                               |
| `GET  /media/anonymized/{id}` | Streams the anonymized output mp4                   |
