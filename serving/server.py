# server.py — FastAPI backend that processes uploaded videos frame-by-frame,
# runs SSD detection + norfair tracking, and pushes results to the browser
# via WebSocket (Option A: server-paced, video waits for each detection).
#
# Start with:  uvicorn server:app --reload
# Then open:   http://localhost:8000

import os
import uuid
import asyncio
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from norfair import Tracker, Detection

from serving.inference import load_ssd_model, detect_frame

app = FastAPI()
@app.get("/health")
def health():
    """Endpoint utilisé par le HEALTHCHECK Docker et les sondes Kubernetes."""
    return {"status": "ok", "model_loaded": model is not None}

# --- globals: loaded once at startup, shared across all requests ---
model = None
config = None
transform = None
device = None

# maps session_id → path of the uploaded temp video file
video_sessions: dict[str, str] = {}

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.on_event("startup")
def startup():
    """Called once when uvicorn starts. Loads the SSD model into memory."""
    global model, config, transform, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, transform = load_ssd_model(device)
    print("Server ready — model loaded on", device)


MAX_VIDEO_DURATION_SEC=40

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Receive a video file from the browser, save it to a temp file,
    return a session_id the browser will use to start detection."""
    session_id = str(uuid.uuid4())

    suffix = os.path.splitext(file.filename or ".mp4")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    video_sessions[session_id] = tmp.name
    return {"session_id": session_id}


# ── WebSocket detection endpoint ─────────────────────────────────────

@app.websocket("/ws/detect")
async def detect_ws(websocket: WebSocket):
    """Server-paced detection loop (Option A).
    1. Browser sends config {session_id, score_thr, frame_skip}
    2. Server reads frames with OpenCV, runs SSD + tracker, sends JSON per frame
    3. Browser receives JSON → seeks video → draws boxes
    4. Server sends {type: "done"} when finished
    """
    await websocket.accept()

    try:
        # wait for the browser to send session config
        init_msg = await websocket.receive_json()
        session_id = init_msg["session_id"]
        score_thr = init_msg.get("score_thr", 0.25)
        frame_skip = init_msg.get("frame_skip", 0)

        video_path = video_sessions.get(session_id)
        if not video_path:
            await websocket.send_json({"type": "error", "message": "Invalid session"})
            await websocket.close()
            return

        # open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # tell the browser the video metadata so it can seek correctly
        await websocket.send_json({
            "type": "metadata",
            "fps": fps,
            "total_frames": total_frames,
        })

        # create a fresh tracker for this video
        tracker = Tracker(
            distance_function="iou",
            distance_threshold=0.7,
            hit_counter_max=15,
            initialization_delay=3,
        )
        max_frame=int(fps*MAX_VIDEO_DURATION_SEC)

        # keep track of every unique ID seen across all frames
        all_track_ids: set[int] = set()
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_idx >=max_frame:
                break

            # skip frames if requested (frame_skip=1 means process every 2nd frame)
            if frame_idx % (frame_skip + 1) != 0:
                frame_idx += 1
                continue

            # convert OpenCV BGR numpy array → PIL RGB image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # run SSD detection on this frame
            detections_np = detect_frame(model, pil_image, transform, device, score_thr)

            # convert SSD detections to norfair Detection objects
            norfair_dets = []
            for det in detections_np:
                x1, y1, x2, y2, score, cls = det
                norfair_dets.append(
                    Detection(
                        points=np.array([[x1, y1], [x2, y2]]),
                        scores=np.array([score, score]),
                        data={"class": int(cls), "score": float(score)},
                    )
                )

            # update tracker with this frame's detections
            tracked_objects = tracker.update(detections=norfair_dets)

            # build JSON response for this frame
            boxes = []
            track_ids = []
            scores_list = []
            classes_list = []

            for obj in tracked_objects:
                if obj.estimate is not None:
                    box = obj.estimate.flatten().tolist()
                    boxes.append(box)
                    track_ids.append(obj.id)
                    all_track_ids.add(obj.id)

                    if obj.last_detection and obj.last_detection.data:
                        scores_list.append(obj.last_detection.data["score"])
                        cls_id = obj.last_detection.data["class"]
                        classes_list.append(config.get(cls_id, f"class_{cls_id}"))
                    else:
                        scores_list.append(0.0)
                        classes_list.append("unknown")

            # send this frame's results to the browser
            await websocket.send_json({
                "type": "detection",
                "frame": frame_idx,
                "time": frame_idx / fps,
                "boxes": boxes,
                "track_ids": track_ids,
                "scores": scores_list,
                "classes": classes_list,
                "current_count": len(tracked_objects),
                "total_unique": len(all_track_ids),
            })

            frame_idx += 1

            # yield control so the WebSocket message actually gets sent
            await asyncio.sleep(0)

        cap.release()

        # tell the browser we're done
        await websocket.send_json({
            "type": "done",
            "total_unique": len(all_track_ids),
            "total_frames_processed": frame_idx,
        })

        # clean up the temp video file
        try:
            os.unlink(video_path)
        except OSError:
            pass
        video_sessions.pop(session_id, None)

    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Serve the website static files (HTML, CSS, JS) ──────────────────

app.mount("/", StaticFiles(directory=os.path.join(project_root, "website"), html=True), name="website")
