from flask import Flask, Response
from picamera2 import Picamera2
from picamera2 import MappedArray  # not strictly needed but harmless
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
import cv2
import threading
import numpy as np
import os

app = Flask(__name__)

# -------------------------------------------------------------------
# IMX500 / model setup
# -------------------------------------------------------------------
MODEL_FILE = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

DETECTION_THRESHOLD = 0.55
IOU_THRESHOLD = 0.65
MAX_DETECTIONS = 10

imx500 = None
intrinsics = None

try:
    if os.path.exists(MODEL_FILE):
        imx500 = IMX500(MODEL_FILE)
        intrinsics = imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            print("[WARN] IMX500 network is not an object detection task")
        intrinsics.update_with_defaults()
        print(f"[INFO] IMX500 model loaded: {MODEL_FILE}")
    else:
        print(f"[WARN] IMX500 model file not found: {MODEL_FILE}")
except Exception as e:
    print(f"[WARN] IMX500 disabled: {e}")
    imx500 = None
    intrinsics = None

# Only used for filtering labels, if labels are present
INTERESTING = {
    "bird",
    "person",
    "bicycle", "car", "motorcycle",
    "bus", "train", "truck", "boat",
}

# -------------------------------------------------------------------
# Cameras init (fail-safe)
# -------------------------------------------------------------------
cam0 = None
cam1 = None
cam0_ok = False
cam1_ok = False

# cam0 = NoIR IMX219 (bird box)
try:
    cam0 = Picamera2(camera_num=0)
    cfg0 = cam0.create_video_configuration(
        main={"size": (3280, 2464), "format": "RGB888"},
        controls={
            "FrameRate": 5,
            "AwbMode": 0,
            "ColourGains": (1.2, 1.6),
        },
    )
    cam0.configure(cfg0)
    cam0_cfg = None  # we already configured, so we just call start()/stop()
    cam0_ok = True
    print("[INFO] cam0 (IMX219) initialised")
except Exception as e:
    print(f"[ERROR] cam0 init failed: {e}")
    cam0_cfg = None

# cam1 = AI IMX500
cam1_index = imx500.camera_num if imx500 is not None else 1
cam1_cfg = None

try:
    cam1 = Picamera2(camera_num=cam1_index)

    # Match the demo: preview configuration, RGB888, correct FPS, buffer_count=12
    fps = intrinsics.inference_rate if (intrinsics and getattr(intrinsics, "inference_rate", None)) else 5
    cam1_cfg = cam1.create_preview_configuration(
        main={"format": "RGB888"},
        controls={"FrameRate": fps},
        buffer_count=12,
    )
    cam1_ok = True
    print(f"[INFO] cam1 (IMX500) initialised on index {cam1_index}")
except Exception as e:
    print(f"[ERROR] cam1 init failed: {e}")
    cam1_cfg = None

# -------------------------------------------------------------------
# Viewer counters (low consumption)
# -------------------------------------------------------------------
viewers0 = 0
viewers1 = 0
lock0 = threading.Lock()
lock1 = threading.Lock()

# -------------------------------------------------------------------
# Detection classes / helpers (from demo)
# -------------------------------------------------------------------
class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        # convert to (x, y, w, h) in cam1 main stream coords
        self.box = imx500.convert_inference_coords(coords, metadata, cam1)


def parse_detections(metadata: dict):
    if imx500 is None or intrinsics is None:
        return []

    bbox_normalization = intrinsics.bbox_normalization
    threshold = DETECTION_THRESHOLD
    iou = IOU_THRESHOLD
    max_detections = MAX_DETECTIONS

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return None

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=threshold,
            iou_thres=iou,
            max_out_dets=max_detections,
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(
            boxes,
            1, 1,
            input_h, input_w,
            False, False
        )
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return detections

# -------------------------------------------------------------------
# Generators
# -------------------------------------------------------------------
def gen0():
    global viewers0
    if not cam0_ok:
        return

    with lock0:
        viewers0 += 1
        if viewers0 == 1:
            cam0.start()
            print("[INFO] cam0 started")

    try:
        while True:
            frame = cam0.capture_array()
            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg.tobytes() +
                b"\r\n"
            )
    finally:
        with lock0:
            viewers0 -= 1
            if viewers0 == 0:
                cam0.stop()
                print("[INFO] cam0 stopped")


def gen1():
    global viewers1
    if not cam1_ok:
        return

    with lock1:
        viewers1 += 1
        if viewers1 == 1:
            # Start with the exact config we created above
            if cam1_cfg is not None:
                cam1.start(cam1_cfg)
            else:
                cam1.start()
            print("[INFO] cam1 started")

            # Match demo: set ROI/aspect ratio if needed
            if intrinsics and intrinsics.preserve_aspect_ratio:
                imx500.set_auto_aspect_ratio()

            # Show FW progress bar once (safe even if already loaded)
            if imx500 is not None:
                imx500.show_network_fw_progress_bar()

    last_detections = []
    try:
        while True:
            req = cam1.capture_request()
            try:
                metadata = req.get_metadata()
                if metadata:
                    dets = parse_detections(metadata)
                    if dets is None:
                        dets = last_detections
                    last_detections = dets
                else:
                    dets = []

                frame = req.make_array("main")

                labels = getattr(intrinsics, "labels", None) if intrinsics else None

                for d in dets:
                    x, y, w, h = d.box
                    label_name = None
                    interesting = True

                    if labels and int(d.category) < len(labels):
                        label_name = labels[int(d.category)]
                        if label_name not in INTERESTING:
                            interesting = False

                    if not interesting:
                        continue

                    if label_name is None:
                        text = f"{int(d.category)} ({d.conf:.2f})"
                    else:
                        text = f"{label_name} ({d.conf:.2f})"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        text,
                        (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                )
            finally:
                req.release()

            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg.tobytes() +
                b"\r\n"
            )
    finally:
        with lock1:
            viewers1 -= 1
            if viewers1 == 0:
                cam1.stop()
                print("[INFO] cam1 stopped")

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/cam0")
def stream0():
    if not cam0_ok:
        return Response("cam0 not available\n", status=503)
    return Response(
        gen0(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/cam1")
def stream1():
    if not cam1_ok:
        return Response("cam1 not available\n", status=503)
    return Response(
        gen1(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
