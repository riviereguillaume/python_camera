from flask import Flask, Response
from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500
import cv2
import threading

app = Flask(__name__)

#########################################################
# CAMERA INITIALIZATION + FAIL-SAFE
#########################################################

cam0 = None
cam1 = None
imx500 = None

# ----- Try CAM0 (NoIR IMX219) -----
try:
    cam0 = Picamera2(camera_num=0)
    cfg0 = cam0.create_video_configuration(
        main={"size": (3280, 2464), "format": "RGB888"},
        controls={
            "FrameRate": 5,
            "AwbMode": 0,
            "ColourGains": (1.2, 1.6),
        }
    )
    cam0.configure(cfg0)
except Exception as e:
    print(f"[WARN] cam0 not available: {e}")
    cam0 = None

# ----- Try CAM1 (IMX500 AI Camera) -----
try:
    MODEL_FILE = "/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk"

    # Load model only once
    imx500 = IMX500(MODEL_FILE)
    imx500.show_network_fw_progress_bar()

    cam1 = Picamera2(camera_num=1)
    cfg1 = cam1.create_video_configuration(
        main={"size": (4056, 3040), "format": "RGB888"},
        controls={"FrameRate": 5}
    )
    cam1.configure(cfg1)

except Exception as e:
    print(f"[WARN] cam1 not available or AI model failed: {e}")
    cam1 = None
    imx500 = None

#########################################################
# STREAMING CONTROL
#########################################################

viewers0 = 0
viewers1 = 0
lock0 = threading.Lock()
lock1 = threading.Lock()

#########################################################
# IMX500 AI DETECTION (cam1 only)
#########################################################

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

INTERESTING = {
    "bird",
    "person",
    "bicycle", "car", "motorcycle",
    "bus", "train", "truck", "boat"
}

CONF_THRESHOLD = 0.40


class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = int(category)
        self.conf = float(conf)
        scaled = imx500.convert_inference_coords(coords, metadata, cam1)
        self.x = scaled.x
        self.y = scaled.y
        self.w = scaled.width
        self.h = scaled.height


def parse_detections(req):
    metadata = req.get_metadata()
    outputs = imx500.get_outputs(metadata)
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]

    out = []
    for b, s, c in zip(boxes, scores, classes):
        if s < CONF_THRESHOLD:
            continue
        label = COCO_LABELS[int(c)]
        if label not in INTERESTING:
            continue
        out.append(Detection(b, c, s, metadata))
    return out


def draw_detections(req, dets):
    with MappedArray(req, "main") as m:
        for d in dets:
            label = COCO_LABELS[d.category]
            text = f"{label} {d.conf:.2f}"
            p1 = (d.x, d.y)
            p2 = (d.x + d.w, d.y + d.h)
            cv2.rectangle(m.array, p1, p2, (0, 0, 255), 2)
            cv2.putText(m.array, text, (d.x+5, d.y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

#########################################################
# STREAM GENERATORS
#########################################################

def gen0():
    global viewers0
    with lock0:
        viewers0 += 1
        if viewers0 == 1:
            cam0.start()

    try:
        while True:
            frame = cam0.capture_array()
            ok, jpg = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")
    finally:
        with lock0:
            viewers0 -= 1
            if viewers0 == 0:
                cam0.stop()


def gen1():
    global viewers1
    with lock1:
        viewers1 += 1
        if viewers1 == 1:
            cam1.start()

    try:
        while True:
            req = cam1.capture_request()

            if imx500 is not None:
                dets = parse_detections(req)
                draw_detections(req, dets)

            frame = req.make_array("main")
            req.release()

            ok, jpg = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")
    finally:
        with lock1:
            viewers1 -= 1
            if viewers1 == 0:
                cam1.stop()

#########################################################
# ROUTES
#########################################################

@app.route("/cam0")
def stream0():
    if cam0 is None:
        return "cam0 missing", 503
    return Response(gen0(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cam1")
def stream1():
    if cam1 is None:
        return "cam1 missing", 503
    return Response(gen1(), mimetype="multipart/x-mixed-replace; boundary=frame")


#########################################################
# MAIN
#########################################################

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
