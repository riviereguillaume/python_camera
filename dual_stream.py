from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import threading

app = Flask(__name__)

# cam0 = NoIR IMX219 (bird box)
# cam1 = AI IMX500
cam0 = None
cam1 = None

# Try to init cam0
try:
    cam0 = Picamera2(camera_num=0)

    # IMX219 max: 3280x2464 - night vision style, long exposure + high gain
    cfg0 = cam0.create_video_configuration(
        main={"size": (3280, 2464), "format": "RGB888"},
        controls={
            "FrameRate": 5,
            "AeEnable": False,
            "AwbEnable": False,
            "AnalogueGain": 10.0,
            "ExposureTime": 100000
        }
    )
    cam0.configure(cfg0)
except Exception as e:
    print(f"[WARN] cam0 (NoIR IMX219) not available: {e}")
    cam0 = None

# Try to init cam1
try:
    cam1 = Picamera2(camera_num=1)

    # IMX500 max: 4056x3040
    cfg1 = cam1.create_video_configuration(
        main={"size": (4056, 3040), "format": "RGB888"},
        controls={
            "FrameRate": 5
        }
    )
    cam1.configure(cfg1)
except Exception as e:
    print(f"[WARN] cam1 (IMX500) not available: {e}")
    cam1 = None

viewers0 = 0
viewers1 = 0
lock0 = threading.Lock()
lock1 = threading.Lock()


def gen0():
    """Stream generator for cam0 (NoIR IMX219)."""
    global viewers0
    if cam0 is None:
        return
    with lock0:
        viewers0 += 1
        if viewers0 == 1:
            cam0.start()
            cam0.set_controls({
                "FrameRate": 5,
                "AeEnable": False,
                "AwbEnable": False,
                "AnalogueGain": 10.0,
                "ExposureTime": 100000
            })
    try:
        while True:
            frame = cam0.capture_array()
            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            ok, jpg = cv2.imencode(".jpg", grey, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
    finally:
        with lock0:
            viewers0 -= 1
            if viewers0 == 0 and cam0 is not None:
                cam0.stop()


def gen1():
    """Stream generator for cam1 (IMX500)."""
    global viewers1
    if cam1 is None:
        return
    with lock1:
        viewers1 += 1
        if viewers1 == 1:
            cam1.start()
    try:
        while True:
            frame = cam1.capture_array()
            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
    finally:
        with lock1:
            viewers1 -= 1
            if viewers1 == 0 and cam1 is not None:
                cam1.stop()


@app.route("/cam0")
def stream0():
    if cam0 is None:
        return "Error: cam0 (NoIR IMX219) not detected at startup", 503
    return Response(
        gen0(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/cam1")
def stream1():
    if cam1 is None:
        return "Error: cam1 (IMX500) not detected at startup", 503
    return Response(
        gen1(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
