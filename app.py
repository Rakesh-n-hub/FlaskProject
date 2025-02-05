from flask import Flask, render_template, Response, jsonify
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

app = Flask(__name__)

cap = None  # Camera instance (initialized as None)
detector = FaceMeshDetector(maxFaces=1)

W = 6.3  # Real-world width
f = 310  # Focal length
running = False  # Camera state
distance_value = "--"  # Distance placeholder


def generate_frames():
    """Generates video frames for streaming"""
    global cap, running, distance_value
    while running:
        success, img = cap.read()
        if not success:
            break  # Exit loop if the camera feed fails
        else:
            img, faces = detector.findFaceMesh(img, draw=False)
            if faces:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]

                w, _ = detector.findDistance(pointLeft, pointRight)
                if w > 0:
                    d = (W * f) / w
                    distance_value = int(d)

                    cvzone.putTextRect(img, f'Distance: {int(d)} cm',
                                       (face[10][0] - 75, face[10][1] - 50), scale=2)

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Starts the video stream only if running is True"""
    if running:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"error": "Camera is not running"})


@app.route('/start')
def start():
    """Starts the camera and video feed"""
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
    return jsonify({"status": "started"})


@app.route('/stop')
def stop():
    """Stops the camera and releases resources"""
    global cap, running
    if running:
        running = False
        if cap:
            cap.release()
            cap = None  # Set cap to None so it resets when started again
    return jsonify({"status": "stopped"})


@app.route('/get_distance')
def get_distance():
    """Fetches the latest calculated distance"""
    return jsonify({"distance": distance_value})


@app.route('/exit')
def exit_app():
    """Properly shuts down the app and releases resources"""
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None
    return jsonify({"status": "exited"})


if __name__ == '__main__':
    app.run(debug=True)
