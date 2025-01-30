import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from flask import Flask, render_template, Response, redirect, url_for

app = Flask(__name__)

# Initialize video capture and face mesh detector
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

# Real-world width and focal length
W = 6.3  # Real-world width of the object
f = 310  # Focal length

def generate_frames():
    """Generate video frames for streaming."""
    global cap
    while cap and cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        else:
            img, faces = detector.findFaceMesh(img, draw=False)
            if faces:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]

                w, _ = detector.findDistance(pointLeft, pointRight)
                if w > 0:  # Avoid division by zero
                    d = (W * f) / w

                    # Display distance on the image
                    cvzone.putTextRect(img, f'Distance: {int(d)} cm',
                                       (face[10][0] - 75, face[10][1] - 50), scale=2)

            # Convert the frame to JPEG format for streaming
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the HTML page."""
    return render_template('index.html')

@app.route('/start')
def start():
    """Start the video capture."""
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    """Stop the video capture."""
    global cap
    if cap:
        cap.release()
        cap = None
    return redirect(url_for('index'))

@app.route('/exit')
def exit_app():
    """Exit the app."""
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    return "Application exited. Please close this browser tab."

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global cap
    if cap:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera not started. Please click Start."


if __name__ == '__main__':
    app.run(debug=True)