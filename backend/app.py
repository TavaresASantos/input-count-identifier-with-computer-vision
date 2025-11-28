from flask import Flask, render_template, Response
import cv2
from collections import defaultdict

app = Flask(__name__)

cap = cv2.VideoCapture('media/1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

posL = 150
offset = 30
xy1 = (20, posL)
xy2 = (300, posL)

up = 0
down = 0
total = 0

detects = []

def center(x, y, w, h):
    """Retorna o centro de um retângulo"""
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

def gen_frames():
    """Gera frames do vídeo para streaming MJPEG no Flask"""
    global up, down, total, detects

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        dilation = cv2.dilate(opening, kernel, iterations=8)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=8)

        cv2.line(frame, xy1, xy2, (255,0,0), 3)
        cv2.line(frame, (xy1[0], posL-offset), (xy2[0], posL-offset), (255,255,0), 2)
        cv2.line(frame, (xy1[0], posL+offset), (xy2[0], posL+offset), (255,255,0), 2)

        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if area > 3000:
                cX, cY = center(x, y, w, h)
                cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.circle(frame, (cX, cY), 4, (0,0,255), -1)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                if len(detects) <= i:
                    detects.append([])

                if posL - offset < cY < posL + offset:
                    detects[i].append((cX, cY))
                else:
                    detects[i].clear()
                i += 1

        if i == 0 or len(contours) == 0:
            detects.clear()

        for detect in detects:
            for c, point in enumerate(detect):
                if c > 0:
                    y_prev = detect[c-1][1]
                    y_now = point[1]

                    if y_prev < posL and y_now > posL:
                        up += 1
                        total += 1
                        detect.clear()
                        cv2.line(frame, xy1, xy2, (0,255,0), 5)
                        continue

                    if y_prev > posL and y_now < posL:
                        down += 1
                        total += 1
                        detect.clear()
                        cv2.line(frame, xy1, xy2, (0,0,255), 5)
                        continue

                    cv2.line(frame, detect[c-1], point, (0,0,255), 1)

        cv2.putText(frame, f"Entrada: {up}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, f"Saida: {down}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html') 

if __name__ == '__main__':
    app.run(debug=True)
