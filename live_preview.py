import threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from ultralytics import YOLO

model = YOLO("best_ncnn_model", task="detect")
latest_jpg = b""
lock = threading.Lock()

def camera_loop():
    global latest_jpg
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        t0 = time.time()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.82)
        roi = frame[y1:y2, x1:x2]
        res = model.predict(roi, imgsz=416, conf=0.30, verbose=False)[0]
        frame[y1:y2, x1:x2] = res.plot()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        fps = 1.0 / max(1e-3, time.time() - t0)
        labels = [f"{res.names[int(b.cls[0])]}:{float(b.conf[0]):.2f}" for b in res.boxes]
        status = f"FPS:{fps:.1f} | " + (", ".join(labels) if labels else "PASS (Align PCB inside yellow ROI)")
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(frame, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with lock:
            latest_jpg = buf.tobytes()

HTML = b"""<!doctype html><html><body style="background:#111;color:#eee;font-family:sans-serif;text-align:center">
<h3>Live Edge AOI Alignment View (YOLOv8n NCNN)</h3>
<p>Fill the <b>yellow 65% ROI box</b> completely with green PCB traces (no white paper or table edges).</p>
<img id="v" src="/frame.jpg" style="max-width:92vw;border:2px solid #444;border-radius:8px">
<script>
const img = document.getElementById('v');
img.onload = () => setTimeout(() => { img.src = '/frame.jpg?t=' + Date.now(); }, 30);
img.onerror = () => setTimeout(() => { img.src = '/frame.jpg?t=' + Date.now(); }, 200);
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            with lock:
                data = latest_jpg
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML)
    def log_message(self, *args):
        pass

threading.Thread(target=camera_loop, daemon=True).start()
print("Live preview running at http://raspberrypi.local:5050 (Ctrl+C to stop)")
ThreadingHTTPServer(("0.0.0.0", 5050), Handler).serve_forever()
