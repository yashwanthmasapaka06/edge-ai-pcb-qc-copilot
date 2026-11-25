import glob, os, time, cv2
from ultralytics import YOLO
from qc_assistant import CLASS_REMAP

model = YOLO("best.pt")

src = cv2.imread("test_images/l_light_01_short_03_1_600.jpg")
if src is not None:
    cv2.imwrite("test_images/good_pcb_clean.jpg", src[:, :320])

files = sorted(glob.glob("test_images/*.jpg"))
print(f"{'Image':<42} | {'Verdict':<7} | {'Detected (Conf)':<30} | {'Time'}")
print("-" * 94)

for path in files:
    t0 = time.time()
    res = model.predict(source=path, imgsz=640, conf=0.30, verbose=False)[0]
    for k, v in list(res.names.items()):
        res.names[k] = CLASS_REMAP.get(v, v)
    dt = time.time() - t0
    dets = [f"{res.names[int(b.cls[0])]}({float(b.conf[0]):.2f})" for b in res.boxes]
    verdict = "FAIL" if dets else "PASS"
    base = os.path.basename(path)
    cv2.imwrite(f"annotated_{base}", res.plot())
    print(f"{base:<42} | {verdict:<7} | {', '.join(dets) or 'none':<30} | {dt:.2f}s")
