import csv
from datetime import datetime
import gc
import os
import sys
import time
import cv2
import torch
from ultralytics import YOLO
import executorch.kernels.quantized
from executorch.runtime import Runtime
from tokenizers import Tokenizer

torch.set_num_threads(4)

YOLO_WEIGHTS = "best.pt"
LLAMA_PTE = "model.pte"
TOKENIZER_PATH = "tokenizer.json"

CLASS_REMAP = {}

DEFECT_NOTES = {
    "short": "unintended copper/solder bridge connecting adjacent conductors (IPC-A-610 Class 2/3 fail)",
    "open_circuit": "broken copper trace or missing conductive path causing electrical discontinuity",
    "missing_hole": "un-drilled or obstructed via/through-hole pad preventing component lead insertion",
    "mouse_bite": "small circular indentations along trace edges from depaneling stress, risking high resistance",
    "spur": "sharp unintended copper protrusion extending from a trace, risking shorting/arcing",
    "spurious_copper": "isolated residual copper island left on substrate due to incomplete etching",
}


def detect_defects(image_path, conf_thresh=0.35):
    model = YOLO(YOLO_WEIGHTS)
    results = model.predict(source=image_path, imgsz=640, conf=conf_thresh, verbose=False)
    res = results[0]
    for k, v in list(res.names.items()):
        res.names[k] = CLASS_REMAP.get(v, v)
    cv2.imwrite("annotated_pcb.jpg", res.plot())

    detections = []
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        detections.append({
            "defect": CLASS_REMAP.get(res.names[cls_id], res.names[cls_id]),
            "confidence": round(float(box.conf[0].item()), 3),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        })
    del results, model
    gc.collect()
    return detections


def build_prompt(detections):
    if not detections:
        return None, "**Verdict:** PASS\n**Root Cause:** Board meets IPC-A-610 visual criteria.\n**Rework Steps:**\n- No rework required; release board to next stage."

    summary = ", ".join(f"{d['defect']} ({d['confidence']:.2f})" for d in detections)
    prefill = "**Verdict:** FAIL\n**Root Cause:**"
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "IPC-A-610 engineer: write 1 brief Root Cause sentence, then '**Rework Steps:**' with 2 short bullets.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"PCB defects: {summary}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{prefill}"
    )
    return prompt, prefill


def capture_usb(dev_idx=0, out_path="usb_capture.jpg"):
    cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open /dev/video{dev_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = False, None
    for _ in range(12):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to grab frame from /dev/video{dev_idx}")
    h, w = frame.shape[:2]
    frame = frame[int(h * 0.18):int(h * 0.82), int(w * 0.18):int(w * 0.82)]
    cv2.imwrite(out_path, frame)
    return out_path


def log_inspection(source, detections, t_yolo, t_llm, report, csv_path="inspection_log.csv"):
    exists = os.path.exists(csv_path)
    verdict = "FAIL" if detections else "PASS"
    summary = "; ".join(f"{d['defect']}({d['confidence']:.2f})@{d['bbox']}" for d in detections) or "none"
    clean_rep = " | ".join(line.strip() for line in report.splitlines() if line.strip())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "source", "verdict", "defect_count", "defects", "vision_sec", "llm_sec", "ipc_a_610_report"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source, verdict, len(detections), summary, round(t_yolo, 2), round(t_llm, 2), clean_rep])


def generate_rework_advice(detections, max_new_tokens=42):
    prompt, prefill = build_prompt(detections)
    if prompt is None:
        print(prefill)
        return prefill

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    program = Runtime.get().load_program(LLAMA_PTE)
    forward = program.load_method("forward")
    print(prefill, end="", flush=True)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    stop_ids = {tokenizer.token_to_id("<|eot_id|>"), tokenizer.token_to_id("<|end_of_text|>")}

    logits = None
    for pos, tok_id in enumerate(prompt_ids):
        logits = forward.execute([torch.tensor([[tok_id]], dtype=torch.long), torch.tensor([pos], dtype=torch.long)])[0]

    chunks, cur_pos = [prefill], len(prompt_ids)
    for _ in range(max_new_tokens):
        next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        if next_tok in stop_ids:
            break
        piece = tokenizer.decode([next_tok], skip_special_tokens=True)
        print(piece, end="", flush=True)
        chunks.append(piece)
        logits = forward.execute([torch.tensor([[next_tok]], dtype=torch.long), torch.tensor([cur_pos], dtype=torch.long)])[0]
        cur_pos += 1
    print()
    return "".join(chunks)


def main():
    arg = sys.argv[1] if len(sys.argv) >= 2 else ""
    if arg == "cam" or arg.startswith("cam:"):
        dev_idx = int(arg.split(":", 1)[1]) if ":" in arg else 0
        image_path = capture_usb(dev_idx=dev_idx)
        source_label = f"usb_cam:{dev_idx}"
    elif arg and os.path.exists(arg):
        image_path = arg
        source_label = arg
    else:
        raise FileNotFoundError(f"Pass 'cam', 'cam:2', or a valid image path (got: {sys.argv[1:]})")

    print(f"[1/2] Running YOLOv8n PCB inspection on: {image_path}")
    t0 = time.time()
    detections = detect_defects(image_path)
    t_yolo = time.time() - t0
    print(f"      Detections ({t_yolo:.2f}s): {detections}")
    print("      Saved bounding-box overlay to: annotated_pcb.jpg\n")

    print("[2/2] Generating offline rework guidance with Llama 3.2 1B (ExecuTorch)...")
    t1 = time.time()
    report = generate_rework_advice(detections)
    t_llm = time.time() - t1

    log_inspection(source_label, detections, t_yolo, t_llm, report)
    print(f"\n[Done] Vision: {t_yolo:.2f}s | LLM Reasoning: {t_llm:.2f}s | Logged to: inspection_log.csv")


if __name__ == "__main__":
    main()
