import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from executorch.runtime import Runtime
from tokenizers import Tokenizer
from qc_assistant import CLASS_REMAP, LLAMA_PTE, TOKENIZER_PATH, DEFECT_NOTES, capture_usb, log_inspection

LOCK = threading.Lock()
TEST_DIR = "test_images"

print("[Init] Loading YOLOv8n NCNN model (imgsz=416)...", flush=True)
YOLO_MODEL = YOLO("best_ncnn_model", task="detect")

print("[Init] Loading Llama 3.2 1B (ExecuTorch) into RAM...", flush=True)
TOKENIZER = Tokenizer.from_file(TOKENIZER_PATH)
PROGRAM = Runtime.get().load_program(LLAMA_PTE)
FORWARD = PROGRAM.load_method("forward")
STOP_IDS = {TOKENIZER.token_to_id("<|eot_id|>"), TOKENIZER.token_to_id("<|end_of_text|>")}


REWORK_HINTS = {
    "short": "Remove solder/copper bridge with desoldering braid or micro-knife; clean with IPA and verify isolation.",
    "open_circuit": "Scrape soldermask on trace ends, bridge break with 30 AWG jumper wire, solder, and seal with UV mask.",
    "missing_hole": "Drill missing via/through-hole to spec diameter, install eyelet or wire jumper, and verify continuity.",
    "mouse_bite": "Check trace width per IPC-A-610; reinforce narrowed conductor with solder/jumper and coat with epoxy.",
    "spur": "Trim protruding copper spur flush with trace edge using a precision blade; clean residue and check spacing.",
    "spurious_copper": "Scrape isolated residual copper island off laminate without gouging substrate; clean with IPA.",
}


def run_vision(image_path, conf_thresh=0.30):
    t0 = time.time()
    results = YOLO_MODEL.predict(source=image_path, imgsz=416, conf=conf_thresh, verbose=False)
    res = results[0]
    for k, v in list(res.names.items()):
        res.names[k] = CLASS_REMAP.get(v, v)
    plotted = res.plot()
    cv2.imwrite("annotated_pcb.jpg", plotted)
    _, buf = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    detections = []
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        detections.append({
            "defect": res.names[cls_id],
            "confidence": round(float(box.conf[0].item()), 3),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        })
    return detections, round(time.time() - t0, 3), img_b64


def run_llm(detections, max_new_tokens=32):
    if not detections:
        return "**Verdict:** PASS\n**Root Cause:** Board meets IPC-A-610 visual criteria.\n**Rework Steps:**\n- No rework required; release board to next stage.", 0.0
    t0 = time.time()
    primary = detections[0]["defect"]
    cause = DEFECT_NOTES.get(primary, "IPC-A-610 conductor anomaly detected")
    hint = REWORK_HINTS.get(primary, "Inspect under magnification and rework per IPC-A-610.")
    summary = ", ".join(f"{d['defect']} ({d['confidence']:.2f})" for d in detections)
    prefill = f"**Verdict:** FAIL\n**Root Cause:** {cause}.\n**Rework Steps:**\n-"
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "IPC-A-610 technician: complete 2 short bench-rework bullets based on the IPC action. No extra text.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Defects: {summary}. IPC action: {hint}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{prefill}"
    )
    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False).ids
    logits = None
    for pos, tok_id in enumerate(prompt_ids):
        logits = FORWARD.execute([torch.tensor([[tok_id]], dtype=torch.long), torch.tensor([pos], dtype=torch.long)])[0]
    chunks, cur_pos = [prefill], len(prompt_ids)
    for _ in range(max_new_tokens):
        next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        if next_tok in STOP_IDS:
            break
        chunks.append(TOKENIZER.decode([next_tok], skip_special_tokens=True))
        logits = FORWARD.execute([torch.tensor([[next_tok]], dtype=torch.long), torch.tensor([cur_pos], dtype=torch.long)])[0]
        cur_pos += 1
    return "".join(chunks), round(time.time() - t0, 2)

def render_html():
    samples = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(TEST_DIR) else []
    opts = "".join(f'<option value="{s}">{s}</option>' for s in samples)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Edge AI PCB QC Copilot</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;max-width:960px;margin:24px auto;padding:0 16px}}
.card{{background:#1e293b;border-radius:10px;padding:16px;margin-bottom:16px}}
.row{{display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
button,select,input{{padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#0f172a;color:#f8fafc}}
button{{background:#2563eb;border:none;cursor:pointer;font-weight:600}}
button:disabled{{opacity:.5;cursor:wait}}
.pass{{color:#4ade80;font-weight:700}} .fail{{color:#f87171;font-weight:700}}
img{{max-width:100%;border-radius:8px;border:1px solid #334155;margin-top:10px}}
pre{{background:#0f172a;padding:12px;border-radius:8px;white-space:pre-wrap;line-height:1.45}}
</style></head><body>
<h2>Offline Edge AI PCB Quality-Control Copilot (Raspberry Pi 4B)</h2>
<div class="card"><div class="row">
  <select id="smp">{opts}</select>
  <button onclick="go('sample')">Inspect Sample</button>
  <button onclick="go('usb')">Capture USB &amp; Inspect</button>
  <input type="file" id="upl" accept="image/*">
  <button onclick="go('upload')">Inspect Upload</button>
  <a href="/download_csv"><button type="button">Download CSV</button></a>
</div><p id="st">Ready (YOLOv8n NCNN + Llama 3.2 1B warm in RAM).</p></div>
<div class="card" id="out" style="display:none">
  <h3 id="verd"></h3>
  <p id="lat"></p>
  <pre id="rep"></pre>
  <img id="img">
</div>
<script>
async function go(mode){{
  const st=document.getElementById('st'), btns=document.querySelectorAll('button');
  btns.forEach(b=>b.disabled=true);
  st.textContent='Running YOLOv8n NCNN (~0.17s) + gated Llama 3.2 1B (~90s if defect found)...';
  let payload={{mode}};
  if(mode==='sample') payload.filename=document.getElementById('smp').value;
  if(mode==='upload'){{
    const f=document.getElementById('upl').files[0];
    if(!f){{alert('Select an image first');btns.forEach(b=>b.disabled=false);return;}}
    payload.filename=f.name;
    payload.image_b64=await new Promise(r=>{{const fr=new FileReader();fr.onload=()=>r(fr.result);fr.readAsDataURL(f);}});
  }}
  try{{
    const res=await fetch('/api/inspect',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});
    const d=await res.json();
    if(d.error) throw new Error(d.error);
    document.getElementById('out').style.display='block';
    const v=document.getElementById('verd');
    v.textContent='Verdict: '+d.verdict+' ('+d.detections.length+' defect(s))';
    v.className=d.verdict==='PASS'?'pass':'fail';
    document.getElementById('lat').textContent='Source: '+d.source+' | Vision (NCNN): '+d.vision_sec+'s | LLM (ExecuTorch): '+d.llm_sec+'s';
    document.getElementById('rep').textContent=d.report;
    document.getElementById('img').src='data:image/jpeg;base64,'+d.image_b64;
    st.textContent='Completed and logged to inspection_log.csv.';
  }}catch(e){{st.textContent='Error: '+e.message;}}
  btns.forEach(b=>b.disabled=false);
}}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/":
            self._send(200, render_html())
        elif self.path == "/download_csv":
            data = open("inspection_log.csv", "rb").read() if os.path.exists("inspection_log.csv") else b""
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", 'attachment; filename="inspection_log.csv"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/inspect":
            self._send(404, '{"error":"Not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            mode = data.get("mode")
            with LOCK:
                if mode == "sample":
                    fname = os.path.basename(data["filename"])
                    img_path = os.path.join(TEST_DIR, fname)
                    source = img_path
                elif mode == "usb":
                    dev_idx = int(data.get("dev_idx", 0))
                    img_path = capture_usb(dev_idx=dev_idx)
                    source = f"usb_cam:{dev_idx}"
                elif mode == "upload":
                    b64 = data["image_b64"].split(",", 1)[-1]
                    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Invalid uploaded image")
                    img_path = "uploaded_pcb.jpg"
                    cv2.imwrite(img_path, frame)
                    source = f"upload:{os.path.basename(data.get('filename', 'pcb.jpg'))}"
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                print(f"[Inspect] {source}", flush=True)
                detections, t_yolo, img_b64 = run_vision(img_path)
                report, t_llm = run_llm(detections)
                log_inspection(source, detections, t_yolo, t_llm, report)

            out = {
                "source": source,
                "verdict": "FAIL" if detections else "PASS",
                "detections": detections,
                "vision_sec": t_yolo,
                "llm_sec": t_llm,
                "report": report,
                "image_b64": img_b64,
            }
            self._send(200, json.dumps(out), "application/json")
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}), "application/json")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 5000), Handler)
    print("[Ready] Dashboard live at http://raspberrypi.local:5000", flush=True)
    server.serve_forever()
