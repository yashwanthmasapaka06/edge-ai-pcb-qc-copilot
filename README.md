# Offline Edge AI PCB Quality-Control & Rework Copilot (Raspberry Pi 4B)

**ARM Bharat AI-SoC Student Challenge 2026 — Problem Statement 5**  
**Institute:** BV Raju Institute of Technology (BVRIT) | **Team Lead:** Yashwanth (`23211a04e4@bvrit.ac.in`)

## Overview
An air-gapped, two-stage Edge AI optical inspection and IPC-A-610 rework assistant running standalone on a **4GB Raspberry Pi 4 Model B (Arm Cortex-A72)** with live **Sony UVC (`CEVCECM`)** camera capture and zero cloud dependency.

- **Stage 1 — Optical Defect Detector (`YOLOv8n` + `Tencent NCNN`):** Inspects `416x416` central ROI frames across 6 bare-board PCB defect classes (`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`) at **169.8 ms/frame (~5.9 FPS)** using Arm NEON SIMD kernels.
- **Stage 2 — Vision-Gated Rework Copilot (`Llama 3.2 1B` + `ExecuTorch 8da4w`):** Defect-free (`PASS`) boards bypass the LLM with **0.00 s** overhead. Defect-positive (`FAIL`) detections trigger quantized `8da4w` XNNPACK reasoning on-device to generate root-cause diagnosis and 3-step IPC-A-610 rework guidance.

## Repository Structure
```text
├── dashboard.py               # Flask operator console (port 5000)
├── qc_assistant.py            # Two-stage NCNN + ExecuTorch inference & CSV logger
├── best_ncnn_model/           # Compiled YOLOv8n NCNN graph (.param, .bin, metadata.yaml)
├── test_images/               # 7-sample baseline PCB validation suite
├── annotated/                 # Verified bounding-box outputs (baseline + live Sony UVC)
├── inspection_log.csv         # Persistent shop-floor audit log
├── requirements.txt           # Python runtime dependencies
└── Final_Project_Report.pdf   # Complete 10-page technical project report
```

## Setup & Execution on Raspberry Pi 4B
1. Install dependencies inside a Python virtual environment:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place your exported `8da4w` Llama 3.2 1B ExecuTorch artifact (`model.pte`, ~915 MB) and `tokenizer.json` in the local model directory referenced by `qc_assistant.py` (excluded from Git due to GitHub's 100 MB file limit).
3. Launch the inspection dashboard (`http://<PI_IP>:5000`):
   ```bash
   python3 dashboard.py
   ```

## Validation Highlights & Roadmap
- **Phase 1 Results:** **100% accuracy** across the 7-sample baseline suite, 4 unseen out-of-distribution (OOD) test boards, and live Sony UVC (`CEVCECM`) captures (`0.83` live confidence on `missing_hole`), running within **~1.35 GB active RSS** out of 4.00 GB LPDDR4 RAM.
- **Phase 2 Roadmap:** Transitioning from calibrated matte PCB targets to physical bare and populated PCBs under diffused LED ring lighting (solder bridges, tombstoning), decoupling the 170 ms vision gate onto an asynchronous conveyor worker queue, and porting the `.pte` binary to Armv9 SoCs with KleidiAI `SDOT`/`I8MM` acceleration.
