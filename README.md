## Local Dev (Stage 1)

### 1. Setup

```bash
cd ai_deployment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn app.api:app --reload
```

By default, it will:

- Try to detect CUDA.
- On unsupported GPU (like MX250), automatically fall back to cpu.
- Log device decisions with [MobileSAM] ... and [YOLO] ....

Open the interactive docs at:
- http://127.0.0.1:8000/docs

### 3. Run the API

```bash
TEST_IMG=./test_images/board_01.jpg

# YOLO segmentation
curl -X POST "http://127.0.0.1:8000/yolo/predict" \
  -F "file=@${TEST_IMG}"

# SAM overlay + stats
curl -X POST "http://127.0.0.1:8000/sam/segment-overlay-with-stats" \
  -F "file=@${TEST_IMG}"

# Combined void-rate analysis
curl -X POST "http://127.0.0.1:8000/analyze/void-rate" \
  -F "file=@${TEST_IMG}"

# Health check (devices)
curl "http://127.0.0.1:8000/healthz"
```

On my laptop I expect:

```bash
{
  "status": "ok",
  "sam_device": "cpu",
  "yolo_device": "cpu"
}
```

On an Azure GPU node (sm_70+ with DEVICE=cuda):

```bash
{
  "status": "ok",
  "sam_device": "cuda",
  "yolo_device": "cuda"
}
```

### 3. Test all the endpoints at once:

```bash
source .venv/bin/activate
export API_TIMEOUT=600
python scripts/test_endpoints.py ./test_images/board_01.jpg
```
