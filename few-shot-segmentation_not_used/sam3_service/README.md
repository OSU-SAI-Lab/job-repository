# SAM3 Service (FastAPI + Redis + Kubernetes)

This service provides object segmentation APIs powered by SAM3 models from Hugging Face:

- **Concept segmentation** using text prompts (`Sam3Model`)
- **Point-based segmentation** using clicked points (`Sam3TrackerModel`)

The service is implemented in [main.py](main.py), containerized with [Dockerfile](Dockerfile), and deployed to NRP Nautilus using [sam3_nrp_deployment.yaml](sam3_nrp_deployment.yaml) and [deploy.sh](deploy.sh).

---

## 1) What this service does

### Endpoints

- `POST /predict`
	- Text mode: send `text_prompts`
	- Point mode: send `x`, `y`
- `POST /predict_box` (legacy)
	- Backward-compatible point endpoint

### Runtime behavior

1. On startup, the app loads:
	 - `facebook/sam3` concept model/processor
	 - `facebook/sam3` tracker model/processor
2. For each request, image content is fetched from Tapis (`https://icicleai.tapis.io`) using request header `token`.
3. For point inference, image embeddings are cached in Redis for 1 hour to speed repeated calls.

---

## 2) Repository files for this service

- [main.py](main.py): FastAPI app and inference logic
- [Dockerfile](Dockerfile): GPU-ready container image
- [sam3_nrp_deployment.yaml](sam3_nrp_deployment.yaml): Kubernetes objects (Redis, API Deployment, Services, Ingress)
- [deploy.sh](deploy.sh): Helper script for deployment

---

## 3) Prerequisites

## Local/Docker prerequisites

- NVIDIA GPU machine recommended
- Docker with GPU runtime support (for container mode)
- Internet access for downloading Hugging Face model weights

## Kubernetes (NRP Nautilus) prerequisites

- `kubectl` configured to your target cluster/namespace
- Access to nodes with matching GPU label (`nvidia.com/gpu.product: NVIDIA-A10`)
- Ingress class `haproxy` available

---

## 4) Run locally (without Kubernetes)

This is useful for development or debugging.

1. Create environment and install dependencies (Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn requests Pillow numpy redis huggingface-hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/transformers
```

2. Start Redis (example):

```bash
docker run -d --name sam3-redis -p 6379:6379 redis:7-alpine
```

3. Export environment variables:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

4. Run API:

```bash
python main.py
```

API starts on `http://0.0.0.0:2128`.

---

## 5) Build and run with Docker

From [smart-labeller/sam3_service](../sam3_service):

1. Build image:

```bash
docker build -t sam3-service:local .
```

2. Run Redis:

```bash
docker run -d --name sam3-redis -p 6379:6379 redis:7-alpine
```

3. Run API container with GPU:

```bash
docker run --gpus all --rm -p 2128:2128 \
	-e REDIS_HOST=host.docker.internal \
	-e REDIS_PORT=6379 \
	--name sam3-api sam3-service:local
```

---

## 6) Deploy to Kubernetes (NRP Nautilus)

The Kubernetes manifest includes:

- Deployment: `redis-server`
- Service: `redis-service`
- Deployment: `sam3-fastapi`
- Service: `sam3-service`
- Ingress: `sam3-ingress`

`sam3-secrets` is created by [deploy.sh](deploy.sh) from the HF token argument.

### Quick deploy

From [smart-labeller/sam3_service](../sam3_service):

```bash
./deploy.sh <HF_TOKEN>
```

The script:

1. Creates/updates secret `sam3-secrets` with `HF_TOKEN`
2. Applies [sam3_nrp_deployment.yaml](sam3_nrp_deployment.yaml)
3. Waits for deployments to become ready

It waits for:

- `deployment/redis-server`
- `deployment/sam3-fastapi`

### Manual deploy

```bash
kubectl create secret generic sam3-secrets --from-literal=HF_TOKEN='<YOUR_TOKEN>' --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f sam3_nrp_deployment.yaml
kubectl wait --for=condition=available --timeout=300s deployment/redis-server
kubectl wait --for=condition=available --timeout=600s deployment/sam3-fastapi
```

### Verify deployment

```bash
kubectl get pods
kubectl get services
kubectl get ingress
kubectl logs deployment/sam3-fastapi -f
```

Expected host (from ingress):

- `https://sam3-sailab.nrp-nautilus.io`

---

## 7) API documentation

## Authentication header

- Header name: `token`
- Value: Tapis token used to download image content

For text mode, token is required. For point mode, token is required only when cache miss happens.

## 7.1 `POST /predict`

### Request body

Common fields:

- `image_id` (string)
- `pipe_id` (string)
- `system_id` (string)
- `image_path` (string)

Choose one mode:

1. **Text mode**
	 - `text_prompts`: array of strings
 	 - Optional `patch_size` (or alias `crop_size`): image is tiled into overlapping patches for inference
 	 - Optional `overlap_ratio` (default `0.2`): overlap used between neighboring patches
2. **Point mode**
	 - `x`: integer
	 - `y`: integer
	 - Optional `patch_size` (or alias `crop_size`): image is center-cropped around `(x, y)` before inference

Do **not** send text + point in the same request.

### Text mode example

```bash
curl -X POST "http://localhost:2128/predict" \
	-H "Content-Type: application/json" \
	-H "token: <TAPIS_TOKEN>" \
	-d '{
		"image_id": "img_123",
		"pipe_id": "pipe_456",
		"system_id": "my_tapis_system",
		"image_path": "/path/to/image.jpg",
		"text_prompts": ["person", "chair", "red bottle"],
		"patch_size": 960,
		"overlap_ratio": 0.2
	}'
```

### Point mode example

```bash
curl -X POST "http://localhost:2128/predict" \
	-H "Content-Type: application/json" \
	-H "token: <TAPIS_TOKEN>" \
	-d '{
		"image_id": "img_123",
		"pipe_id": "pipe_456",
		"system_id": "my_tapis_system",
		"image_path": "/path/to/image.jpg",
		"x": 420,
		"y": 315,
		"crop_size": 960
	}'
```

### Typical response shape

```json
{
	"bboxes": [
		{
			"x_min": 100,
			"y_min": 120,
			"x_max": 260,
			"y_max": 390,
			"confidence": 0.94,
			"label": "person",
			"prompt_index": 0
		}
	],
	"prompt_type": "text",
	"prompt_count": 3,
	"total_detections": 1,
	"model": "SAM3-Concept",
	"total_time_seconds": 1.237,
	"per_prompt_breakdown": {
		"person": { "detections": 1, "time_seconds": 0.42 }
	}
}
```

## 7.2 `POST /predict_box` (legacy)

Backward-compatible point-based endpoint.

```bash
curl -X POST "http://localhost:2128/predict_box" \
	-H "Content-Type: application/json" \
	-H "token: <TAPIS_TOKEN>" \
	-d '{
		"image_id": "img_123",
		"pipe_id": "pipe_456",
		"system_id": "my_tapis_system",
		"image_path": "/path/to/image.jpg",
		"x": 420,
		"y": 315
	}'
```

---

## 8) Error cases

`/predict` returns HTTP `400` when:

- neither text prompts nor point coordinates are provided
- both text prompts and point coordinates are provided
- token is missing and image fetch is required

`/predict` returns HTTP `500` for inference failures.

---

## 9) Operations and troubleshooting

- Check pod logs:

```bash
kubectl logs deployment/sam3-fastapi -f
```

- Check Redis pod:

```bash
kubectl logs deployment/redis-server -f
```

- Restart API deployment:

```bash
kubectl rollout restart deployment/sam3-fastapi
```

- Delete all service resources:

```bash
kubectl delete -f sam3_nrp_deployment.yaml
```

---

## 10) Security note

No Hugging Face token is stored in [sam3_nrp_deployment.yaml](sam3_nrp_deployment.yaml).

Use [deploy.sh](deploy.sh) with runtime argument:

```bash
./deploy.sh <HF_TOKEN>
```

Recommended hardening:

1. Pass token from a CI secret manager.
2. Avoid shell history leaks (for example, use a temporary env var and unset it).
3. Rotate token periodically.

