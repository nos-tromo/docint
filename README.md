# Wizard

## Frontend

```bash
cd frontend
npm install
```

Run frontend:

```bash
npm run dev
```

## Backend

```bash
cd backend
uv venv
uv pip install .
uv sync
```

Run backend:

```bash
uv run uvicorn wizard.app:app --reload
```

## Docker setup

Select whether to use the CPU or GPU (requires a CUDA compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up):

```bash
# CPU
docker compose up
# GPU
docker compose up --build --gpus all
```

Launch app: `http://localhost:8080/`
