# Gossip Handshake Protocol — Interactive POC

An interactive web demo that demonstrates the **Gossip Handshake Protocol**: a routing-based alternative to LoRA weight-space merging for decentralised knowledge sharing.

## What It Does

1. **You type a question** about African agriculture (crops, livestock, irrigation, soil, or aquaculture)
2. **The router classifies it** to the correct specialist domain using keyword matching or cosine similarity
3. **The correct LoRA adapter is loaded** via PEFT hot-swap
4. **The specialist generates a response** with domain-specific knowledge
5. **Optionally compare** with the TIES-merged model to see how merging destroys specialised knowledge

## Architecture

```
┌─────────────────┐         ┌──────────────────────────┐
│   React Frontend│ ──API── │   FastAPI Backend         │
│   (Vite :5173)  │         │   (uvicorn :8000)         │
│                 │         │                           │
│  • Chat UI      │         │  • Keyword/Cosine Router  │
│  • Routing Viz  │         │  • Model + 5 LoRA Adapters│
│  • Protocol     │         │  • TIES Merged Adapter    │
│    Diagram      │         │  • Generation Engine      │
└─────────────────┘         └──────────────────────────┘
```

## Quick Start

```bash
# From the poc/ directory:
chmod +x start.sh
./start.sh        # default: 0.5B model
./start.sh 1.5B   # use 1.5B model instead
```

Open [http://localhost:5173](http://localhost:5173) once the model has loaded.

## Manual Start

### Backend

```bash
cd backend
pip install -r requirements.txt

# Start with default 0.5B model
python main.py

# Or specify 1.5B
GH_MODEL_SCALE=1.5B python main.py
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Controls

| Control | Options | Description |
|---------|---------|-------------|
| **Model** | 0.5B / 1.5B | Switch between model scales (reloads model) |
| **Router** | Keyword / Cosine | Keyword matching or embedding similarity |
| **Compare** | On / Off | Also run query through TIES-merged model |

## Requirements

- **Python 3.10+** with PyTorch, Transformers, PEFT
- **Node.js 18+** with npm
- **Pre-trained adapters** in `../adapters/` (0.5B) and `../adapters_1.5B/` (1.5B)
- ~4GB RAM for 0.5B, ~8GB for 1.5B

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GH_MODEL_SCALE` | `0.5B` | Initial model scale |
| `GH_TEMPERATURE` | `0.3` | Generation temperature |
| `GH_TOP_P` | `0.9` | Top-p sampling |
| `GH_MAX_TOKENS` | `256` | Max new tokens |
| `GH_CORS_ORIGINS` | `localhost:5173` | Allowed CORS origins |
| `GH_RATE_LIMIT` | `10` | Requests per minute |
