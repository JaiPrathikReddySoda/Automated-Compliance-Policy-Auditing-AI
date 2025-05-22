# Policy Audito

A powerful policy document Q&A system using DeBERTa-v3 and FAISS for semantic search.

## Features

- 🤖 INT8-quantized DeBERTa-v3 model for efficient inference
- 🔍 FAISS HNSW vector store for fast semantic search
- 📄 Support for PDF and text documents
- 🌐 URL indexing capability
- 🐳 Docker support for both CPU and GPU
- ✅ Comprehensive test coverage
- 🔄 CI/CD pipeline with GitHub Actions

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audit-ai.git
   cd audit-ai
   ```

2. Build and run with Docker Compose:
   ```bash
   docker compose up --build
   ```

3. Ask a question:
   ```bash
   curl -X POST http://localhost:8000/api/v1/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is personal data under GDPR?"}'
   ```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │     │  DeBERTa-v3 │     │   FAISS     │
│   Server    │────▶│   Model     │────▶│  HNSW Index │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   ▲                   ▲
       │                   │                   │
       ▼                   │                   │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Document   │     │  Sentence   │     │  Document   │
│  Upload     │────▶│ Transformer │────▶│  Chunking   │
└─────────────┘     └─────────────┘     └─────────────┘
```

## API Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/ask` - Ask a question
- `POST /api/v1/upload` - Upload a document or URL

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. Run tests:
   ```bash
   pytest
   ```

4. Build the index:
   ```bash
   python -m scripts.build_index
   ```

## Docker Images

- CPU: `ghcr.io/yourusername/audit-ai:cpu`
- GPU: `ghcr.io/yourusername/audit-ai:gpu`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
