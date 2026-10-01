# ContextForge

> Work in progress. Functionality has not been implemented yet.

## Project Structure

```
ContextForge/
├── app/                  # Application entry point / interface
├── src/
│   ├── ingestion/        # Document loading and preprocessing
│   ├── retrieval/        # Embedding, indexing, and search
│   ├── qa/               # Question answering
│   ├── generation/       # Response generation
│   ├── evaluation/       # Evaluation and metrics
│   └── utils/            # Shared helpers
├── tests/                # Test suite
├── notebooks/            # Exploration and experiments
├── data/
│   ├── raw/              # Original source data
│   ├── processed/        # Cleaned / chunked data
│   └── vector_store/     # Persisted vector indexes
├── configs/              # Configuration files
└── docs/                 # Documentation
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
cp .env.example .env
```
