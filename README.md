# Biomedical NLP Knowledge Extractor

An AI-augmented, publication-grade pipeline for structured extraction of biomedical knowledge from scientific literature. Uses a hybrid **Regex + LLM** architecture to extract interaction data (e.g., drug-target, compound-CYP relationships) with high precision and recall.

---

## Overview

This pipeline transforms unstructured biomedical text (PubMed abstracts, PDF articles) into structured, queryable data. It combines the reliability of pattern matching with the semantic understanding of local LLM inference via [Ollama](https://ollama.com/).

### Architecture

```
Query Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — AI Prompt Refinement (Ollama)                    │
│  Normalizes user query into canonical terms & mechanisms    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 — Conservative Regex Pre-filter                    │
│  Broad multi-pattern filter (zero false negatives)          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 — AI Structured Extraction (Ollama per sentence)   │
│  Compound, target, mechanism, IC50, Ki, confidence           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 — JSON + CSV Persistence                           │
│  Per-file JSON, master JSON, two CSVs (compound/mechanism)  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 — Self-contained HTML Dashboard                    │
│  Interactive charts, filterable tables, CSV/JSON download   │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **Query-Driven** — Describe what to extract in plain English
- **Hybrid AI + Regex** — Zero false negatives, LLM-quality extraction
- **Local LLM** — Full privacy, no API costs, Ollama handles inference
- **Structured Output** — JSON + CSV for downstream analysis
- **Interactive Dashboard** — Self-contained HTML, no server required
- **Confidence Scoring** — Per-record quality assessment (0.0–1.0)
- **Knowledge Memory** — Learns from previous extractions for consistency

---

## Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running

### Option 1: Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate biomedical_nlp

# Install Ollama
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: Download from https://ollama.com/download
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

### Download LLM Model

```bash
# Start Ollama server
ollama serve

# Pull a model (3B = fast, 8B = better quality)
ollama pull llama3.2:3b    # Recommended for speed
# OR
ollama pull llama3.1:8b    # Higher quality, slower
```

---

## Quick Start

### 1. Prepare Input

Place `.txt` files containing PubMed abstracts (or any biomedical text) in a folder. Filenames should contain PubMed IDs for traceability.

```
input/
├── 12345678.txt
├── 87654321.txt
└── ...
```

### 2. Run the Pipeline

```bash
# Interactive mode
python biomedical_nlp_pipeline.py --folder ./input --query "CYP3A4 irreversible inhibitors"

# With explicit model
python biomedical_nlp_pipeline.py \
    --folder ./input \
    --query "time-dependent CYP inhibitors" \
    --model llama3.1:8b

# Skip AI prompt refinement (faster, permissive defaults)
python biomedical_nlp_pipeline.py --folder ./input --query "" --no-prompt
```

### 3. View Results

Open the generated dashboard in your browser:
```
results/dashboard/dashboard.html
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/csv/cyp_inhibitors_compound.csv` | Compound-level: compound + CYP + mechanism + IC50/Ki |
| `results/csv/cyp_mechanism_general.csv` | Mechanism-level: CYP + mechanism (no compound) |
| `results/master_extractions.json` | All records merged into one JSON |
| `results/json/<pubmed_id>.json` | Per-abstract structured JSON |
| `results/knowledge_memory.json` | Learned facts for consistent extraction |
| `results/dashboard/dashboard.html` | Interactive HTML dashboard |

---

## Extraction Fields

| Field | Description |
|-------|-------------|
| `compound_name` | Drug/chemical name (NER + AI extraction) |
| `cyp_isoform` | Canonical CYP (e.g., CYP3A4, CYP2D6) |
| `mechanism_of_inhibition` | mechanism-based / irreversible / time-dependent / reversible |
| `inhibition_type` | competitive / non-competitive / suicide / mixed |
| `IC50` / `Ki` / `kinact` / `KI` | Quantitative inhibition constants |
| `confidence` | 0.0–1.0 quality score |
| `evidence_sentence` | Source sentence from text |
| `pubmed_id` | Source document identifier |

---

## Query Examples

| Query | Targets |
|-------|---------|
| `"CYP3A4 irreversible inhibitors"` | CYP3A4 + irreversible mechanisms only |
| `"time-dependent inhibition IC50 shift"` | TDI across all CYP isoforms |
| `"mechanism-based inactivation kinact KI"` | MBI with kinetic parameters |
| `"reactive metabolite formation CYP2C9"` | Reactive metabolites + CYP2C9 |
| `""` (empty) | All CYP inhibition data |

---

## Dashboard Preview

The self-contained HTML dashboard provides:
- Summary statistics cards (total records, CYP distribution, confidence histogram)
- Filterable data tables
- Mechanism breakdown charts
- Confidence score visualization
- Download buttons for CSV and JSON exports

---

## Citation

If you use this pipeline in academic work, please cite:

```bibtex
@software{biomedical_nlp_extractor,
  title = {AI-Augmented Biomedical NLP Knowledge Extractor},
  author = {Gaurav Verma},
  year = {2025},
  url = {https://github.com/[user]/Biomedical_NLP_Extractor}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) file.
