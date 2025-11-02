# Streamlining Legal Document Analysis for Compliance in Financial Services

## 🎯 Project Overview

This is a 4-week Proof of Concept (PoC) demonstrating automated ingestion, extraction, and querying of legal/regulatory documents using open-source GenAI on Databricks Free Edition.

### Goal
Automate the analysis of legal/regulatory documents (e.g., SEC filings) using open-source GenAI tools, enabling fast extraction of key attributes and natural language querying.

### Platform & Tech Stack
- **Platform**: Databricks Free Edition (2025)
- **Data Storage**: Delta Lake
- **Processing**: PySpark, PyMuPDF, PyTesseract
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS
- **LLM**: Llama 3 / Mistral (via Hugging Face)
- **RAG Framework**: LangChain
- **Workflow**: Databricks Workflows
- **Visualization**: Streamlit

### Dataset
- SEC EDGAR filings (public data)
- Open legal contract datasets (GitHub/Kaggle)

## 📁 Repository Structure

```
genai-legal-doc-poc/
│
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_ingest_sec_data.ipynb
│   ├── 02_preprocess_text_ocr.ipynb
│   ├── 03_embeddings_faiss_index.ipynb
│   ├── 04_rag_pipeline_llm.ipynb
│   └── 05_reporting_streamlit.ipynb
│
├── src/
│   ├── ingest/
│   │   └── sec_ingest.py
│   ├── process/
│   │   ├── ocr_text_extractor.py
│   │   └── text_cleaner.py
│   ├── rag/
│   │   ├── embed_faiss.py
│   │   ├── query_langchain.py
│   │   └── ensemble_validation.py
│   └── utils/
│       ├── delta_helpers.py
│       ├── config.py
│       └── logger.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── models/
│   └── faiss_index/
│
├── workflows/
│   └── dag_workflow.json
│
└── dashboards/
    └── streamlit_app.py
```

## 🗓️ 4-Week Implementation Timeline

### Week 1 — Data Ingestion & Foundation Setup
**Goal**: Establish Databricks workspace, source data, and base architecture.

**Tasks**:
1. Setup Environment — Create Databricks Free workspace, cluster, repos
2. Data Source Connection — Use `sec-edgar-downloader` to fetch sample filings
3. Bronze Layer Ingestion — Load raw documents into Delta tables
4. Metadata Logging — Create Delta table for document tracking

**Deliverable**: 
- ✅ Databricks repo initialized
- ✅ Raw data ingested into Bronze Delta layer
- ✅ Metadata captured for all documents

### Week 2 — Document Processing (OCR + Text Extraction + Cleaning)
**Goal**: Process raw files into structured text for embedding.

**Tasks**:
1. OCR / Text Extraction — PyMuPDF for PDFs, pytesseract for scanned images
2. Data Normalization — Standardize document structure
3. Language Detection & Translation — Langdetect + Hugging Face translation models
4. Store in Silver Layer — Write cleaned text to Delta

**Deliverable**:
- ✅ Text extraction notebook (OCR + parsing)
- ✅ Cleaned, normalized text stored in Silver
- ✅ Translation and preprocessing pipeline ready

### Week 3 — Embedding, RAG & LLM Pipeline
**Goal**: Build retrieval + generation workflow.

**Tasks**:
1. Chunking & Embeddings — Split documents into ~500-token chunks; compute embeddings
2. FAISS Integration — Store embeddings + metadata in FAISS index
3. RAG Pipeline — LangChain for retrieval + generation
4. Ensemble Validation — Run two LLMs and compute agreement rate

**Deliverable**:
- ✅ Working RAG pipeline (retrieval + generation)
- ✅ FAISS index stored in DBFS
- ✅ Validated extractions stored in Delta

### Week 4 — Automation, Visualization & Reporting
**Goal**: Package, automate, and visualize results.

**Tasks**:
1. Orchestration — Automate pipeline with Databricks Workflows
2. Dashboard & Reports — Build Streamlit app for natural language queries
3. Audit & Traceability — Use Delta history for logging
4. Documentation — Add README, architecture diagram, results summary

**Deliverable**:
- ✅ End-to-end automated pipeline
- ✅ Interactive dashboard
- ✅ Documentation + presentation-ready summary

## 🚀 Getting Started

### Prerequisites
- Databricks Free Edition account
- Python 3.9+
- Access to SEC EDGAR database (public)

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up Databricks workspace**:
   - Create a new workspace in Databricks Free Edition
   - Create a cluster (or use default)
   - Set up a repository and import this project

4. **Configure environment variables**:
   - Copy `src/utils/config.py` and update paths as needed
   - Set up DBFS mount points for data storage

5. **Run notebooks in order**:
   - Start with `01_ingest_sec_data.ipynb`
   - Follow the sequence through `05_reporting_streamlit.ipynb`

## 📊 Key Features

### 1. Automated Data Ingestion
- Fetch SEC filings automatically using `sec-edgar-downloader`
- Store raw documents in Bronze Delta layer with full audit trails

### 2. Intelligent Document Processing
- OCR for scanned documents and handwritten text
- Multi-language support (English, French, German, etc.)
- Automatic translation to English for consistent processing
- Text normalization and cleaning

### 3. RAG Pipeline
- Document chunking and embedding generation
- FAISS vector search for millisecond similarity lookups
- LangChain-powered retrieval and generation
- Ensemble validation for improved accuracy

### 4. Natural Language Querying
- Streamlit dashboard for interactive queries
- Example: "Show all data privacy clauses in 2024 Q2 filings"
- Chatbot interface for document insights

### 5. Full Traceability
- Delta Lake history for audit trails
- Model version tracking
- Extraction output logging
- Run time and failure tracking

## 🎯 Expected Results

Based on the case study, similar implementations have achieved:
- **95% reduction** in processing time
- **90% accuracy** (validated by SMEs)
- **92% reduction** in full process time
- Seamless multilingual processing
- Scalable architecture for handling large document volumes

## 🔧 Configuration

Update `src/utils/config.py` with your specific settings:
- Databricks paths and mount points
- Model configurations
- Embedding dimensions
- Chunk sizes
- LLM parameters

## 📝 Notes

- This PoC uses only open-source tools and free Databricks compute
- For production, consider upgrading to paid Databricks for better performance
- Optional: Integrate OpenAI API or Databricks Foundation Models for enhanced capabilities
- Fine-tuning LLMs on specialized compliance terms can improve accuracy

## 🤝 Contributing

This is a learning project. Feel free to extend it with:
- Additional document types
- More sophisticated RAG strategies
- Fine-tuned models for legal domain
- Enhanced visualization dashboards

## 📄 License

This project is for educational purposes. Ensure compliance with SEC EDGAR data usage terms.

## 🔗 References

- [Databricks Documentation](https://docs.databricks.com/)
- [SEC EDGAR](https://www.sec.gov/edgar)
- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)

