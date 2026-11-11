# AI-Powered Question Paper Analysis and Automated Question Paper Generation System

> **A Production-Quality Academic Examination Engineering Platform using NLP and Generative AI**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C.svg)](https://pytorch.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-PP--OCRv5-red.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![ReportLab](https://img.shields.io/badge/ReportLab-5.1.0-orange.svg)](https://www.reportlab.com/)
[![Tests](https://img.shields.io/badge/tests-140%20passed-brightgreen.svg)]()

---

## 📌 Project Overview

This application allows academic institutions, examination boards, and educators to:
1. **Upload previous-year question papers** in PDF format (both digital vector text and scanned sheets).
2. **Preprocess and extract text** using PyMuPDF and authentic PaddleOCR (PP-OCRv5) with OpenCV contrast enhancement, CLAHE, and deskewing.
3. **Parse and segment exam structure** into sections, questions, sub-parts, allocated marks, and internal choice (`OR`) groups.
4. **Linguistically normalize questions** using spaCy (`en_core_web_sm`), expanding academic acronyms (e.g., ADT, BST, DFS, BFS, MST) and extracting canonical Bloom's action verbs.
5. **Detect semantic similarity and duplicates** using 384-dimensional dense sentence embeddings (`all-MiniLM-L6-v2`) and connected-component clustering.
6. **Classify questions into syllabus units and topics** using a 3-tier hybrid engine (Keyword Matching + TF-IDF Lexical Relevance + Dense Embedding Similarity).
7. **Map questions to Bloom's Revised Taxonomy cognitive levels** (Remember L1 to Create L6) with positional directive heuristics and calibrated confidence.
8. **Analyze multi-year historical examination patterns**, identifying longitudinal topic recurrence, year-over-year trajectory trends, and paper repetition rates.
9. **Calculate unit-wise marks weightage** and blueprint deviations against target syllabus distributions.
10. **Generate new syllabus-compliant questions** using Google Gemini Generative AI (`google-genai`).
11. **Assemble complete examination papers** adhering to strict blueprint constraints (Part A, Part B with internal choices, balanced unit representation).
12. **Independently validate generated papers** across marks totals, question counts, unit coverage, and absence of semantic duplicate questions.
13. **Export camera-ready, publication-grade PDFs** formatted with formal university headers and dynamic page numbering via ReportLab.
14. **Interact via a full Analytics & Generation Dashboard** in React with live chart visualizations and question authoring workbench.

---

## 🏗️ Architecture & 14 Implemented Modules

| Module # | Module Name | Core Technology | Primary Responsibility |
|:---:|:---|:---|:---|
| **01** | **PDF Upload & Extraction** | PyMuPDF (`fitz`), Pillow | Classifies digital vs scanned pages; extracts text blocks and coordinates; renders pages at 300 DPI. |
| **02** | **OCR & Image Preprocessing** | PaddleOCR, OpenCV | Deskew, Otsu binarization, CLAHE, and PaddleOCR textline recognition. |
| **03** | **Question Segmentation** | Regex Engine, Layout Heuristics | Identifies question numbers, sections, sub-parts `(a)/(b)/(i)`, marks `[X Marks]`, and internal `[OR]` choices. |
| **04** | **NLP Preprocessing** | spaCy (`en_core_web_sm`) | Lemmatization, acronym expansion, tokenization, technical keyword extraction, and action verb discovery. |
| **05** | **Semantic Similarity** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Normalizes embeddings, computes pairwise cosine matrices, identifies near-duplicates ($\ge 0.75$), clusters recurrence. |
| **06** | **Topic & Unit Classifier** | Scikit-Learn TF-IDF, Semantic Vectors | Multi-tier syllabus matching to assign unit numbers, topic labels, and confidence scores. |
| **07** | **Bloom's Taxonomy Classifier** | Directive Heuristics, Linguistic Scoring | Maps prompts to L1 (Remember) through L6 (Create) with directive positioning weights and rationales. |
| **08** | **Historical Question Analysis** | Longitudinal Statistical Engine | Computes yearly topic frequencies, trend directions (increasing/decreasing/stable), and multi-year repetition rates. |
| **09** | **Marks & Weightage Analysis** | Blueprint Deviation Engine | Computes unit marks distribution, percentage weights, question-type breakdown (short/medium/long), and deviations. |
| **10** | **Question Generation Agent** | Google Gemini API (`google-genai`) | Synthesizes authentic exam questions targeted to syllabus topics, marks, Bloom's levels, with marking schemes. |
| **11** | **Question Paper Generator** | Examination Assembly Engine | Assembles complete papers meeting section constraints, unit coverage balance, and internal OR-choice pairing. |
| **12** | **Validation Engine** | Multi-Constraint Verification Rulebase | Independently verifies marks arithmetic, section counts, unit coverage, and guarantees zero duplicate questions. |
| **13** | **PDF Renderer** | ReportLab 5.1 | Builds camera-ready academic exam papers with headers, metadata tables, section formatting, and page numbers. |
| **14** | **Analytics Dashboard** | React 18, Tailwind CSS, Vite | Responsive web interface for paper upload, longitudinal analytics, Blueprint authoring, and PDF export. |

---

## 💻 Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn, SQLAlchemy (Async), PostgreSQL (`asyncpg`), Pydantic v2
- **OCR & Computer Vision**: PaddleOCR PP-OCRv5, OpenCV (`opencv-python-headless`), PyMuPDF (`pymupdf`)
- **NLP & Embeddings**: spaCy (`en_core_web_sm`), Sentence-Transformers (`all-MiniLM-L6-v2`), Scikit-learn, NumPy
- **Generative AI**: Google Gemini (`google-genai` v3.0.0, `gemini-2.5-flash`)
- **Document Output**: ReportLab 5.1 (`reportlab.platypus`, `reportlab.pdfgen`)
- **Frontend**: React 18, Tailwind CSS, Lucide icons, Vite
- **Testing**: Pytest, Pytest-Asyncio, HTTPX

---

## 🚀 Getting Started

### 1. Environment Configuration

Create a `.env` file in the project root:

```env
# Application Settings
APP_NAME="Question Paper Analysis & Generation System"
APP_VERSION="1.0.0"
DEBUG=false
LOG_LEVEL=INFO

# Database Connection
DATABASE_URL="postgresql+asyncpg://qpaper_user:qpaper_pass@localhost:5432/qpaper_db"
DATABASE_URL_SYNC="postgresql+psycopg2://qpaper_user:qpaper_pass@localhost:5432/qpaper_db"

# Google Gemini API Key
GEMINI_API_KEY="your-gemini-api-key-here"
GEMINI_MODEL="gemini-2.5-flash"

# File Directories
UPLOAD_DIR="./uploads"
OUTPUT_DIR="./outputs"
```

### 2. Start PostgreSQL Database

Using Docker:

```bash
docker run --name qpaper_postgres -e POSTGRES_USER=qpaper_user -e POSTGRES_PASSWORD=qpaper_pass -e POSTGRES_DB=qpaper_db -p 5432:5432 -d postgres:16
```

### 3. Run the Backend API & Interactive Dashboard

Activate your Python virtual environment and run Uvicorn:

```bash
# Using the workspace Python environment
PYTHONPATH=. .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- **Interactive Web Dashboard**: Open [http://localhost:8000/](http://localhost:8000/) in any web browser.
- **Swagger Interactive API Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs).
- **Health Check**: Open [http://localhost:8000/health](http://localhost:8000/health).

---

## 🧪 Running the Test Suite

The test suite contains **140 comprehensive unit and integration tests** verifying all 14 modules with zero mocked ML outputs:

```bash
PYTHONPATH=. .venv/bin/python -m pytest backend/tests/ -v --tb=short
```

Output:
```
================== 140 passed in 60.55s (0:01:00) ==================
```

---

## 📡 REST API Endpoints Reference

### Papers (`/api/v1/papers`)
- `POST /api/v1/papers/upload`: Upload PDF (multipart/form-data), trigger OCR/text extraction, segment questions, and save to DB.
- `GET /api/v1/papers`: List all uploaded examination papers.
- `GET /api/v1/papers/{paper_id}`: Retrieve detailed paper record with page-by-page extractions.
- `GET /api/v1/papers/{paper_id}/questions`: List segmented questions extracted from paper.
- `DELETE /api/v1/papers/{paper_id}`: Delete paper record and remove uploaded files from disk.

### Analysis & Mining (`/api/v1/analysis`)
- `POST /api/v1/analysis/classify-question`: Run 3-tier syllabus topic classification and Bloom's cognitive level classification on an arbitrary question prompt.
- `GET /api/v1/analysis/historical?subject=...`: Aggregate historical trends, topic occurrence counts, trend trajectories, and repetition rates.
- `GET /api/v1/analysis/weightage?subject=...`: Compute unit-wise marks distributions, percentage weights, and syllabus blueprint deviations.
- `POST /api/v1/analysis/duplicates`: Perform dense sentence embedding clustering to identify recurring and near-duplicate questions.

### Question & Paper Generation (`/api/v1/generation`)
- `POST /api/v1/generation/generate-question`: Generate an individual exam question targeted to specific topics and Bloom's levels using Gemini GenAI.
- `POST /api/v1/generation/generate-paper`: Assemble a complete examination paper adhering to blueprint constraints and syllabus coverage.
- `POST /api/v1/generation/validate-paper`: Independently verify paper constraints (marks arithmetic, question counts, unit balance, zero duplicate questions).
- `POST /api/v1/generation/render-pdf`: Export generated paper into a camera-ready, downloadable PDF document.

### Syllabus (`/api/v1/syllabus`)
- `GET /api/v1/syllabus/default`: Retrieve the standard reference syllabus (Data Structures & Algorithms).
- `GET /api/v1/syllabus`: List all syllabi in database.
- `POST /api/v1/syllabus`: Create a custom subject syllabus with unit keywords and topics.

---

## 📄 License
Academic & Commercial Open Source License.
