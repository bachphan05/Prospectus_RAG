# Intelligent Document Processing and Analysis System

A comprehensive enterprise-grade solution for extracting, analyzing, and querying financial data from investment fund prospectuses. This system leverages advanced Large Language Models (LLMs) including Google Gemini 2.0 Flash and Mistral Large, combined with vector-based Retrieval-Augmented Generation (RAG) for deep document understanding.

## System Architecture

### Core Components
- **Backend**: Django REST Framework (DRF) serving as the orchestration layer.
- **Frontend**: React.js with Vite, providing a responsive interface for document management and analytics.
- **Database**: PostgreSQL with `pgvector` extension for storing structured relational data and high-dimensional vector embeddings.
- **AI Engine**: Hybrid integration of Google Gemini 2.0 and Mistral AI for OCR, data extraction, and semantic reasoning.

### Workflow
1. **Ingestion**: PDF documents are uploaded, validated, and securely stored.
2. **Preprocessing**: 
   - **Optimization**: Smart page segmentation identifies and isolates high-value pages (e.g., fee schedules, portfolio tables) to reduce context window usage.
   - **Hybrid OCR**: Routing logic selects between Text-PDF parsing and Mistral/Gemini vision capabilities for scanned documents.
3. **Extraction**: Structured financial data (NAV, fees, portfolio holdings) is extracted into normalized JSON schemas.
4. **Vectorization (RAG)**:
   - Full document text is segmented into semantic chunks.
   - Embeddings are generated using Gemini's text-embedding models.
   - Vectors are indexed in PostgreSQL using HNSW (Hierarchical Navigable Small World) graphs for sub-millisecond similarity search.
5. **Analysis**: Users can query documents via a chat interface. The system retrieves relevant context (RAG) and combines it with structured database records to provide hallucination-free answers.

## Features

### Advanced Data Extraction
- **Multi-Model Support**: Toggle between Gemini 2.0 Flash and Mistral Large/OCR models based on document complexity.
- **Structured Normalization**: Automatically standardizes varying prospectus formats into a unified schema:
  - Fund Identity (Name, Code, Management Company)
  - Fee Structures (Subscription, Redemption, Management, Switching)
  - Portfolio Holdings (Assets, Allocation percentages)
  - Historical Data (NAV History, Dividend Distributions)
- **Visual Grounding**: Generates bounding boxes for extracted fields, allowing users to visually verify data origins on the PDF.

### Retrieval-Augmented Generation (RAG)
- **Context-Aware Chat**: specialized Q&A system capable of answering complex financial questions (e.g., "Explain the risk profile," "Compare fees with industry average").
- **Dual-Source Reasoning**: The answering engine synthesizes information from:
  1. **Structured Data**: For precise queries (fees, codes, dates).
  2. **Unstructured Vector Search**: For qualitative queries (investment strategy, risk factors).
- **Source Attribution**: Citations link responses back to specific pages and text chunks.

### Enterprise Features
- **Audit Logging**: Comprehensive tracking of all data modifications and user actions (`DocumentChangeLog`).
- **Versioning**: Track edits to extracted data with rollback capabilities.
- **Performance Monitoring**: Dashboard for tracking processing success rates, latency, and confidence scores.

## Technical Stack

### Backend
- **Framework**: Django 6.0, Django REST Framework
- **Database**: PostgreSQL 16 + `pgvector`
- **AI/ML**: 
  - Google Gemini SDK
  - Mistral AI SDK
  - `langchain-text-splitters` for RAG chunking
  - `PyMuPDF` (Fitz) & `RapidOCR` for PDF manipulation
- **Async Processing**: Python threading for non-blocking I/O

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: React Hooks & Context
- **Visualization**: `recharts` for financial data, `react-pdf` for rendering

## Setup Instructions

### Prerequisites
- Python 3.12+ (Recommended)
- Node.js 18+
- PostgreSQL 15+ with `pgvector` extension installed
- API Keys for Google Gemini and/or Mistral AI

### Backend Configuration

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Environment Setup**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure C++ Build Tools are installed if compiling specific wheels is required.*

4. **Environment Variables**:
   Copy `.env.example` to `.env` and configure:
   ```env
   SECRET_KEY=your-secure-key
   DATABASE_URL=postgresql://user:pass@localhost:5432/db_name
   GEMINI_API_KEY=your_key
   MISTRAL_API_KEY=your_key
   DJANGO_DEBUG=False
   ```

5. **Database Initialization**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```
   *This will enable the `vector` extension and create schemas.*

6. **Launch Server**:
   ```bash
   python manage.py runserver
   ```

### Frontend Configuration

1. **Navigate**:
   ```bash
   cd frontend
   ```

2. **Install Packages**:
   ```bash
   npm install
   ```

3. **Launch Client**:
   ```bash
   npm run dev
   ```

## API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/` | Upload document (triggers async extraction) |
| `GET` | `/api/documents/{id}/` | Retrieve metadata and extracted JSON |
| `PATCH` | `/api/documents/{id}/` | Manually correct extracted data |
| `POST` | `/api/chat/` | Send RAG-based query to document context |
| `GET` | `/api/documents/{id}/preview-page/{page}/` | Get rendered page with bounding box overlays |
| `GET` | `/api/documents/{id}/change_logs/` | View audit trail of edits |

## Future Improvements

The following roadmap outlines planned enhancements to elevate the system's capabilities:

1.  **Distributed Task Queue**: Migrate from Python threading to Celery/Redis for robust, horizontally scalable background processing.
2.  **Advanced RAG Techniques**:
    -   Implement **Hybrid Search** (combining keyword BM25 with dense vector embeddings).
    -   Add **Re-ranking** step to optimize context relevance provided to the LLM.
    -   **Multi-Document Querying**: Allow users to ask questions across the entire document corpus (e.g., "Compare the management fees of all Balanced Funds").
3.  **Authentication & Multi-Tenancy**: Implement OAuth2/JWT authentication to support multiple organizations with isolated data context.
4.  **Feedback Loop**: Implement User-in-the-Loop (HITL) fine-tuning, where manual corrections to extracted data are fed back to improve future prompts or fine-tune smaller models.
5.  **CI/CD Pipeline**: GitHub Actions workflows for automated testing, linting, and containerized deployment (Docker/Kubernetes).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
