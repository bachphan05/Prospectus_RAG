# OCR System Architecture & Flow Documentation

**Last Updated:** February 19, 2026

## Table of Contents
1. [System Overview](#system-overview)
2. [Document Processing Pipeline](#document-processing-pipeline)
3. [RAG (Retrieval-Augmented Generation) Pipeline](#rag-pipeline)
4. [RAGAS Evaluation Flow](#ragas-evaluation-flow)
5. [Key Components](#key-components)
6. [Data Flow Diagrams](#data-flow-diagrams)

---

## System Overview

This is a **Vietnamese Investment Fund Prospectus OCR System** that:
- Extracts structured data from PDF documents using AI models (Gemini/Mistral)
- Enables semantic search and chat via RAG (Retrieval-Augmented Generation)
- Evaluates RAG quality using RAGAS framework

**Tech Stack:**
- **Backend:** Django 5.0 + PostgreSQL with pgvector
- **Frontend:** React + Vite
- **Python:** 3.13.5
- **AI Models:**
  - Gemini 2.5 Flash Lite (structured extraction + RAG fallback OCR) — via `google.genai` new SDK
  - Mistral OCR Latest (primary RAG text extraction, always-on)
  - Mistral Large / Mistral OCR + Small (alternative structured extraction)
- **Embedding:** Mistral `mistral-embed-2312` (1024 dimensions)
- **Vector Search:** pgvector with HNSW index (cosine similarity)
- **Keyword Search:** PostgreSQL full-text search with GIN index (BM25-like)
- **Retrieval:** Hybrid (Vector + Keyword) with Reciprocal Rank Fusion (RRF)
- **Reranker:** FlashRank `ms-marco-MiniLM-L-12-v2` (optional)
- **RAG Chat:** Configurable — Ollama (`qwen2.5:7b` default), Gemini, or Mistral

---

## Document Processing Pipeline

### Flow Overview
```
Upload PDF → PDF Optimization → Structured Data Extraction → Storage → (Auto) RAG Ingestion
```

### Stage 1: Document Upload
**Endpoint:** `POST /api/documents/`
**File:** `backend/api/views.py` - `DocumentViewSet.create()`

1. User uploads PDF via frontend
2. Django saves file to `media/documents/YYYY/MM/DD/`
3. Document record created with status `'pending'`
4. Asynchronous processing triggered via background thread
5. RAG ingestion is **auto-triggered** after extraction completes (controlled by `AUTO_RAG_INGEST_ON_UPLOAD`, default `true`)

### Stage 2: PDF Optimization
**Function:** `backend/api/services.py` - `create_optimized_pdf()`

**Purpose:** Filter out irrelevant pages to reduce processing time and cost

**Process:**
1. Open PDF with PyMuPDF (fitz)
2. For each page:
   - **Digital PDF:** Extract text with `page.get_text("text")`
   - **Scanned PDF:** Use RapidOCR to extract text from rendered image
3. Check for Vietnamese keywords (normalized via `unidecode` for robustness):
   - "ban cao bach" (prospectus)
   - "noi dung" (contents)
   - "phi", "phi dich vu" (fees)
   - "quy", "fund"
4. Keep only pages containing >=2 keywords (up to `max_selected_pages`)
5. Create optimized PDF with `garbage=4, deflate=True` compression
6. Save to `media/optimized_documents/YYYY/MM/DD/`
7. Return `(temp_path, page_map)` — `page_map` maps optimized index -> original 1-based page number

**Fallback:** If optimization fails, use original PDF

### Stage 3: Structured Data Extraction
**Service:** `backend/api/services.py` - `DocumentProcessingService._process_document_task()`

**Models Available:**
- **Gemini 2.5 Flash Lite** (`gemini` — default) — uploads PDF via `google.genai` Files API, extracts JSON with bounding boxes
- **Mistral Large** (`mistral`) — Mistral OCR -> Mistral chat for JSON extraction
- **Mistral OCR + Small** (`mistral-ocr`) — same pipeline, different extraction model

**Process:**
1. Choose PDF: optimized (preferred) or original (fallback on failure)
2. Call AI service with detailed extraction schema
3. AI extracts 50+ fields including:
   - Basic info: fund name, code, type, license
   - Fees: management, subscription, redemption, switching, TER, custody, audit, supervisory, other
   - Governance: management company, custodian bank, auditor, regulator
   - Investment: strategy, restrictions, limits, benchmark, style
   - Operations: trading frequency, cut-off time, NAV calculation, settlement cycle
   - Valuation: method, pricing source
   - Performance: portfolio, NAV history, dividends
   - Risk factors: concentration, liquidity, interest rate
4. Remap page numbers from optimized -> original using `page_map`
5. Parse and validate JSON response
6. Store in `Document.extracted_data` (PostgreSQL JSONB)
7. Create/update `ExtractedFundData` normalized model
8. Update status to `'completed'` or `'failed'`

**Extraction Schema:** `_get_extraction_schema()` — fields return `{value, page, bbox}` objects where `bbox` is `[ymin, xmin, ymax, xmax]` on a 0-1000 scale per page.

### Stage 4: Storage
**Models:** `backend/api/models.py`

1. **Document** — Main table
   - File paths (original + optimized + markdown)
   - Processing status and timestamps
   - JSON extracted data (JSONB)
   - Chat history (JSONB)
   - RAG ingestion status + progress (0-100)
   - Edit tracking

2. **ExtractedFundData** — Normalized structured data
   - 50+ fields as database columns
   - Easier querying and filtering

3. **DocumentChangeLog** — Audit trail
   - Tracks all user edits
   - Stores before/after values

---

## RAG Pipeline

### Overview
```
Extract Full Text -> Clean -> Chunk -> Embed -> Store (Vector + ASCII) -> Query -> Hybrid Retrieve -> Rerank -> Generate Answer
```

### Trigger Points
1. **Automatic:** After document processing completes (if `AUTO_RAG_INGEST_ON_UPLOAD=true`)
2. **Manual:** `POST /api/documents/{id}/ingest_for_rag/`
3. **Guard:** Skipped if `rag_status` is already `queued`, `running`, or `completed`

### Stage 1: Full Text Extraction for RAG
**Function:** `RAGService._extract_content_for_rag()`

**Process:**
1. **File Selection:**
   - Prefer ORIGINAL PDF (higher quality)
   - Falls back to optimized if original missing

2. **Mistral OCR (Primary, always-on):**
   ```python
   mistral_service = MistralOCRService()
   markdown_text = mistral_service.get_markdown(chosen_path)
   # Returns page-marked markdown: "=== PAGE N ===\n[content]"
   # Saved to document.markdown_file
   ```
   - Uses `mistral-ocr-latest` with retry logic (4 attempts, exponential backoff)
   - Returns `=== PAGE N ===` markers per page

3. **Fallback (if Mistral OCR fails) — Gemini + PyMuPDF:**
   - For each page (in batches of 20):
     - **Digital pages** (`get_text()` >= 50 chars): direct text extraction
     - **Scanned pages**: render at 1.2x scale -> send to Gemini as images
   - Gemini OCR prompt produces `=== PAGE N ===` markers
   - 3 attempts per batch with exponential backoff

4. **Output Format:**
   ```
   === PAGE 1 ===
   [Page 1 content with Vietnamese accents preserved...]

   === PAGE 2 ===
   [Page 2 content...]
   ```

5. **Debug Output:** Saved to `media/debug_markdown/document_{id}_extracted.md`

### Stage 2: Text Cleaning
**Function:** `RAGService._clean_text_for_rag()`

**Cleans:**
- Standard headers: "UY BAN CHUNG KHOAN NHA NUOC", "BAN CAO BACH", etc.
- Looping phrase glitches via regex: `re.sub(r'(.{10,})\1+', r'\1', text)`

### Stage 3: Chunking Strategy

**Multi-Stage Approach:**

1. **Page Segmentation:**
   - Supports both `=== PAGE N ===` and `--- PAGE N ---` markers

2. **Header-Based Splitting (per page):**
   ```python
   MarkdownHeaderTextSplitter(headers_to_split_on=[
       ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")
   ])
   ```

3. **Recursive Text Splitting:**
   ```python
   RecursiveCharacterTextSplitter(
       chunk_size=800,
       chunk_overlap=100,
       separators=["\n\n", "\n", ".", " ", ""]
   )
   ```

4. **Metadata tagging:**
   ```python
   chunk.metadata['page_number'] = page_num
   ```

**Output:** Typically 200-500 `DocumentChunk` objects per document

### Stage 4: Embedding Generation & Storage
**Embedding Model:** Mistral `mistral-embed-2312` (1024 dimensions)

**Process:**
1. For each batch of 50 chunks:
   ```python
   resp = self.mistral_client.embeddings.create(
       model="mistral-embed-2312",
       inputs=batch_texts,
   )
   ```
2. Add header context (H1/H2) as prefix to chunk content
3. Compute `content_ascii = unidecode(final_content)` per chunk
4. Create `DocumentChunk` with `content`, `content_ascii`, `page_number`, `embedding`
5. Bulk insert every 200 chunks (`bulk_create(batch_size=500)`)
6. Retry logic: 3 attempts, exponential backoff + jitter per batch

**After all chunks saved:**
```python
# Single bulk update — ensures index and query use identical normalization
DocumentChunk.objects.filter(document_id=document_id).update(
    search_vector=SearchVector('content_ascii', config='simple')
)
```

### Stage 5: Vector Storage
**Model:** `backend/api/models.py` — `DocumentChunk`

```python
class DocumentChunk:
    document      = ForeignKey(Document)
    content       = TextField()             # Original chunk text
    content_ascii = TextField(null=True)    # unidecode() version for keyword search
    page_number   = IntegerField()          # Source page
    embedding     = VectorField(dimensions=1024)    # Mistral embed-2312
    search_vector = SearchVectorField(null=True)    # tsvector (simple, ASCII)
    created_at    = DateTimeField()

    indexes = [
        HnswIndex(fields=['embedding'], m=16, ef_construction=64,
                  opclasses=['vector_cosine_ops']),  # Fast ANN search
        GinIndex(fields=['search_vector'])           # Fast keyword search
    ]
```

### Stage 6: Query & Retrieval — Hybrid Search + Reranking
**Endpoint:** `POST /api/documents/{id}/chat/`
**Service:** `RAGService.chat()` -> `RAGService.hybrid_search()` -> `RAGService._rerank_chunks()`

#### 6a. Hybrid Search (RRF)
```python
def hybrid_search(document_id, query_text, top_k=10, k_fusion=60):
    # 1. Semantic (Vector) Search
    query_embedding = mistral_client.embeddings.create(
        model="mistral-embed-2312", inputs=[query_text]
    ).data[0].embedding
    semantic_results = DocumentChunk.objects
        .annotate(distance=CosineDistance('embedding', query_embedding))
        .filter(distance__lt=0.85)
        .order_by('distance')[:30]

    # 2. Keyword Search (BM25-like, normalized)
    ascii_query = remove_vietnamese_diacritics(query_text)
    search_query = SearchQuery(ascii_query, config='simple')
    keyword_results = DocumentChunk.objects
        .filter(document_id=document_id, search_vector=search_query)
        .annotate(rank=SearchRank(F('search_vector'), search_query))
        .order_by('-rank')[:50]

    # 3. Reciprocal Rank Fusion: score = 1 / (k_fusion + rank + 1)
    # 4. Return top_k by fused score
```

**Why `content_ascii` matters:** Both the stored `search_vector` (built from `content_ascii`) and the query (stripped of diacritics) are normalized identically — ensuring "Quy" and "Qu?" match correctly.

#### 6b. Reranking (FlashRank)
```
RAG_ENABLE_RERANK=true
FLASHRANK_MODEL=ms-marco-MiniLM-L-12-v2
RAG_RERANK_TOP_K=5
RAG_RETRIEVAL_CANDIDATES_K=15
```
- Fetches `max(RAG_RETRIEVAL_CANDIDATES_K, RAG_RERANK_TOP_K)` candidates from hybrid search
- Reranks with cross-encoder FlashRank, returns top `RAG_RERANK_TOP_K`
- Falls back to original RRF order if FlashRank unavailable or errors

#### 6c. Prompt & Generation
```
NGUON 1 (Structured): {structured_info from ExtractedFundData}
NGUON 2 (Text):       {rag_context = top reranked chunks with page citations}
USER QUERY:           {user_query}
```

**Chat providers (configurable via `RAG_CHAT_PROVIDER`):**

| Provider | Model | Notes |
|----------|-------|-------|
| `ollama` (default) | `qwen2.5:7b` | Local, via OpenAI-compatible API |
| `gemini` | `gemini-2.5-flash-lite` | Via `google.genai` new SDK |
| `mistral` | `mistral-small-latest` | Via Mistral API |

**Response format:**
```json
{
  "text": "Answer text... [Trang 5]",
  "contexts": ["chunk1...", "chunk2..."],
  "structured_data_used": "...",
  "citations": [{"chunk_id": 42, "page": 5, "quote": "..."}]
}
```

---

## RAGAS Evaluation Flow

### Overview
```
Load CSV -> Trim Contexts -> Run Local Metrics -> Save Results
```

### Script Location
`backend/api/evaluation.py`

### Dataset Format
**Input:** `backend/ragas_dataset.csv`

```csv
question,answer,contexts,ground_truth
"Ten quy TCSME?","Quy Dau tu...","['context1', 'context2']","Ten chinh thuc..."
```

### Evaluation Configuration

**Judge LLM:** Ollama `llama3.1:8b` (local, json mode, num_ctx=8192)
**Embeddings:** Ollama `nomic-embed-text` (local)
**Workers:** 1 (sequential to avoid Ollama OOM)
**Timeout:** 420s per job, 8 retries

**Context trimming (applied before evaluation):**
```python
MAX_CONTEXT_CHUNKS = 5      # Keep top-5 chunks only
MAX_CHUNK_CHARS = 1200      # Hard cap per chunk
```

### Metrics Used
```python
from ragas.metrics import (   # ragas.metrics (NOT .collections — incompatible with ChatOllama)
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
```

### RunConfig
```python
RunConfig(
    max_workers=1,
    max_wait=420,
    max_retries=8,
)
```

### Output
**File:** `ragas_results_local.csv`

### Metric Definitions

| Metric | Range | Meaning | Target |
|--------|-------|---------|--------|
| **Faithfulness** | 0-1 | Answer grounded in retrieved contexts | >0.7 |
| **Answer Relevancy** | 0-1 | Answer addresses the question | >0.8 |
| **Context Precision** | 0-1 | Relevant chunks ranked highest | >0.6 |
| **Context Recall** | 0-1 | Ground truth covered by contexts | >0.7 |

---

## Key Components

### Backend Services

#### 1. DocumentProcessingService
**File:** `backend/api/services.py`
- Manages async threading for document processing
- Orchestrates: PDF optimization -> AI extraction -> storage -> auto RAG ingestion
- Model selection: `gemini`, `mistral`, `mistral-ocr`
- Auto-retries extraction with original PDF if optimized PDF fails

#### 2. GeminiOCRService
**File:** `backend/api/services.py`
- Uses `google.genai` new SDK (`genai.Client`)
- Model: `gemini-2.5-flash-lite`
- Uploads PDF via Files API, waits for processing, generates JSON with bounding boxes
- Returns `{value, page, bbox}` structured fields

#### 3. RAGService
**File:** `backend/api/services.py`
- Primary extraction: Mistral OCR (always-on)
- Fallback extraction: Gemini batch OCR + PyMuPDF
- Chunking: MarkdownHeader + Recursive (800 chars, 100 overlap)
- Embedding: Mistral `mistral-embed-2312` (1024 dims, batch=50)
- Retrieval: Hybrid search (Vector + Keyword via RRF)
- Reranking: FlashRank cross-encoder (optional)
- Chat: Configurable provider (Ollama/Gemini/Mistral)

#### 4. MistralOCRService
**File:** `backend/api/services.py`
- `mistral-ocr-latest` for raw markdown extraction
- Returns `=== PAGE N ===` markers
- 4 attempts with exponential backoff + jitter

#### 5. MistralOCRSmallService
**File:** `backend/api/services.py`
- `mistral-ocr-latest` for raw OCR
- `mistral-small-latest` for JSON extraction
- Used when `ocr_model='mistral-ocr'`

### Frontend Components

#### 1. Dashboard
**File:** `frontend/src/components/Dashboard.jsx`
- Document list, upload interface, search and filtering

#### 2. ChatPanel
**File:** `frontend/src/components/ChatPanel.jsx`
- RAG chat interface with markdown rendering and persistent history

#### 3. FileUpload
**File:** `frontend/src/components/FileUpload.jsx`
- Drag-and-drop upload, model selection, progress tracking

### Database Models

#### Document
Primary model — file paths, processing status, extracted JSON, chat history, RAG status (queued/running/completed/failed with 0-100 progress %)

#### ExtractedFundData
Normalized fund fields as DB columns — 50+ fields, foreign key to Document

#### DocumentChunk
Vector embeddings for RAG:
- `content` — original chunk text
- `content_ascii` — `unidecode()` version for keyword search normalization
- `embedding` — 1024-dim vector (Mistral embed-2312)
- `search_vector` — pre-computed `tsvector` from `content_ascii`, `config='simple'`
- HNSW index (cosine) + GIN index (keyword)

#### DocumentChangeLog
Field-level audit trail with before/after values

---

## Data Flow Diagrams

### Document Upload Flow
```
+----------+
|  User    |
| Uploads  |
|   PDF    |
+----+-----+
     |
     v
+----------------+
|  FileUpload    |
|  Component     |
+----+-----------+
     | POST /api/documents/
     v
+--------------------+
|  DocumentViewSet   |
|   .create()        |
+----+---------------+
     |
     +-> Save to media/documents/
     +-> Create Document record (status='pending')
     +-> Start background thread
          |
          v
     +------------------------+
     | DocumentProcessing     |
     | Service                |
     | ._process_document_task|
     +----+-------------------+
          |
          +-> create_optimized_pdf()
          |    +-> RapidOCR + keyword filtering
          |    +-> Returns (path, page_map)
          |
          +-> extract_structured_data()
          |    +-> Gemini 2.5 Flash Lite (default)
          |    |    +-> google.genai Files API -> JSON + bbox
          |    +-> Mistral Large (alternative)
          |    +-> Mistral OCR + Small (alternative)
          |
          +-> Remap page numbers via page_map
          |
          +-> Save to Document.extracted_data + ExtractedFundData
          |    +-> Status = 'completed'
          |
          +-> Auto-trigger RAG ingestion (if enabled)
               +-> RAGService().ingest_document(doc_id) in new thread
```

### RAG Ingestion Flow
```
+----------------+
|  Trigger:      |
|  - Auto POST   |
|  - Manual API  |
+----+-----------+
     |
     v
+------------------------+
|  RAGService            |
|  .ingest_document()    |
+----+-------------------+
     |
     +-> 1. Extract full text
     |       +-> Mistral OCR (primary, always-on)
     |       |    +-> mistral-ocr-latest -> === PAGE N === markdown
     |       +-> Fallback: PyMuPDF + Gemini batch OCR
     |
     +-> 2. Clean text (_clean_text_for_rag)
     |       +-> Remove headers, fix loops
     |
     +-> 3. Chunk
     |       +-> Split by === PAGE N ===
     |       +-> MarkdownHeaderTextSplitter (H1/H2/H3)
     |       +-> RecursiveCharacterTextSplitter (800, overlap 100)
     |
     +-> 4. Embed (batches of 50)
     |       +-> Mistral mistral-embed-2312 (1024 dims)
     |       +-> Per chunk: content_ascii = unidecode(final_content)
     |
     +-> 5. Bulk insert DocumentChunk (every 200)
     |       +-> content, content_ascii, embedding, page_number
     |
     +-> 6. Populate search_vector (single bulk UPDATE)
             +-> SearchVector('content_ascii', config='simple')
```

### RAG Query Flow
```
+----------+
|  User    |
|  Query   |
+----+-----+
     | "Phi quan ly?"
     v
+----------------+
|  ChatPanel     |
+----+-----------+
     | POST /api/documents/{id}/chat/
     v
+--------------------+
|  RAGService.chat() |
+----+---------------+
     |
     +-> 1. Get structured data
     |       +-> ExtractedFundData (fees, names, dates, etc.)
     |
     +-> 2. Hybrid Search
     |       +-> Semantic: Mistral embed -> CosineDistance < 0.85 -> top 30
     |       +-> Keyword:  remove_diacritics(query) -> search_vector -> top 50
     |       +-> RRF merge -> top RAG_RETRIEVAL_CANDIDATES_K (default 15)
     |
     +-> 3. Rerank (FlashRank ms-marco-MiniLM-L-12-v2)
     |       +-> Top RAG_RERANK_TOP_K (default 5) chunks
     |
     +-> 4. Build prompt
     |       +-> NGUON 1: structured_info (fees, names, dates)
     |       +-> NGUON 2: reranked chunks with PAGE citations
     |
     +-> 5. Generate response
             +-> Ollama qwen2.5:7b (default)
             +-> Gemini 2.5 Flash Lite (optional)
             +-> Mistral small (optional)
```

---

## Configuration Summary

### Chunking Parameters
```python
CHUNK_SIZE = 800           # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
```

### Embedding Parameters
```python
EMBEDDING_MODEL = "mistral-embed-2312"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 50            # Chunks per API call
DB_WRITE_INTERVAL = 200    # Chunks per DB write
```

### Retrieval Parameters
```python
SEMANTIC_CANDIDATES = 30         # Top semantic results (distance < 0.85)
KEYWORD_CANDIDATES  = 50         # Top keyword results
RETRIEVAL_CANDIDATES_K = 15      # Candidates sent to reranker (RAG_RETRIEVAL_CANDIDATES_K)
RERANK_TOP_K = 5                 # Final chunks after reranking (RAG_RERANK_TOP_K)
SIMILARITY_METRIC = "cosine"
K_FUSION = 60                    # RRF constant
```

### Vector Index Parameters
```python
INDEX_TYPE = "HNSW"
M = 16                     # Max connections per layer
EF_CONSTRUCTION = 64       # Dynamic candidate list size
OPCLASS = "vector_cosine_ops"
```

### RAGAS Evaluation Parameters
```python
JUDGE_MODEL = "llama3.1:8b"        # Ollama local model
EMBED_MODEL = "nomic-embed-text"   # Ollama local embeddings
MAX_CONTEXT_CHUNKS = 5             # Chunks per sample sent to judge
MAX_CHUNK_CHARS = 1200             # Hard cap per chunk
MAX_WORKERS = 1
MAX_WAIT = 420                     # seconds per job
MAX_RETRIES = 8
```

---

## Environment Variables

```env
# Google AI (new SDK: google-genai)
GEMINI_API_KEY=...

# Mistral AI (required — used for both OCR and RAG embeddings)
MISTRAL_API_KEY=...

# RAG chat provider: ollama | gemini | mistral
RAG_CHAT_PROVIDER=ollama

# Ollama (when RAG_CHAT_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Mistral chat model (when RAG_CHAT_PROVIDER=mistral)
MISTRAL_CHAT_MODEL=mistral-small-latest

# FlashRank reranker
RAG_ENABLE_RERANK=true
FLASHRANK_MODEL=ms-marco-MiniLM-L-12-v2
RAG_RERANK_TOP_K=5
RAG_RETRIEVAL_CANDIDATES_K=15

# Auto RAG ingestion on upload
AUTO_RAG_INGEST_ON_UPLOAD=true

# Database
DATABASE_URL=postgresql://...

# Storage
MEDIA_ROOT=backend/media/
```

---

## API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/documents/` | POST | Upload document |
| `/api/documents/{id}/` | GET | Get document details |
| `/api/documents/{id}/ingest_for_rag/` | POST | Trigger RAG ingestion |
| `/api/documents/{id}/rag_status/` | GET | Check RAG status (0-100%) |
| `/api/documents/{id}/chat/` | POST | Chat with document (RAG) |
| `/api/documents/{id}/extracted-data/` | GET | Get structured data |

---

## Known Issues & Troubleshooting

### Low RAGAS Scores
**Root Causes:**
1. OCR quality noise -> repetitive or garbage text indexed
2. Low retrieval precision before reranking
3. Hybrid search mismatch if `search_vector` and query normalization differ

**Fixes applied:**
- `content_ascii` stored per chunk at ingestion time (via `unidecode`)
- `search_vector` built from `content_ascii` using `SearchVector('content_ascii', config='simple')` bulk update — ensures index/query use identical normalization
- FlashRank reranker reduces final context to top-5 most relevant chunks
- Context trimmed to max 5 chunks x 1200 chars for evaluation

### google-generativeai Conflict
The deprecated `google-generativeai` package must **not** be installed alongside `google-genai`. It triggers a protobuf descriptor crash via `instructor`'s conditional import chain. `requirements.txt` now uses `google-genai` only.

---

**Document Version:** 2.0
**System Version:** Based on codebase as of February 19, 2026
