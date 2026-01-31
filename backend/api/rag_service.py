"""
RAG (Retrieval-Augmented Generation) Service for Chat with PDF.
Handles:
1. Extracting full text/markdown from documents.
2. Chunking text and generating embeddings.
3. Storing vectors in PostgreSQL.
4. Retrieving relevant chunks and answering user queries.
"""
import os
import json
import logging
import io
import time
import google.generativeai as genai
from django.conf import settings
from django.utils import timezone
from django.db import close_old_connections
import fitz  # PyMuPDF
import PIL.Image
from pgvector.django import CosineDistance
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .models import Document, ExtractedFundData, DocumentChunk

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for Retrieval-Augmented Generation (Chat with PDF).
    """

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')

    def ingest_document(self, document_id: int) -> bool:
        """
        Process a document into vector chunks for RAG.
        Returns True if successful.
        """
        try:
            document = Document.objects.get(id=document_id)
            logger.info(f"Starting RAG ingestion for Doc {document_id}")

            # Mark as running (best-effort)
            try:
                Document.objects.filter(id=document_id).update(
                    rag_status='running',
                    rag_progress=0,
                    rag_error_message=None,
                    rag_started_at=timezone.now(),
                    rag_completed_at=None,
                )
            except Exception:
                pass

            # 1. Check if already ingested to avoid duplicates
            if document.chunks.exists():
                logger.info(f"Document {document_id} already ingested. Deleting old chunks...")
                document.chunks.all().delete()

            try:
                Document.objects.filter(id=document_id).update(rag_progress=5)
            except Exception:
                pass

            # 2. Extract Raw Content (optimized for Scanned PDFs)
            full_text = self._extract_content_for_rag(document)
            
            if not full_text:
                raise ValueError("Could not extract text content from document")

            try:
                Document.objects.filter(id=document_id).update(rag_progress=15)
            except Exception:
                pass

            # 3. Chunking Strategy
            page_sections = []
            current_page = 1
            current_text = ""
            
            for line in full_text.split('\n'):
                # Check for page markers in either format
                is_page_marker = False
                if ('--- PAGE ' in line and ' ---' in line) or ('=== PAGE ' in line and ' ===' in line):
                    is_page_marker = True
                    # Save previous page section if exists
                    if current_text.strip():
                        page_sections.append((current_page, current_text))
                    # Extract new page number
                    try:
                        # Remove both marker formats
                        page_str = line.strip().replace('--- PAGE ', '').replace(' ---', '')
                        page_str = page_str.replace('=== PAGE ', '').replace(' ===', '')
                        current_page = int(page_str)
                        current_text = ""
                    except ValueError:
                        is_page_marker = False
                
                if not is_page_marker:
                    current_text += line + '\n'
            
            # Add last section
            if current_text.strip():
                page_sections.append((current_page, current_text))
            
            logger.info(f"Parsed {len(page_sections)} page sections")
            
            # Split by Markdown headers to keep logical sections together
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            
            # Process each page section separately to maintain page tracking
            all_chunks_with_pages = []
            for page_num, page_text in page_sections:
                # Split by headers
                docs = markdown_splitter.split_text(page_text)
                
                # Then split into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=500,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                split_docs = text_splitter.split_documents(docs)
                
                # Tag each chunk with its page number
                for doc in split_docs:
                    doc.metadata['page_number'] = page_num
                    all_chunks_with_pages.append(doc)
            
            logger.info(f"Created {len(all_chunks_with_pages)} chunks from document.")

            try:
                Document.objects.filter(id=document_id).update(rag_progress=30)
            except Exception:
                pass

            # 4. Generate Embeddings & Save (Batch Processing)
            batch_size = 10  # Reduced batch size to avoid SSL timeouts
            chunks_to_create = []

            total_chunks = len(all_chunks_with_pages) or 1
            
            for i in range(0, len(all_chunks_with_pages), batch_size):
                batch = all_chunks_with_pages[i:i + batch_size]
                batch_texts = [d.page_content for d in batch]

                # Progress: 30% -> 95% across embedding work
                try:
                    done = min(i, total_chunks)
                    pct = 30 + int((done / total_chunks) * 65)
                    Document.objects.filter(id=document_id).update(rag_progress=min(max(pct, 30), 95))
                except Exception:
                    pass
                
                # Call Gemini Embedding API with retry logic
                max_retries = 3
                retry_count = 0
                embeddings = None
                
                while retry_count < max_retries:
                    try:
                        logger.info(f"Embedding batch {i//batch_size + 1}/{(len(all_chunks_with_pages) + batch_size - 1)//batch_size} ({len(batch)} chunks)")
                        
                        result = genai.embed_content(
                            model=self.embedding_model,
                            content=batch_texts,
                            task_type="retrieval_document"
                        )
                        embeddings = result['embedding']
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                            logger.warning(f"Embedding API error (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to embed batch after {max_retries} attempts: {str(e)}")
                            raise
                
                # Prepare DB objects
                for j, doc_chunk in enumerate(batch):
                    # Combine header metadata into content for better context
                    header_context = ""
                    if 'Header 1' in doc_chunk.metadata:
                        header_context += f"# {doc_chunk.metadata['Header 1']}\n"
                    if 'Header 2' in doc_chunk.metadata:
                        header_context += f"## {doc_chunk.metadata['Header 2']}\n"
                        
                    final_content = header_context + doc_chunk.page_content

                    # Extract page number from metadata (now properly set)
                    page_num = doc_chunk.metadata.get('page_number', 1)
                    
                    chunks_to_create.append(DocumentChunk(
                        document=document,
                        content=final_content,
                        page_number=page_num,
                        embedding=embeddings[j]
                    ))
                
                # Save chunks incrementally every batch to avoid large transactions
                if chunks_to_create:
                    try:
                        # Refresh DB connection in case it timed out during API calls
                        close_old_connections()
                        
                        DocumentChunk.objects.bulk_create(chunks_to_create)
                        logger.info(f"Saved batch {i//batch_size + 1}: {len(chunks_to_create)} chunks")
                        chunks_to_create = []  # Clear for next batch
                    except Exception as e:
                        logger.error(f"Failed to save chunk batch: {str(e)}")
                        raise
            
            # Final count
            total_chunks = document.chunks.count()
            logger.info(f"Successfully saved {total_chunks} vector chunks total.")

            try:
                Document.objects.filter(id=document_id).update(
                    rag_status='completed',
                    rag_progress=100,
                    rag_error_message=None,
                    rag_completed_at=timezone.now(),
                )
            except Exception:
                pass
            
            return True

        except Exception as e:
            logger.error(f"RAG Ingestion Error: {str(e)}")
            try:
                Document.objects.filter(id=document_id).update(
                    rag_status='failed',
                    rag_error_message=str(e),
                )
            except Exception:
                pass
            raise

    def chat(self, document_id: int, user_query: str, history: list = None) -> str:
        """
        Answer a user question using RAG.
        """
        try:
            document = Document.objects.get(id=document_id)

            def get_value(field_data):
                if isinstance(field_data, dict) and 'value' in field_data:
                    return field_data['value']
                return field_data

            def safe_text(val, max_len: int = 1200) -> str:
                if val is None:
                    return "Không có"
                if isinstance(val, (dict, list)):
                    try:
                        val = json.dumps(val, ensure_ascii=False)
                    except Exception:
                        val = str(val)
                val = str(val)
                val = val.strip()
                if not val:
                    return "Không có"
                return val if len(val) <= max_len else (val[:max_len] + " …")

            # 1. Get structured data
            structured_info = ""
            extracted_data = document.extracted_data or {}
            minimum_investment = extracted_data.get('minimum_investment')
            investment_objective = extracted_data.get('investment_objective')
            asset_allocation = extracted_data.get('asset_allocation')
            inception_date = extracted_data.get('inception_date')
            effective_date = extracted_data.get('effective_date')

            fees_extracted = extracted_data.get('fees') or {}
            operational_details = extracted_data.get('operational_details') or {}
            valuation = extracted_data.get('valuation') or {}
            risk_factors = extracted_data.get('risk_factors') or {}

            try:
                fund_data = ExtractedFundData.objects.get(document_id=document_id)
                structured_info = f"""
THÔNG TIN CƠ BẢN ĐÃ ĐƯỢC TRÍCH XUẤT (ƯU TIÊN DÙNG CHO CÂU HỎI VỀ PHÍ / TÊN / MÃ / NGÂN HÀNG):
- Tên quỹ: {fund_data.fund_name}
- Mã quỹ: {fund_data.fund_code}
- Loại quỹ (fund_type): {safe_text(fund_data.fund_type, 300)}
- Cấu trúc pháp lý (legal_structure): {safe_text(fund_data.legal_structure, 300)}
- Số giấy phép (license_number): {safe_text(fund_data.license_number, 300)}
- Cơ quan quản lý (regulator): {safe_text(fund_data.regulator, 300)}
- Công ty quản lý: {fund_data.management_company}
- Ngân hàng giám sát: {fund_data.custodian_bank}
- Người/đơn vị giám sát quỹ (fund_supervisor): {safe_text(fund_data.fund_supervisor, 300)}
- Kiểm toán (auditor): {safe_text(fund_data.auditor, 300)}

- Phí quản lý: {fund_data.management_fee}
- Phí phát hành (mua): {fund_data.subscription_fee}
- Phí mua lại (bán): {fund_data.redemption_fee}
- Phí chuyển đổi: {fund_data.switching_fee}
- Tổng chi phí (TER): {safe_text(fund_data.total_expense_ratio, 300)}
- Phí lưu ký: {safe_text(fund_data.custody_fee, 300)}
- Phí kiểm toán: {safe_text(fund_data.audit_fee, 300)}
- Phí giám sát: {safe_text(fund_data.supervisory_fee, 300)}
- Chi phí khác: {safe_text(fund_data.other_expenses, 600)}

- Ngày thành lập/quỹ bắt đầu hoạt động (inception_date): {inception_date or 'Không có'}
- Ngày hiệu lực (effective_date): {effective_date or 'Không có'}
- Mục tiêu đầu tư: {investment_objective or 'Không có'}
- Chiến lược đầu tư: {safe_text(fund_data.investment_strategy, 900)}
- Phong cách đầu tư: {safe_text(fund_data.investment_style, 200)}
- Ngành/nhóm tài sản trọng tâm: {safe_text(fund_data.sector_focus, 600)}
- Benchmark: {safe_text(fund_data.benchmark, 300)}

- Hạn chế đầu tư: {safe_text(fund_data.investment_restrictions, 900)}
- Giới hạn vay (borrowing_limit): {safe_text(fund_data.borrowing_limit, 300)}
- Giới hạn đòn bẩy (leverage_limit): {safe_text(fund_data.leverage_limit, 300)}

- Thông tin giao dịch (trading_frequency): {safe_text(fund_data.trading_frequency, 300)}
- Cut-off time: {safe_text(fund_data.cut_off_time, 300)}
- Tần suất tính NAV: {safe_text(fund_data.nav_calculation_frequency, 300)}
- Công bố NAV: {safe_text(fund_data.nav_publication, 300)}
- Chu kỳ thanh toán (settlement_cycle): {safe_text(fund_data.settlement_cycle, 300)}

- Phương pháp định giá: {safe_text(fund_data.valuation_method, 900)}
- Nguồn giá: {safe_text(fund_data.pricing_source, 900)}

- Quyền nhà đầu tư: {safe_text(fund_data.investor_rights, 900)}
- Đại lý phân phối: {safe_text(fund_data.distribution_agent, 400)}
- Kênh phân phối: {safe_text(fund_data.sales_channels, 600)}

- Rủi ro tập trung: {safe_text(fund_data.concentration_risk, 700)}
- Rủi ro thanh khoản: {safe_text(fund_data.liquidity_risk, 700)}
- Rủi ro lãi suất: {safe_text(fund_data.interest_rate_risk, 700)}

- Số tiền đầu tư tối thiểu (ban đầu / bổ sung): {json.dumps(minimum_investment or {}, ensure_ascii=False)}
- Cơ cấu phân bổ tài sản: {json.dumps(asset_allocation or {}, ensure_ascii=False)}
- Danh mục đầu tư (trích xuất): {json.dumps(fund_data.portfolio or [], ensure_ascii=False)}
""".strip()
            except ExtractedFundData.DoesNotExist:
                structured_info = f"""
THÔNG TIN CƠ BẢN ĐÃ ĐƯỢC TRÍCH XUẤT (từ Document.extracted_data):
- Ngày thành lập/quỹ bắt đầu hoạt động (inception_date): {inception_date or 'Không có'}
- Ngày hiệu lực (effective_date): {effective_date or 'Không có'}
- Mục tiêu đầu tư: {investment_objective or 'Không có'}
- Chiến lược đầu tư: {safe_text(get_value(extracted_data.get('investment_strategy')), 900)}
- Phí (fees): {safe_text(fees_extracted, 900)}
- Thông tin giao dịch (operational_details): {safe_text(operational_details, 900)}
- Định giá (valuation): {safe_text(valuation, 900)}
- Hạn chế/giới hạn đầu tư: {safe_text(get_value(extracted_data.get('investment_restrictions')) or get_value(extracted_data.get('borrowing_limit')) or get_value(extracted_data.get('leverage_limit')), 900)}
- Quyền NĐT / Phân phối: {safe_text(get_value(extracted_data.get('investor_rights')) or get_value(extracted_data.get('distribution_agent')) or get_value(extracted_data.get('sales_channels')), 900)}
- Rủi ro (risk_factors): {safe_text(risk_factors, 900)}
- Số tiền đầu tư tối thiểu (ban đầu / bổ sung): {json.dumps(minimum_investment or {}, ensure_ascii=False)}
- Cơ cấu phân bổ tài sản: {json.dumps(asset_allocation or {}, ensure_ascii=False)}
""".strip()

            # 2. Vector Search (Semantic Retrieval)
            rag_context = ""
            try:
                query_embedding = genai.embed_content(
                    model=self.embedding_model,
                    content=user_query,
                    task_type="retrieval_query"
                )['embedding']

                relevant_chunks = DocumentChunk.objects.filter(document_id=document_id) \
                    .annotate(distance=CosineDistance('embedding', query_embedding)) \
                    .order_by('distance')[:15]

                rag_context = "\n\n---\n\n".join(
                    [f"=== PAGE {c.page_number} ===\n{c.content}" for c in relevant_chunks]
                )
            except Exception as e:
                logger.warning(f"RAG retrieval failed for document {document_id}: {str(e)}")
                rag_context = ""

            # 3. Build prompt with both sources
            system_prompt = f"""
Bạn là trợ lý phân tích tài chính thông minh chuyên về Quỹ đầu tư.

HÃY SỬ DỤNG CẢ HAI NGUỒN THÔNG TIN SAU ĐỂ TRẢ LỜI:

NGUỒN 1: DỮ LIỆU CẤU TRÚC (ƯU TIÊN dùng cho câu hỏi về Phí, Tên, Mã số, Ngân hàng, Công ty quản lý)
{structured_info}

NGUỒN 2: TRÍCH ĐOẠN VĂN BẢN CHI TIẾT (dùng cho câu hỏi giải thích, chiến lược, rủi ro, điều khoản...)
{rag_context}

QUY TẮC:
1. Nếu người dùng hỏi về Phí hoặc Số liệu cụ thể, hãy kiểm tra NGUỒN 1 trước.
2. Nếu NGUỒN 1 không có / không đủ, hãy dùng NGUỒN 2.
3. Trả lời bằng tiếng Việt, ngắn gọn, chuyên nghiệp.
4. Nếu không tìm thấy thông tin từ cả hai nguồn, hãy nói: "Tôi không tìm thấy thông tin đó trong tài liệu."
""".strip()
            
            # Prepare chat history for Gemini
            chat_history = []
            if history:
                for h in history:
                    role = "user" if h.get('sender') == 'user' else "model"
                    chat_history.append({"role": role, "parts": [h.get('text', '')]})

            # Start chat session
            chat = self.chat_model.start_chat(history=chat_history)
            response = chat.send_message(f"{system_prompt}\n\nCÂU HỎI: {user_query}")
            
            return response.text

        except Exception as e:
            logger.error(f"RAG Chat Error: {str(e)}")
            return "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi."

    def _extract_content_for_rag(self, document) -> str:
        """
        Helper to get raw text for RAG with page markers.
        Processes ALL pages and handles batching correctly.
        """
        try:
            # Prefer ORIGINAL uploaded PDF for RAG extraction
            original_path = None
            try:
                original_path = document.file.path
            except Exception:
                original_path = None

            optimized_path = None
            if getattr(document, 'optimized_file', None):
                try:
                    optimized_path = document.optimized_file.path
                except Exception:
                    optimized_path = None

            chosen_path = None
            if original_path and os.path.exists(original_path):
                chosen_path = original_path
                if optimized_path:
                    logger.info("RAG: using original upload (preferred).")
                logger.info(f"Extracting RAG content from ORIGINAL PDF: {chosen_path}")
            elif optimized_path and os.path.exists(optimized_path):
                chosen_path = optimized_path
                logger.warning(f"RAG: original PDF missing, using optimized: {optimized_path}")
            else:
                raise ValueError("No PDF file found on disk for RAG extraction.")
            
            doc = fitz.open(chosen_path)
            full_text = ""
            last_error: str | None = None
            
            # Process pages in batches of 15
            batch_size = 15
            total_pages = len(doc)
            
            logger.info(f"Total pages to ingest: {total_pages}")
            
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"Processing RAG batch: Pages {batch_start + 1} to {batch_end}")

                batch_parts: list[str] = []
                
                # Reset model_inputs for each batch
                model_inputs = [
                    "ROLE: You are a strict OCR engine. Your ONLY job is to convert images to text.",
                    "TASK: Transcribe the extracted text from these images into Markdown.",
                    "OUTPUT FORMAT (CRITICAL): For EACH page, you MUST start its transcription with a single line exactly like: === PAGE N === (N is the page number provided in the text right before the image).",
                    "RULES:",
                    "1. Do NOT add any intro like 'Here is the text' or 'Okay'.",
                    "2. Do NOT summarize. Transcribe EXACTLY what is written, word-for-word.",
                    "3. PRESERVE ALL TABLES. Use Markdown table syntax for 'Biểu phí' (Fee Schedule) and 'Danh mục đầu tư'.",
                    "4. If a page contains numbers (fees, NAV, dates), extract them precisely.",
                    "5. Output Vietnamese text with correct accents."
                ]

                ocr_pages_in_batch = 0
                
                # Convert pages in this batch to images
                for i in range(batch_start, batch_end):
                    page = doc[i]

                    # Fast path: extract selectable text directly
                    try:
                        direct_text = page.get_text("text")
                    except Exception:
                        direct_text = ""

                    if direct_text and direct_text.strip() and len(direct_text.strip()) >= 50:
                        batch_parts.append(f"=== PAGE {i + 1} ===\n{direct_text.strip()}")
                        continue

                    # Fallback: OCR via Gemini on rendered image
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
                        img_data = pix.tobytes("png")
                        image = PIL.Image.open(io.BytesIO(img_data)).convert("RGB")
                        image.load()
                        image = image.copy()
                        model_inputs.append(f"=== PAGE {i + 1} ===")
                        model_inputs.append(image)
                        ocr_pages_in_batch += 1
                    except Exception as e:
                        last_error = f"Render page {i + 1} failed: {e}"
                        logger.warning(last_error)
                
                # Call Gemini with retry mechanism
                if ocr_pages_in_batch > 0:
                    batch_text = ""
                    for attempt in range(3):
                        try:
                            response = self.chat_model.generate_content(model_inputs)
                            batch_text = response.text or ""
                            if batch_text.strip():
                                break  # success
                            last_error = f"Gemini OCR returned empty for batch {batch_start + 1}-{batch_end} (attempt {attempt + 1})"
                            logger.warning(last_error)
                        except Exception as e:
                            wait = (attempt + 1) * 5
                            last_error = f"Gemini OCR failed for batch {batch_start + 1}-{batch_end} (attempt {attempt + 1}): {e}"
                            logger.warning(f"{last_error}. Retrying in {wait}s...")
                            time.sleep(wait)

                    if batch_text.strip():
                        batch_parts.append(batch_text.strip())
                    else:
                        logger.error(f"Failed to extract OCR text for pages {batch_start + 1}-{batch_end} after 3 attempts.")

                if batch_parts:
                    full_text += "\n\n".join(batch_parts) + "\n\n"

                # Rest between batches to avoid rate limit
                time.sleep(2)

            doc.close()
            
            if not full_text.strip():
                raise ValueError(f"Extracted text is empty. Last error: {last_error or 'unknown'}")
                
            return full_text

        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return ""
