"""
Service layer for OCR using Gemini 2.0 Flash
"""
import os
import json
import logging
import re
from pathlib import Path
import threading
import unicodedata
import google.generativeai as genai
from mistralai import Mistral
from django.conf import settings
from django.utils import timezone
import tempfile
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from .models import Document, ExtractedFundData

logger = logging.getLogger(__name__)

try:
    # Prefer unidecode when available; it handles Vietnamese well and is fast.
    from unidecode import unidecode  # type: ignore
except Exception:  # pragma: no cover
    unidecode = None

def remove_vietnamese_diacritics(text: str) -> str:
    """
    Remove Vietnamese diacritics and convert to plain ASCII.
    Used for flexible matching when OCR strips diacritics.
    
    Examples:
        'tên quỹ' → 'ten quy'
        'công ty quản lý' → 'cong ty quan ly'
    """
    # Normalize to NFD (decompose accented chars)
    text = unicodedata.normalize('NFD', text)
    # Remove combining marks (diacritics)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # Handle special Vietnamese characters that don't decompose
    replacements = {
        'đ': 'd', 'Đ': 'D',
        'ð': 'd',  # Alternative encoding
    }
    for viet_char, ascii_char in replacements.items():
        text = text.replace(viet_char, ascii_char)
    return text.lower()

def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for keyword matching:
    1. Remove Vietnamese diacritics (OCR often strips them)
    2. Remove spaces and special characters
    3. Convert to lowercase
    
    This handles cases where OCR outputs 'noi dung ban cao bach'
    and we want to match 'nội dung bản cáo bạch'.
    """
    if not text:
        return ""

    # Preferred path: unidecode() tends to match OCR output best.
    if unidecode is not None:
        text = unidecode(text).lower()
        return re.sub(r"[^a-z0-9]", "", text)

    # Fallback: unicode decomposition + strip combining marks.
    text = remove_vietnamese_diacritics(text)
    return "".join(c for c in text if c.isalnum())

try:
    ocr_engine = RapidOCR(lang_list=['en', 'vi'], gpu_id=0 if getattr(settings, 'USE_GPU', False) else -1)
    logger.info("RapidOCR engine initialized")
except Exception as e:
    logger.warning(f"Could not initialize RapidOCR with GPU, falling back to CPU: {e}")
    ocr_engine = RapidOCR(lang_list=['en', 'vi'], gpu_id=-1)

def create_optimized_pdf(original_pdf_path: str) -> str:
    """
    Hỗ trợ cả PDF dạng Text và PDF dạng Scanned Image.
    Sử dụng RapidOCR để 'đọc lướt' tìm keyword trên các trang ảnh.
    
    Args:
        original_pdf_path: Path to the original PDF file
        
    Returns:
        Path to the optimized PDF file or original if too short
    """
    try:
        doc = fitz.open(original_pdf_path)
        total_pages = len(doc)
        
        # Nếu file ngắn, lấy hết luôn cho nhanh
        if total_pages <= 5:
            logger.info(f"PDF has only {total_pages} pages, returning original")
            return original_pdf_path

        # --- TỪ KHÓA QUAN TRỌNG ---
        # Lưu dạng có dấu (dễ đọc), nhưng so khớp sẽ dùng normalize_text_for_matching()
        # để chịu được OCR mất dấu / sai khoảng trắng.
        # NOTE: Avoid overly-generic keywords that appear in headers/footers on *every* page
        # (e.g. “công ty quản lý”, “ngân hàng giám sát”), otherwise we keep almost the whole PDF.
        keywords = {
            "identity": [
                "tên quỹ",
                "mã giao dịch",
                "mã chứng khoán",
                "mã quỹ",
                "giấy phép",
                "giấy phép thành lập",
            ],
            "fees": [
                "biểu phí",
                "các loại phí",
                "phí phát hành",
                "phí quản lý",
                "phí mua lại",
                "phí chuyển đổi",
                "chi phí của quỹ",
                "phí mua",
                "phí bán",
                "phí đăng ký",
            ],
            "tables": [
                "danh mục đầu tư",
                "cơ cấu tài sản",
                "tài sản ròng",
                "giá trị tài sản ròng",
                "nav",
                "biến động nav",
                "lịch sử chia cổ tức",
                "phân phối lợi nhuận",
                "hoạt động đầu tư",
            ],
        }
        
        # Normalize keywords for matching (remove diacritics + spaces)
        normalized_keywords = {
            category: [normalize_text_for_matching(k) for k in kws]
            for category, kws in keywords.items()
        }
        
        # Luôn lấy 4 trang đầu (trang bìa, mục lục, thông tin chung)
        selected_pages = {0, 1, 2, 3}

        # Thường 3 trang cuối có chữ ký / bảng tóm tắt, giữ lại để tránh bỏ sót.
        if total_pages > 10:
            selected_pages.update({total_pages - 1, total_pages - 2, total_pages - 3})
        
        logger.info(f"Scanning {total_pages} pages (Hybrid Mode: Text + OCR, DPI=60)...")
        pages_with_ocr = 0

        # Practical guardrail: keep the optimized PDF small enough for downstream AI.
        # (Scanned PDFs are huge; if we keep too many pages, Gemini often fails.)
        max_selected_pages = getattr(settings, "MAX_OPTIMIZED_PDF_PAGES", 60)

        # Identity fields are usually near the beginning; restricting this reduces header/footer matches.
        max_identity_page = getattr(settings, "MAX_IDENTITY_SCAN_PAGES", 40)
        
        # Quét từ trang 4 trở đi
        # Quét từ trang 4 trở đi, nhưng chừa 3 trang cuối vì đã auto-keep.
        scan_end = total_pages
        if total_pages > 10:
            scan_end = max(4, total_pages - 3)

        for page_num in range(4, scan_end):
            page = doc.load_page(page_num)
            
            # BƯỚC 1: Thử lấy text thông thường (nhanh nhất)
            text = page.get_text().lower()
            used_ocr = False
            
            # BƯỚC 2: Nếu text quá ít (dưới 50 ký tự) -> Khả năng cao là Scanned PDF
            if len(text) < 50:
                try:
                    # Chuyển trang PDF thành ảnh (Pixmap) để OCR
                    # dpi=150 đủ để tìm keyword, giảm thời gian xử lý đáng kể.
                    pix = page.get_pixmap(dpi=150)
                    
                    # Chuyển đổi định dạng ảnh cho RapidOCR
                    img_bytes = pix.tobytes("png")
                    
                    # Chạy OCR (trả về list kết quả, mỗi kết quả có text và toạ độ)
                    result = ocr_engine(img_bytes)
                    
                    if result and isinstance(result, tuple):
                        result = result[0]
                    
                    if result:
                        # Gộp các đoạn text lại thành 1 chuỗi để tìm keyword
                        text = " ".join([res[1] for res in result]).lower()
                        used_ocr = True
                        pages_with_ocr += 1
                        
                        # Log mỗi 10 trang để theo dõi tiến độ
                        if page_num % 10 == 0:
                            logger.debug(f"Page {page_num}: OCR extracted {len(text)} characters")
                except Exception as ocr_error:
                    logger.debug(f"OCR failed on page {page_num}: {ocr_error}")
                    text = ""

            # Skip trang quá ít chữ (trang trắng / hình minh hoạ)
            if len(normalize_text_for_matching(text)) < 20:
                continue
            
            # BƯỚC 3: Kiểm tra Keyword trên đoạn text (dù là gốc hay OCR ra)
            # Normalize text for comparison (handles OCR without diacritics)
            normalized_text = normalize_text_for_matching(text)
            
            is_relevant = False
            for category, normalized_keys in normalized_keywords.items():
                # Avoid selecting tons of pages just because identity keywords appear in headers.
                if category == "identity" and page_num > max_identity_page:
                    continue

                if any(k in normalized_text for k in normalized_keys):
                    is_relevant = True
                    logger.debug(f"Page {page_num}: Matched category '{category}'")
                    # Logic lấy thêm trang sau nếu là bảng biểu
                    if category == "tables" and page_num + 1 < total_pages:
                        selected_pages.add(page_num + 1)
                    break
            
            if is_relevant:
                selected_pages.add(page_num)

            # Stop early if we already collected enough pages.
            if len(selected_pages) >= max_selected_pages:
                logger.info(
                    f"Reached max_selected_pages={max_selected_pages}; stopping scan early at page {page_num}."
                )
                break

        # Kết thúc quét
        sorted_pages = sorted(list(selected_pages))
        logger.info(f"Selected {len(sorted_pages)}/{total_pages} pages via OCR-scan. Used OCR on {pages_with_ocr} pages.")

        # Tạo file PDF mới
        doc.select(sorted_pages)
        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        # Save with cleanup/compression to reduce size (important for scanned PDFs)
        doc.save(temp_path, garbage=4, deflate=True)
        doc.close()
        
        return temp_path

    except Exception as e:
        logger.error(f"Error optimizing PDF: {str(e)}")
        return original_pdf_path

class GeminiOCRService:
    """Service for OCR using Gemini 2.0 Flash API"""
    
    def __init__(self):
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def extract_structured_data(self, pdf_path: str) -> dict:
        """
        Extract financial data from PDF using Gemini 2.0 Flash OCR.
        Always uploads the PDF directly to Gemini (no image conversion).
        Returns a dictionary of extracted data.
        """
        try:
            file_size = os.path.getsize(pdf_path)
            logger.info(f"Uploading PDF to Gemini: {pdf_path} (Size: {file_size} bytes)")
            
            uploaded_file = genai.upload_file(pdf_path, mime_type="application/pdf")
            logger.info(f"Uploaded file URI: {uploaded_file.uri}")

            # Wait for processing to finish before generating content
            import time
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise ValueError("File processing failed on Gemini API")

            prompt = self._get_extraction_prompt()
            
            try:
                # Use temperature=0 for deterministic, consistent results
                generation_config = genai.types.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
                response = self.model.generate_content(
                    [uploaded_file, prompt],
                    generation_config=generation_config
                )
            except Exception as gen_error:
                logger.error(f"Gemini generate_content failed: {gen_error}")
                # If it's a 400 error, it might be due to the optimized PDF being weird.
                # But we can't easily retry with original here without passing it in.
                raise gen_error

            # Cleanup uploaded file
            try:
                genai.delete_file(uploaded_file.name)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup uploaded file: {cleanup_error}")

            json_text = self._clean_response(response.text)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {json_text[:200]}...")
                # Return a partial dict or empty dict to avoid crashing
                return {"error": "Failed to parse JSON", "raw_text": json_text}

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def _get_extraction_schema(self) -> dict:
        """Define the expected JSON schema for extraction"""
        return {
            "type": "object",
            "properties": {
                "fund_name": {
                    "type": "string",
                    "description": "Full name of the investment fund"
                },
                "fund_code": {
                    "type": "string",
                    "description": "Unique identifier/code for the fund"
                },
                "management_company": {
                    "type": "string",
                    "description": "Name of the fund management company"
                },
                "custodian_bank": {
                    "type": "string",
                    "description": "Name of the custodian bank"
                },
                "inception_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Fund inception/establishment date (YYYY-MM-DD)"
                },
                "investment_objective": {
                    "type": "string",
                    "description": "Main investment objective of the fund"
                },
                "fees": {
                    "type": "object",
                    "properties": {
                        "management_fee": {
                            "type": "string",
                            "description": "Annual management fee (e.g. '2%', '1.5%/năm')"
                        },
                        "subscription_fee": {
                            "type": "string",
                            "description": "Fee for buying/subscribing units (e.g. '1%', 'Miễn phí'). Synonyms: Phí phát hành, Phí mua, Phí đăng ký mua."
                        },
                        "redemption_fee": {
                            "type": "string",
                            "description": "Fee for selling/redeeming units (e.g. '0.5%', 'Theo thời gian nắm giữ'). Synonyms: Phí mua lại, Phí bán, Phí rút vốn."
                        },
                        "switching_fee": {
                            "type": "string",
                            "description": "Fee for switching between funds"
                        }
                    }
                },
                "minimum_investment": {
                    "type": "object",
                    "properties": {
                        "initial": {
                            "type": "number",
                            "description": "Minimum initial investment amount"
                        },
                        "additional": {
                            "type": "number",
                            "description": "Minimum additional investment amount"
                        },
                        "currency": {
                            "type": "string",
                            "description": "Currency code (e.g., VND, USD)"
                        }
                    }
                },
                "portfolio": {
                    "type": "array",
                    "description": "Top portfolio holdings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "security_name": {"type": "string"},
                            "security_code": {"type": "string"},
                            "asset_type": {"type": "string"},
                            "market_value": {"type": "number"},
                            "percentage": {"type": "number"}
                        }
                    }
                },
                "asset_allocation": {
                    "type": "object",
                    "description": "Asset allocation percentages",
                    "properties": {
                        "stocks": {"type": "number"},
                        "bonds": {"type": "number"},
                        "cash": {"type": "number"},
                        "other": {"type": "number"}
                    }
                },
                "nav_history": {
                    "type": "array",
                    "description": "Recent NAV (Net Asset Value) history - Lịch sử giá trị tài sản ròng (NAV) qua các kỳ",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "format": "date", "description": "Date in YYYY-MM-DD format"},
                            "nav_per_unit": {"type": "number", "description": "NAV per unit value"}
                        }
                    }
                },
                "dividend_history": {
                    "type": "array",
                    "description": "Dividend distribution history - Lịch sử chia cổ tức",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "format": "date"},
                            "dividend_per_unit": {"type": "number"},
                            "payment_date": {"type": "string", "format": "date"}
                        }
                    }
                },
                "performance": {
                    "type": "object",
                    "description": "Fund performance metrics",
                    "properties": {
                        "ytd": {"type": "number", "description": "Year-to-date return %"},
                        "1_year": {"type": "number"},
                        "3_year": {"type": "number"},
                        "5_year": {"type": "number"},
                        "since_inception": {"type": "number"}
                    }
                },
                "risk_profile": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive"],
                    "description": "Risk profile of the fund"
                }
            },
            "required": ["fund_name", "fund_code"]
        }
    
    def _get_extraction_prompt(self) -> str:
        """
        Generate detailed extraction prompt with schema for GeminiOCRService
        """
        schema = self._get_extraction_schema()
        
        return f"""You are an expert financial document analyst specializing in Vietnamese investment fund prospectuses.

Extract ALL relevant financial information from this prospectus document and structure it according to the following JSON schema:

{json.dumps(schema, indent=2)}

IMPORTANT INSTRUCTIONS:
1. Extract data EXACTLY as it appears in the document
2. For Vietnamese text, preserve diacritical marks (ả, ế, ô, etc.)
3. For fees, extract the full text value including percentage sign if present (e.g., "2.5%", "Miễn phí").
4. For currency amounts, extract as numbers without formatting (e.g., "1,000,000 VND" → 1000000)
5. For dates, use ISO format YYYY-MM-DD
6. If a field is not found, use null (not empty string)
7. For portfolio holdings, extract the top 10-15 holdings if available. Look for tables labeled "Danh mục đầu tư", "Cơ cấu tài sản", "Top 10 cổ phiếu", or similar.
8. For NAV history, extract the most recent 10-20 entries if available. Look for tables labeled "Giá trị tài sản ròng", "NAV", "Biến động NAV".
9. For Dividend history, look for "Lịch sử chia cổ tức", "Phân phối lợi nhuận".
10. Pay special attention to:
   - Fee structures (management, subscription, redemption fees)
   - Minimum investment requirements
   - Asset allocation percentages
   - Performance data
   - Risk classifications

### CRITICAL INSTRUCTIONS FOR FEE EXTRACTION:
1. **Search the ENTIRE document** for fee information, typically found in:
   - Section titled "Biểu phí" or "Các loại phí"
   - "Phí và chi phí" section
   - Tables showing different fee types
2. **Common fee terminology (use these to find fees)**:
   - Management Fee (Phí quản lý): Usually "X%/năm" or "Tối đa X%/năm"
   - Subscription/Issue Fee (Phí phát hành/Phí mua/Phí đăng ký): Often shown as "%" or "Miễn phí"
   - Redemption/Exit Fee (Phí mua lại/Phí bán/Phí rút vốn): May vary by holding period
   - Switching Fee (Phí chuyển đổi): Between different fund classes
3. **Extract ALL fee values found**, even if they are:
   - Ranges ("0.5% - 1%")
   - Conditional ("Theo thời gian nắm giữ")
   - Text descriptions ("Miễn phí", "Không thu")
   - Formulas ("Tối đa 1.5%/năm")
4. **Do NOT skip fees** - if you see any fee mentioned, extract it

### CRITICAL INSTRUCTIONS FOR TABLES (Danh mục đầu tư, NAV, Cổ tức):
1. **Detect Table Boundaries**: Financial tables (Portfolio, NAV) often span multiple pages WITHOUT repeating headers. Treat consecutive pages with tabular data as a single continuous table.
2. **"Borderless" Tables**: Many tables do not have grid lines. Use visual alignment (columns) to associate values.
3. **Portfolio (Danh mục đầu tư)**:
   - Extract ALL items available, do not limit to top 10 unless the document explicitly says "Top 10".
   - Look for columns like "Mã CK" (Code), "Tên CK" (Name), "Tỷ trọng" (Weight/%), "Giá trị" (Value).
   - If "Mã CK" is missing but "Tên CK" exists, map "Tên CK" to security_name.
4. **NAV History (Giá trị tài sản ròng)**:
   - Extract the table usually labeled "Biến động giá trị tài sản ròng" or "Lịch sử NAV".
   - Keys: Date (Ngày), NAV/Unit (Giá trị một ccq).
5. **Dividend History (Lịch sử chia cổ tức)**:
   - Look for keywords: "Chi trả cổ tức", "Phân phối lợi nhuận".
   - If the fund says "No dividend distributed" (Không chia cổ tức), return an empty array [].
Return ONLY valid JSON matching the schema above. Do not include any explanatory text before or after the JSON.

Example of expected output format:
{{
  "fund_name": "Quỹ Đầu Tư Cân Bằng VCBF",
  "fund_code": "VCBF-TBF",
  "management_company": "VCBF",
  "custodian_bank": "Vietcombank",
  "fees": {{
    "management_fee": "Tối đa 1.5%/năm",
    "subscription_fee": "1.5%",
    "redemption_fee": "Miễn phí"
  }},
  "portfolio": [
    {{
      "security_name": "Vingroup JSC",
      "security_code": "VIC",
      "asset_type": "Stock",
      "percentage": 8.5
    }}
  ]
}}

Now extract the data from the provided document."""

    def _clean_response(self, text: str) -> str:
        """Clean markdown formatting from response"""
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()

class MistralOCRService:
    """Service for OCR using Mistral AI"""
    
    def __init__(self):
        # Configure Mistral API
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"
    
    def extract_structured_data(self, pdf_path: str) -> dict:
        """
        Extract financial data from PDF using Mistral AI.
        Since Mistral API doesn't accept PDF uploads directly like Gemini,
        we extract text using PyMuPDF (fitz) first, then send to LLM.
        """
        try:
            logger.info(f"Extracting text from PDF for Mistral: {pdf_path}")
            
            # 1. Extract text from the optimized PDF
            doc = fitz.open(pdf_path)
            full_text = ""
            
            # Try standard text extraction first
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                
                # If page has very little text, it's likely a scanned image - use OCR
                if len(page_text.strip()) < 50:
                    try:
                        logger.debug(f"Page {page_num} has little text, using OCR...")
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        result = ocr_engine(img_bytes)
                        
                        if result and isinstance(result, tuple):
                            result = result[0]
                        
                        if result:
                            page_text = " ".join([res[1] for res in result])
                            logger.debug(f"OCR extracted {len(page_text)} characters from page {page_num}")
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed on page {page_num}: {ocr_error}")
                        page_text = ""
                
                full_text += page_text + "\n---PAGE BREAK---\n"
            
            doc.close()

            # If text is still empty after OCR attempts, return error
            if len(full_text.strip()) < 100:
                logger.error("Extracted text is too short even after OCR. Cannot process document.")
                return {
                    "error": "Document appears to be empty or unreadable. Please ensure the PDF contains text or clear scanned images.",
                    "portfolio": [],
                    "nav_history": [],
                    "dividend_history": []
                }

            prompt = self._get_extraction_prompt()
            
            # 2. Send to Mistral
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data extraction assistant. You output only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nDOCUMENT CONTENT:\n{full_text}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0
            )

            response_content = chat_response.choices[0].message.content
            
            # 3. Parse Response
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {response_content[:200]}...")
                return {"error": "Failed to parse JSON", "raw_text": response_content}

        except Exception as e:
            logger.error(f"Error extracting text with Mistral: {str(e)}")
            raise
    
    def _get_extraction_schema(self) -> dict:
        """Define the expected JSON schema for extraction"""
        return {
            "type": "object",
            "properties": {
                "fund_name": {
                    "type": "string",
                    "description": "Full name of the investment fund"
                },
                "fund_code": {
                    "type": "string",
                    "description": "Unique identifier/code for the fund"
                },
                "management_company": {
                    "type": "string",
                    "description": "Name of the fund management company"
                },
                "custodian_bank": {
                    "type": "string",
                    "description": "Name of the custodian bank"
                },
                "inception_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Fund inception/establishment date (YYYY-MM-DD)"
                },
                "investment_objective": {
                    "type": "string",
                    "description": "Main investment objective of the fund"
                },
                "fees": {
                    "type": "object",
                    "properties": {
                        "management_fee": {
                            "type": "string",
                            "description": "Annual management fee (e.g. '2%', '1.5%/năm')"
                        },
                        "subscription_fee": {
                            "type": "string",
                            "description": "Fee for buying/subscribing units (e.g. '1%', 'Miễn phí'). Synonyms: Phí phát hành, Phí mua, Phí đăng ký mua."
                        },
                        "redemption_fee": {
                            "type": "string",
                            "description": "Fee for selling/redeeming units (e.g. '0.5%', 'Theo thời gian nắm giữ'). Synonyms: Phí mua lại, Phí bán, Phí rút vốn."
                        },
                        "switching_fee": {
                            "type": "string",
                            "description": "Fee for switching between funds"
                        }
                    }
                },
                "minimum_investment": {
                    "type": "object",
                    "properties": {
                        "initial": {
                            "type": "number",
                            "description": "Minimum initial investment amount"
                        },
                        "additional": {
                            "type": "number",
                            "description": "Minimum additional investment amount"
                        },
                        "currency": {
                            "type": "string",
                            "description": "Currency code (e.g., VND, USD)"
                        }
                    }
                },
                "portfolio": {
                    "type": "array",
                    "description": "Top portfolio holdings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "security_name": {"type": "string"},
                            "security_code": {"type": "string"},
                            "asset_type": {"type": "string"},
                            "market_value": {"type": "number"},
                            "percentage": {"type": "number"}
                        }
                    }
                },
                "asset_allocation": {
                    "type": "object",
                    "description": "Asset allocation percentages",
                    "properties": {
                        "stocks": {"type": "number"},
                        "bonds": {"type": "number"},
                        "cash": {"type": "number"},
                        "other": {"type": "number"}
                    }
                },
                "nav_history": {
                    "type": "array",
                    "description": "Recent NAV (Net Asset Value) history - Lịch sử giá trị tài sản ròng (NAV) qua các kỳ",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "format": "date", "description": "Date in YYYY-MM-DD format"},
                            "nav_per_unit": {"type": "number", "description": "NAV per unit value"}
                        }
                    }
                },
                "dividend_history": {
                    "type": "array",
                    "description": "Dividend distribution history - Lịch sử chia cổ tức",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "format": "date"},
                            "dividend_per_unit": {"type": "number"},
                            "payment_date": {"type": "string", "format": "date"}
                        }
                    }
                },
                "performance": {
                    "type": "object",
                    "description": "Fund performance metrics",
                    "properties": {
                        "ytd": {"type": "number", "description": "Year-to-date return %"},
                        "1_year": {"type": "number"},
                        "3_year": {"type": "number"},
                        "5_year": {"type": "number"},
                        "since_inception": {"type": "number"}
                    }
                },
                "risk_profile": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive"],
                    "description": "Risk profile of the fund"
                }
            },
            "required": ["fund_name", "fund_code"]
        }
    
    def _get_extraction_prompt(self) -> str:
        """
        Generate detailed extraction prompt with schema
        """
        schema = self._get_extraction_schema()
        
        return f"""You are an expert financial document analyst specializing in Vietnamese investment fund prospectuses.

Extract ALL relevant financial information from this prospectus document and structure it according to the following JSON schema:

{json.dumps(schema, indent=2)}

IMPORTANT INSTRUCTIONS:
1. Extract data EXACTLY as it appears in the document
2. For Vietnamese text, preserve diacritical marks (ả, ế, ô, etc.)
3. For fees, extract the full text value including percentage sign if present (e.g., "2.5%", "Miễn phí").
4. For currency amounts, extract as numbers without formatting (e.g., "1,000,000 VND" → 1000000)
5. For dates, use ISO format YYYY-MM-DD
6. If a field is not found, use null (not empty string)
7. For portfolio holdings, extract the top 10-15 holdings if available. Look for tables labeled "Danh mục đầu tư", "Cơ cấu tài sản", "Top 10 cổ phiếu", or similar.
8. For NAV history, extract the most recent 10-20 entries if available. Look for tables labeled "Giá trị tài sản ròng", "NAV", "Biến động NAV".
9. For Dividend history, look for "Lịch sử chia cổ tức", "Phân phối lợi nhuận".
10. Pay special attention to:
   - Fee structures (management, subscription, redemption fees)
   - Minimum investment requirements
   - Asset allocation percentages
   - Performance data
   - Risk classifications

### CRITICAL INSTRUCTIONS FOR TABLES (Danh mục đầu tư, NAV, Cổ tức):
1. **Detect Table Boundaries**: Financial tables (Portfolio, NAV) often span multiple pages WITHOUT repeating headers. Treat consecutive pages with tabular data as a single continuous table.
2. **"Borderless" Tables**: Many tables do not have grid lines. Use visual alignment (columns) to associate values.
3. **Portfolio (Danh mục đầu tư)**:
   - Extract ALL items available, do not limit to top 10 unless the document explicitly says "Top 10".
   - Look for columns like "Mã CK" (Code), "Tên CK" (Name), "Tỷ trọng" (Weight/%), "Giá trị" (Value).
   - If "Mã CK" is missing but "Tên CK" exists, map "Tên CK" to security_name.
4. **NAV History (Giá trị tài sản ròng)**:
   - Extract the table usually labeled "Biến động giá trị tài sản ròng" or "Lịch sử NAV".
   - Keys: Date (Ngày), NAV/Unit (Giá trị một ccq).
5. **Dividend History (Lịch sử chia cổ tức)**:
   - Look for keywords: "Chi trả cổ tức", "Phân phối lợi nhuận".
   - If the fund says "No dividend distributed" (Không chia cổ tức), return an empty array [].
Return ONLY valid JSON matching the schema above. Do not include any explanatory text before or after the JSON.

Example of expected output format:
{{
  "fund_name": "Quỹ Đầu Tư Cân Bằng VCBF",
  "fund_code": "VCBF-TBF",
  "management_company": "VCBF",
  "custodian_bank": "Vietcombank",
  "fees": {{
    "management_fee": 2.0,
    "subscription_fee": 1.5,
    "redemption_fee": 1.0
  }},
  "portfolio": [
    {{
      "security_name": "Vingroup JSC",
      "security_code": "VIC",
      "asset_type": "Stock",
      "percentage": 8.5
    }}
  ]
}}

Now extract the data from the provided document."""

    def _clean_response(self, text: str) -> str:
        """Clean markdown formatting from response"""
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()

class DocumentProcessingService:
    """Service for processing documents asynchronously"""
    
    def __init__(self):
        self._gemini_service = None
        self._mistral_service = None
    
    def _get_gemini_service(self):
        """Lazy initialization of Gemini service"""
        if self._gemini_service is None:
            self._gemini_service = GeminiOCRService()
        return self._gemini_service
    
    def _get_mistral_service(self):
        """Lazy initialization of Mistral service"""
        if self._mistral_service is None:
            self._mistral_service = MistralOCRService()
        return self._mistral_service
    
    def process_document(self, document_id: int):
        thread = threading.Thread(
            target=self._process_document_task,
            args=(document_id,),
            daemon=True
        )
        thread.start()
        logger.info(f"Started processing thread for document {document_id}")
    
    def _process_document_task(self, document_id: int):
        optimized_pdf_path = None
        try:
            document = Document.objects.get(id=document_id)
            document.status = 'processing'
            document.save(update_fields=['status'])

            # --- STEP 1: Optimize PDF (Page Segmentation) ---
            try:
                # This function returns a temp file path containing only relevant pages
                optimized_pdf_path = create_optimized_pdf(document.file.path)
                logger.info(f"Optimized PDF created at: {optimized_pdf_path}")
            except Exception as e:
                logger.warning(f"PDF optimization failed, using original file: {e}")
                optimized_pdf_path = document.file.path

            # --- STEP 2: Call AI Service ---
            try:
                if document.ocr_model == 'mistral':
                    extracted_data = self._get_mistral_service().extract_structured_data(optimized_pdf_path)
                else:
                    extracted_data = self._get_gemini_service().extract_structured_data(optimized_pdf_path)
            except Exception as e:
                # If optimization caused an issue (e.g. 400 error), try original file as fallback
                if optimized_pdf_path != document.file.path:
                    logger.warning(f"Extraction failed with optimized PDF, retrying with original: {e}")
                    if document.ocr_model == 'mistral':
                        extracted_data = self._get_mistral_service().extract_structured_data(document.file.path)
                    else:
                        extracted_data = self._get_gemini_service().extract_structured_data(document.file.path)
                else:
                    raise e
            
            # extracted_data is now a dict (or should be)
            if not isinstance(extracted_data, dict):
                 # Fallback if something went wrong and we got a string or something else
                 logger.warning(f"extracted_data is not a dict: {type(extracted_data)}")
                 if isinstance(extracted_data, str):
                     try:
                         extracted_data = json.loads(extracted_data)
                     except:
                         extracted_data = {}

            document.extracted_data = extracted_data
            
            # Normalize data - extract fees from nested structure if present
            fees_obj = extracted_data.get('fees', {})
            if isinstance(fees_obj, dict):
                management_fee = fees_obj.get('management_fee')
                subscription_fee = fees_obj.get('subscription_fee')
                redemption_fee = fees_obj.get('redemption_fee')
                switching_fee = fees_obj.get('switching_fee')
            else:
                management_fee = extracted_data.get('management_fee')
                subscription_fee = extracted_data.get('subscription_fee')
                redemption_fee = extracted_data.get('redemption_fee')
                switching_fee = extracted_data.get('switching_fee')

            # Ensure array fields are never null
            portfolio_value = extracted_data.get('portfolio')
            if portfolio_value is None or not isinstance(portfolio_value, list):
                if portfolio_value is not None:
                    logger.warning(f"Unexpected portfolio type: {type(portfolio_value)}; defaulting to []")
                portfolio_value = []
            extracted_data['portfolio'] = portfolio_value
            
            # Log the extraction result for debugging
            logger.info(f"Extracted Portfolio Items: {len(portfolio_value)}")

            nav_history_value = extracted_data.get('nav_history')
            if nav_history_value is None or not isinstance(nav_history_value, list):
                nav_history_value = []
            extracted_data['nav_history'] = nav_history_value
            logger.info(f"Extracted NAV History Items: {len(nav_history_value)}")

            dividend_history_value = extracted_data.get('dividend_history')
            if dividend_history_value is None or not isinstance(dividend_history_value, list):
                dividend_history_value = []
            extracted_data['dividend_history'] = dividend_history_value
            logger.info(f"Extracted Dividend History Items: {len(dividend_history_value)}")

            # Update the document's extracted_data with the sanitized values
            document.extracted_data = extracted_data

            ExtractedFundData.objects.update_or_create(
                document=document,
                defaults={
                    'fund_name': extracted_data.get('fund_name'),
                    'fund_code': extracted_data.get('fund_code'),
                    'management_company': extracted_data.get('management_company'),
                    'custodian_bank': extracted_data.get('custodian_bank'), #Ngân hàng lưu ký
                    'management_fee': str(management_fee) if management_fee is not None else None,
                    'subscription_fee': str(subscription_fee) if subscription_fee is not None else None,
                    'redemption_fee': str(redemption_fee) if redemption_fee is not None else None,
                    'switching_fee': str(switching_fee) if switching_fee is not None else None,
                    'portfolio': portfolio_value,
                    'nav_history': nav_history_value,
                    'dividend_history': dividend_history_value,
                }
            )

            document.status = 'completed'
            document.processed_at = timezone.now()
            
            # Save the optimized PDF if it exists and is different from the original
            if optimized_pdf_path and optimized_pdf_path != document.file.path:
                if os.path.exists(optimized_pdf_path):
                    from django.core.files import File
                    with open(optimized_pdf_path, 'rb') as f:
                        document.optimized_file.save(
                            f"optimized_{document.file_name}",
                            File(f),
                            save=False
                        )
                    # We don't delete the temp file here anymore, Django handles the file storage
                    # But we should still clean up the temp file from the OS temp dir
                    try:
                        os.remove(optimized_pdf_path)
                        logger.info(f"Removed temporary file: {optimized_pdf_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file: {e}")

            document.save(update_fields=['status', 'processed_at', 'extracted_data', 'optimized_file'])
            
            logger.info(f"Successfully processed document {document_id}")

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            try:
                document = Document.objects.get(id=document_id)
                document.status = 'failed'
                document.error_message = str(e)
                document.save(update_fields=['status', 'error_message'])
            except Exception as save_error:
                logger.error(f"Failed to update document status: {str(save_error)}")
        
        finally:
            # --- STEP 3: Cleanup Temp File (Fallback) ---
            # If optimized_pdf_path was NOT saved to the model (e.g. error occurred), clean it up here
            if optimized_pdf_path and optimized_pdf_path != document.file.path:
                if os.path.exists(optimized_pdf_path):
                    try:
                        os.remove(optimized_pdf_path)
                        logger.info(f"Removed temporary file (finally block): {optimized_pdf_path}")
                    except Exception:
                        pass # Already handled or file gone
                    except OSError as e:
                        logger.warning(f"Error removing temp file: {e}")
