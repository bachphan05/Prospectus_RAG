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
import io
import google.generativeai as genai
from mistralai import Mistral
from django.conf import settings
from django.utils import timezone
import tempfile
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from .models import Document, ExtractedFundData, DocumentChunk
from django.db.models import F
from django.db import close_old_connections
from pgvector.django import CosineDistance
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import PIL.Image
import PIL.ImageDraw
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
                "giá dịch vụ",      
                "thù lao",           
                "chi phí",           
                "hoa hồng",          
                "tối đa",            
                "% giá trị",        
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
                # Use temperature=0 and top_p=0 for maximum deterministic, consistent results
                generation_config = genai.types.GenerationConfig(
                    temperature=0,
                    top_p=0.1,
                    top_k=1,
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
        """Define the expected JSON schema with Bounding Boxes"""
        
        # Define a reusable definition for a field containing value + location
        # logic: 0-1000 scale for coordinates [ymin, xmin, ymax, xmax]
        field_with_loc = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number", "null"]},
                "page": {"type": "integer", "description": "Page number in the file (1-based)"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[ymin, xmin, ymax, xmax] on 0-1000 scale"
                }
            }
        }

        return {
            "type": "object",
            "properties": {
                # Update simple fields to use the new object structure
                "fund_name": field_with_loc,
                "fund_code": field_with_loc,
                "fund_type": field_with_loc,
                "legal_structure": field_with_loc,
                "license_number": field_with_loc,
                "regulator": field_with_loc,
                "management_company": field_with_loc,
                "custodian_bank": field_with_loc,
                "fund_supervisor": field_with_loc,
                "auditor": field_with_loc,
                "inception_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Fund inception/establishment date (YYYY-MM-DD)"
                },

                # Investment objective & strategy
                "investment_objective": field_with_loc,
                "investment_strategy": field_with_loc,
                "investment_style": field_with_loc,
                "sector_focus": field_with_loc,
                "benchmark": field_with_loc,
                
                # Fees object with location for each fee type
                "fees": {
                    "type": "object",
                    "properties": {
                        "management_fee": field_with_loc,
                        "subscription_fee": field_with_loc,
                        "redemption_fee": field_with_loc,
                        "switching_fee": field_with_loc,
                        "total_expense_ratio": field_with_loc,
                        "custody_fee": field_with_loc,
                        "audit_fee": field_with_loc,
                        "supervisory_fee": field_with_loc,
                        "other_expenses": field_with_loc
                    }
                },

                # Risk factors
                "risk_factors": {
                    "type": "object",
                    "properties": {
                        "concentration_risk": field_with_loc,
                        "liquidity_risk": field_with_loc,
                        "interest_rate_risk": field_with_loc
                    }
                },

                # Operational details
                "operational_details": {
                    "type": "object",
                    "properties": {
                        "trading_frequency": field_with_loc,
                        "cut_off_time": field_with_loc,
                        "nav_calculation_frequency": field_with_loc,
                        "nav_publication": field_with_loc,
                        "settlement_cycle": field_with_loc
                    }
                },

                # Valuation details
                "valuation": {
                    "type": "object",
                    "properties": {
                        "valuation_method": field_with_loc,
                        "pricing_source": field_with_loc
                    }
                },

                # Investment limits/restrictions
                "investment_restrictions": field_with_loc,
                "borrowing_limit": field_with_loc,
                "leverage_limit": field_with_loc,

                # Investor & distribution
                "investor_rights": field_with_loc,
                "distribution_agent": field_with_loc,
                "sales_channels": field_with_loc,
                
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
                
                # Arrays with location for each row
                "portfolio": {
                    "type": "array",
                    "description": "Top portfolio holdings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "security_name": field_with_loc,
                            "security_code": field_with_loc,
                            "asset_type": field_with_loc,
                            "market_value": field_with_loc,
                            "percentage": field_with_loc,
                            "row_bbox": {
                                "type": "object",
                                "properties": {
                                    "page": {"type": "integer"},
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "Bounding box for the whole table row"
                                    }
                                }
                            }
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
                    "description": "Recent NAV (Net Asset Value) history",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": field_with_loc,
                            "nav_per_unit": field_with_loc,
                            "row_bbox": {
                                "type": "object",
                                "properties": {
                                    "page": {"type": "integer"},
                                    "bbox": {"type": "array", "items": {"type": "integer"}}
                                }
                            }
                        }
                    }
                },
                
                "dividend_history": {
                    "type": "array",
                    "description": "Dividend distribution history",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": field_with_loc,
                            "dividend_per_unit": field_with_loc,
                            "payment_date": field_with_loc,
                            "row_bbox": {
                                "type": "object",
                                "properties": {
                                    "page": {"type": "integer"},
                                    "bbox": {"type": "array", "items": {"type": "integer"}}
                                }
                            }
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

### CRITICAL: STRUCTURED DATA WITH BOUNDING BOXES ###

**IMPORTANT**: For every field, return an object with {{ "value": "...", "page": N, "bbox": [ymin, xmin, ymax, xmax] }}.

### CRITICAL: ACCURATE BOUNDING BOX COORDINATES ###

**COORDINATE SYSTEM - READ CAREFULLY**:
- Use a **0-1000 scale** for EACH page independently
- **Origin (0,0)**: TOP-LEFT corner of the page
- **Maximum (1000,1000)**: BOTTOM-RIGHT corner of the page
- **bbox format**: [ymin, xmin, ymax, xmax]

**MEASUREMENT GUIDE**:
1. **ymin**: Distance from TOP edge to TOP of text (0 = very top of page)
2. **xmin**: Distance from LEFT edge to LEFT of text (0 = very left of page)
3. **ymax**: Distance from TOP edge to BOTTOM of text
4. **xmax**: Distance from LEFT edge to RIGHT of text

**CALIBRATION EXAMPLES** (assuming standard A4 page):
- Text at very top-left corner: bbox ≈ [50, 50, 100, 300]
- Text in center of page: bbox ≈ [450, 350, 500, 650]
- Text at bottom-right: bbox ≈ [900, 700, 950, 950]
- Full-width section title: bbox ≈ [200, 100, 250, 900]

**ACCURACY TIPS**:
1. **Measure text boundaries tightly** - don't include excessive whitespace
2. **For multi-line text**: ymin = top of first line, ymax = bottom of last line
3. **For fee tables**: Extract the ENTIRE table section, not individual cells
4. **Verify coordinates**: ymax > ymin, xmax > xmin, all values 0-1000
5. **Common mistakes to avoid**:
   - Don't use (x,y) order - use [ymin, xmin, ymax, xmax]
   - Don't measure from bottom of page - always from top
   - Don't use pixel coordinates - use 0-1000 scale

**VALIDATION RULES**:
- 0 ≤ ymin < ymax ≤ 1000
- 0 ≤ xmin < xmax ≤ 1000
- Typical text height (ymax - ymin): 30-80 for single line, 100-200 for section
- Typical text width varies by content length

**Example Structure**:
If "Quỹ Đầu Tư Cân Bằng VCBF" appears on page 1 at the top-center:
```json
"fund_name": {{
  "value": "Quỹ Đầu Tư Cân Bằng VCBF",
  "page": 1,
  "bbox": [100, 300, 150, 700]
}}
```

For fees, each fee type should be a structured object:
```json
"fees": {{
  "subscription_fee": {{
    "value": "5,0%",
    "page": 3,
    "bbox": [250, 150, 280, 350]
  }},
  "redemption_fee": {{
    "value": "N/A",
    "page": 3,
    "bbox": [290, 150, 320, 350]
  }},
  "management_fee": {{
    "value": "Tối đa 1,5%/năm",
    "page": 3,
    "bbox": [330, 150, 360, 600]
  }},
  "switching_fee": {{
    "value": "1,5%",
    "page": 3,
    "bbox": [370, 150, 400, 350]
  }}
}}
```

For table rows (portfolio, nav_history, dividend_history), each cell should be a structured object:
```json
"portfolio": [
  {{
    "security_name": {{
      "value": "Vingroup JSC",
      "page": 5,
      "bbox": [100, 50, 130, 300]
    }},
    "security_code": {{
      "value": "VIC",
      "page": 5,
      "bbox": [100, 310, 130, 400]
    }},
    "percentage": {{
      "value": 8.5,
      "page": 5,
      "bbox": [100, 750, 130, 900]
    }},
    "row_bbox": {{
      "page": 5,
      "bbox": [100, 50, 130, 950]
    }}
  }}
]
```

IMPORTANT INSTRUCTIONS:
1. Extract data EXACTLY as it appears in the document
2. For Vietnamese text, preserve diacritical marks (ả, ế, ô, etc.)
3. For fees, extract the full text value including percentage sign if present (e.g., "2.5%", "Miễn phí")
4. For currency amounts, extract as numbers without formatting (e.g., "1,000,000 VND" → 1000000)
5. For dates, use ISO format YYYY-MM-DD
6. If a field is not found, set value to null and omit bbox
7. For portfolio holdings, extract ALL available items
8. For NAV history, extract the most recent 10-20 entries if available
9. For Dividend history, look for "Lịch sử chia cổ tức", "Phân phối lợi nhuận"
10. Pay special attention to:
   - Fee structures (management, subscription, redemption fees)
    - Total Expense Ratio (TER) / tổng chi phí (if present)
   - Minimum investment requirements
    - Trading frequency (T+1 / daily / weekly) and cut-off time (giờ chốt lệnh)
   - Asset allocation percentages
    - Benchmark (chỉ số tham chiếu) and investment style (active/passive)
   - Performance data
   - Risk classifications

### CRITICAL: Additional fields to extract for fund evaluation ###
Extract these fields if present (Vietnamese + English labels may appear):
1. Investment objective & strategy:
    - `investment_objective` (mục tiêu đầu tư)
    - `investment_strategy` (chiến lược đầu tư / chính sách đầu tư)
    - `asset_allocation` (cơ cấu tài sản mục tiêu: cổ phiếu / trái phiếu / tiền)
    - `investment_style` (active / passive; “chủ động” / “thụ động”)
    - `sector_focus` (tập trung ngành / diversified)
    - `benchmark` (chỉ số tham chiếu)
2. Fee structure:
    - `fees.total_expense_ratio` (TER / tổng chi phí / tổng tỷ lệ chi phí)
3. Risk factors:
    - `risk_factors.concentration_risk` (rủi ro tập trung)
    - `risk_factors.liquidity_risk` (rủi ro thanh khoản)
    - `risk_factors.interest_rate_risk` (rủi ro lãi suất)
4. Operational details:
    - `operational_details.trading_frequency` (tần suất giao dịch: hàng ngày/tuần; T+1)
    - `operational_details.cut_off_time` (giờ chốt lệnh)
    - `operational_details.nav_calculation_frequency` (tần suất tính NAV: hàng ngày/tuần/tháng)
    - `operational_details.nav_publication` (công bố NAV ở đâu/khi nào)
    - `operational_details.settlement_cycle` (chu kỳ thanh toán: T+1/T+2)
5. Legal & regulatory:
    - `fund_type` (loại quỹ: quỹ mở/quỹ đóng/ETF)
    - `legal_structure` (cấu trúc pháp lý)
    - `license_number` (số giấy phép)
    - `regulator` (cơ quan quản lý: UBCKNN, etc.)
    - `fund_supervisor` (giám sát quỹ / đại diện quỹ)
6. Valuation:
    - `valuation.valuation_method` (phương pháp định giá)
    - `valuation.pricing_source` (nguồn giá)
7. Investment restrictions & limits:
    - `investment_restrictions` (hạn chế đầu tư)
    - `borrowing_limit` (hạn mức vay)
    - `leverage_limit` (đòn bẩy)
8. Investor & distribution:
    - `investor_rights` (quyền nhà đầu tư)
    - `distribution_agent` (đại lý phân phối)
    - `sales_channels` (kênh phân phối)
9. Detailed expenses:
    - `fees.custody_fee` (phí lưu ký)
    - `fees.audit_fee` (phí kiểm toán)
    - `fees.supervisory_fee` (phí giám sát)
    - `fees.other_expenses` (chi phí khác)
5. Governance and partners:
    - `management_company` (công ty quản lý quỹ)
    - `custodian_bank` (ngân hàng giám sát/lưu ký)
    - `auditor` (đơn vị kiểm toán)

### CRITICAL INSTRUCTIONS FOR FEE EXTRACTION:
**YOU MUST EXTRACT ALL 4 FEE TYPES as structured objects with value, page, and bbox.**

Look in sections titled "Thông tin phí", "Biểu phí", or "Các loại phí".

1. **PHÍ PHÁT HÀNH** (Subscription/Issue Fee):
   - Also called: "Phí mua", "Phí đăng ký mua", "Phí giao dịch mua"
   - Usually shown as a percentage like "5,0%" or "5.0%" or "Miễn phí"
   - Map this to: `subscription_fee`
   - Example: {{"value": "5,0%", "page": 3, "bbox": [...]}}

2. **PHÍ MUA LẠI** (Redemption/Exit Fee):
   - Also called: "Phí bán", "Phí rút vốn", "Phí giao dịch bán"
   - May say "N/A", "Không áp dụng", or show percentage
   - Map this to: `redemption_fee`
   - Example: {{"value": "N/A", "page": 3, "bbox": [...]}}

3. **PHÍ QUẢN LÝ THƯỜNG NIÊN** (Management Fee):
   - Also called: "Phí quản lý quỹ", "Phí quản lý hàng năm"
   - Usually shown as: "Tối đa X%/năm" or "X% một năm"
   - Map this to: `management_fee`
   - Example: {{"value": "Tối đa 1,5%/năm", "page": 3, "bbox": [...]}}

4. **PHÍ CHUYỂN ĐỔI** (Switching Fee):
   - Fee for converting between fund types
   - May say "N/A" if not applicable
   - Map this to: `switching_fee`
   - Example: {{"value": "N/A", "page": 3, "bbox": [...]}}

**EXTRACTION RULES:**
- If you see "N/A" in the document, extract it as "N/A" (not null)
- If a fee is truly not mentioned anywhere, use {{"value": null, "page": null, "bbox": null}}
- Extract the EXACT text including "%" sign, "Tối đa", commas, etc.
- Don't convert "5,0%" to "5.0%" - keep original formatting
- Look in tables, text sections, and fee schedules

### CRITICAL INSTRUCTIONS FOR TABLES (Danh mục đầu tư, NAV, Cổ tức):
1. **Detect Table Boundaries**: Financial tables (Portfolio, NAV) often span multiple pages WITHOUT repeating headers. Treat consecutive pages with tabular data as a single continuous table.
2. **"Borderless" Tables**: Many tables do not have grid lines. Use visual alignment (columns) to associate values.
3. **Portfolio (Danh mục đầu tư)**:
   - Extract ALL items available, do not limit to top 10 unless the document explicitly says "Top 10".
   - Look for columns like "Mã CK" (Code), "Tên CK" (Name), "Tỷ trọng" (Weight/%), "Giá trị" (Value).
   - Each cell should be a structured object with value, page, bbox
4. **NAV History (Giá trị tài sản ròng)**:
   - Extract the table usually labeled "Biến động giá trị tài sản ròng" or "Lịch sử NAV".
   - Keys: Date (Ngày), NAV/Unit (Giá trị một ccq).
   - Each cell should be a structured object
5. **Dividend History (Lịch sử chia cổ tức)**:
   - Look for keywords: "Chi trả cổ tức", "Phân phối lợi nhuận".
   - If the fund says "No dividend distributed" (Không chia cổ tức), return an empty array [].

### CRITICAL: You MUST provide CONSISTENT and DETERMINISTIC results ###
Always extract the same data from the same document. Do not vary your responses.

Return ONLY valid JSON matching the schema above. Do not include any explanatory text before or after the JSON.

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
    
    def generate_annotated_image(self, pdf_path: str, page_number: int, bboxes: list) -> str:
        """
        Render a page to an image and burn in highlights.

        IMPORTANT: We draw in *pixel space* on top of the rendered pixmap.
        This keeps Gemini's 0-1000 "visual" coordinates aligned with what the
        user sees, and avoids PDF user-space quirks (rotation / cropbox offsets).
        """
        try:
            doc = fitz.open(pdf_path)
            try:
                page_idx = page_number - 1
                if page_idx < 0 or page_idx >= len(doc):
                    return None

                page = doc[page_idx]

                # Render: keep this consistent with what the frontend shows.
                # Using a 2x scale (~144 DPI for A4) gives crisp previews.
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

                base = PIL.Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")
                overlay = PIL.Image.new("RGBA", base.size, (0, 0, 0, 0))
                draw = PIL.ImageDraw.Draw(overlay)

                w, h = base.size
                border_px = max(2, int(round(min(w, h) * 0.002)))

                # Optional: OCR-snap highlights to the exact rendered text positions.
                # This helps when Gemini's 0-1000 coordinates are systematically biased.
                ocr_results = None
                wants_ocr_snap = any(
                    isinstance(it, dict) and str(it.get("value") or "").strip()
                    for it in (bboxes or [])
                )
                if wants_ocr_snap:
                    try:
                        img_bytes = pix.tobytes("png")
                        result = ocr_engine(img_bytes)
                        if result and isinstance(result, tuple):
                            result = result[0]
                        ocr_results = result or []
                    except Exception as ocr_error:
                        logger.debug(f"OCR snap failed for preview page {page_number}: {ocr_error}")
                        ocr_results = None

                def _norm_for_match(s: str) -> str:
                    s = str(s or "")
                    # keep alnum + spaces so we can do token-ish matching
                    s = unicodedata.normalize('NFD', s)
                    s = ''.join(c for c in s if not unicodedata.combining(c))
                    s = s.replace('đ', 'd').replace('Đ', 'D')
                    s = s.lower()
                    s = re.sub(r"[^a-z0-9\s]", " ", s)
                    s = re.sub(r"\s+", " ", s).strip()
                    return s

                def _ocr_best_rect(target_value: str):
                    if not ocr_results:
                        return None
                    tv = _norm_for_match(target_value)
                    if not tv or len(tv) < 3:
                        return None

                    best = None
                    best_score = 0.0

                    for res in ocr_results:
                        try:
                            box, text, conf = res[0], res[1], res[2]
                        except Exception:
                            continue

                        if not text:
                            continue
                        ct = _norm_for_match(text)
                        if not ct:
                            continue

                        # Simple robust scoring: containment + token overlap.
                        score = 0.0
                        if tv == ct:
                            score = 1.0
                        elif tv in ct or ct in tv:
                            score = 0.9
                        else:
                            tv_tokens = set(tv.split())
                            ct_tokens = set(ct.split())
                            if tv_tokens and ct_tokens:
                                overlap = len(tv_tokens & ct_tokens) / max(1, len(tv_tokens))
                                score = 0.6 * overlap

                        # Prefer higher OCR confidence.
                        try:
                            score *= float(conf) if conf is not None else 1.0
                        except Exception:
                            pass

                        if score > best_score:
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            x0, x1 = min(xs), max(xs)
                            y0, y1 = min(ys), max(ys)
                            best = (x0, y0, x1, y1)
                            best_score = score

                    # Require at least some confidence.
                    if best and best_score >= 0.35:
                        return best
                    return None

                for item in bboxes or []:
                    bbox = item.get('bbox') if isinstance(item, dict) else None
                    if not bbox or len(bbox) != 4:
                        continue

                    ymin, xmin, ymax, xmax = bbox

                    # Clamp + convert to pixels.
                    xmin = max(0, min(1000, int(xmin)))
                    xmax = max(0, min(1000, int(xmax)))
                    ymin = max(0, min(1000, int(ymin)))
                    ymax = max(0, min(1000, int(ymax)))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    x0 = int(round((xmin / 1000.0) * w))
                    x1 = int(round((xmax / 1000.0) * w))
                    y0 = int(round((ymin / 1000.0) * h))
                    y1 = int(round((ymax / 1000.0) * h))

                    # If we can OCR-locate the actual value on the rendered page,
                    # snap the rectangle to that (much more accurate on scanned PDFs).
                    if isinstance(item, dict):
                        val = item.get("value")
                        if val is not None:
                            snapped = _ocr_best_rect(str(val))
                            if snapped:
                                x0, y0, x1, y1 = [int(round(v)) for v in snapped]

                    # Semi-transparent yellow fill + stronger border.
                    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 0, 80))
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 200, 0, 255), width=border_px)

                out = PIL.Image.alpha_composite(base, overlay).convert("RGB")

                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                safe_pdf_name = re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_name)[:80] or "document"
                output_filename = f"annotated_{safe_pdf_name}_page_{page_number}.png"
                output_path = os.path.join(settings.MEDIA_ROOT, 'temp', output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                out.save(output_path, format="PNG")
                return output_path
            finally:
                doc.close()

        except Exception as e:
            logger.error(f"Failed to generate annotation: {e}")
            return None

class MistralOCRSmallService:
    """
    Service for OCR using Mistral's native OCR API (Step 1) 
    + Mistral Small for JSON extraction (Step 2)
    """
    
    def __init__(self):
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        
        self.client = Mistral(api_key=api_key)
        # We use the specific OCR endpoint, not a chat model name for step 1
        self.extraction_model = "mistral-small-latest"  
    
    def extract_structured_data(self, pdf_path: str) -> dict:
        try:
            logger.info(f"Uploading PDF to Mistral OCR: {pdf_path}")
            
            # --- STEP 1: Upload file to Mistral (Required for OCR API) ---
            with open(pdf_path, "rb") as f:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": os.path.basename(pdf_path),
                        "content": f,
                    },
                    purpose="ocr"
                )
            
            # Get signed URL
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # --- STEP 2: Run Native Mistral OCR ---
            logger.info("Running Mistral OCR...")
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                include_image_base64=False 
            )
            
            # Combine markdown from all pages
            # Mistral OCR returns pages with 'markdown' content
            full_markdown = ""
            for i, page in enumerate(ocr_response.pages):
                full_markdown += f"\n\n--- PAGE {i+1} ---\n{page.markdown}"
            
            logger.info(f"OCR Success. Extracted {len(full_markdown)} characters.")

            # --- STEP 3: Parse with Mistral Small ---
            prompt = self._get_extraction_prompt()
            
            chat_response = self.client.chat.complete(
                model=self.extraction_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data extraction assistant specializing in Vietnamese mutual fund documents. Extract structured data accurately and return ONLY valid JSON. Pay special attention to Vietnamese field names like 'Công ty quản lý quỹ' and 'Ngân hàng giám sát'."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n--- DOCUMENT CONTENT (Markdown from OCR) ---\n{full_markdown}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0
            )

            response_content = chat_response.choices[0].message.content
            
            # Cleanup (Optional: Delete file from Mistral storage if needed, 
            # though Mistral usually cleans up temp files automatically or allows them to expire)
            
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON: {response_content[:200]}...")
                return {"error": "Failed to parse JSON", "raw_text": response_content}

        except Exception as e:
            logger.error(f"Error in Mistral OCR+Small pipeline: {str(e)}")
            raise

    def _get_extraction_prompt(self) -> str:
        """Get the extraction prompt - matches Gemini's detailed instructions"""
        return """You are extracting structured data from a Vietnamese mutual fund prospectus (Bản cáo bạch quỹ mở).

Extract the following information and return as JSON:

{
  "fund_name": "Tên quỹ",
  "fund_code": "Mã quỹ",
    "fund_type": "Loại quỹ (quỹ mở/quỹ đóng/ETF)",
    "legal_structure": "Cấu trúc pháp lý",
    "license_number": "Số giấy phép/Quyết định",
    "regulator": "Cơ quan quản lý (UBCKNN, ...)",
  "management_company": "Công ty quản lý quỹ",
  "custodian_bank": "Ngân hàng giám sát / Ngân hàng lưu ký",
    "fund_supervisor": "Giám sát quỹ / đại diện quỹ (nếu có)",
  "fees": {
    "management_fee": "Phí quản lý (%/năm)",
    "subscription_fee": "Phí mua/đăng ký",
    "redemption_fee": "Phí bán/hoàn mua",
        "switching_fee": "Phí chuyển đổi",
        "total_expense_ratio": "TER / Tổng tỷ lệ chi phí (nếu có)",
        "custody_fee": "Phí lưu ký (nếu có)",
        "audit_fee": "Phí kiểm toán (nếu có)",
        "supervisory_fee": "Phí giám sát (nếu có)",
        "other_expenses": "Chi phí khác (nếu có)"
  },
    "investment_objective": "Mục tiêu đầu tư",
    "investment_strategy": "Chiến lược/Chính sách đầu tư",
    "investment_style": "Chủ động/Thụ động (Active/Passive)",
    "sector_focus": "Tập trung ngành (nếu có)",
    "benchmark": "Chỉ số tham chiếu (nếu có)",
    "investment_restrictions": "Hạn chế đầu tư",
    "borrowing_limit": "Hạn mức vay",
    "leverage_limit": "Giới hạn đòn bẩy",
    "risk_factors": {
        "concentration_risk": "Rủi ro tập trung",
        "liquidity_risk": "Rủi ro thanh khoản",
        "interest_rate_risk": "Rủi ro lãi suất"
    },
    "operational_details": {
        "trading_frequency": "Tần suất giao dịch (Daily/Weekly/T+1, ...)",
        "cut_off_time": "Giờ chốt lệnh",
        "nav_calculation_frequency": "Tần suất tính NAV",
        "nav_publication": "Công bố NAV",
        "settlement_cycle": "Chu kỳ thanh toán (T+1/T+2)"
    },
    "valuation": {
        "valuation_method": "Phương pháp định giá",
        "pricing_source": "Nguồn giá"
    },
    "auditor": "Đơn vị kiểm toán",
    "distribution_agent": "Đại lý phân phối",
    "sales_channels": "Kênh phân phối",
    "investor_rights": "Quyền nhà đầu tư",
    "asset_allocation": {"stocks": 0, "bonds": 0, "cash": 0, "other": 0},
    "minimum_investment": {"initial": 0, "additional": 0, "currency": "VND"},
  "portfolio": [
    {"asset": "Tên tài sản", "value": "Giá trị", "percentage": "Tỷ lệ %"}
  ],
  "nav_history": [
    {"date": "Ngày", "nav": "Giá trị NAV"}
  ],
  "dividend_history": [
    {"date": "Ngày", "amount": "Số tiền"}
  ]
}

IMPORTANT EXTRACTION RULES:
1. Return ONLY valid JSON
2. Use null for missing values
3. Empty arrays [] for missing lists
4. Keep original Vietnamese text
5. Extract exact numbers and percentages

KEY FIELD MAPPINGS (Vietnamese → JSON):
- "Tên quỹ" → fund_name
- "Mã quỹ" / "Mã chứng chỉ quỹ" → fund_code
- "Công ty quản lý" / "Công ty quản lý quỹ" / "CTQL" → management_company
- "Ngân hàng giám sát" / "Ngân hàng lưu ký" / "NH giám sát" → custodian_bank

FEE EXTRACTION (Look in "Thông tin phí", "Biểu phí" sections):
1. "Phí phát hành" / "Phí mua" / "Phí đăng ký mua" → subscription_fee
2. "Phí mua lại" / "Phí bán" / "Phí rút vốn" → redemption_fee
3. "Phí quản lý" / "Phí quản lý thường niên" → management_fee
4. "Phí chuyển đổi" → switching_fee
5. "TER" / "Tổng tỷ lệ chi phí" / "Tổng chi phí" → total_expense_ratio

ADDITIONAL IMPORTANT FIELDS:
- Auditor: "Đơn vị kiểm toán" / "Công ty kiểm toán"
- Benchmark: "Chỉ số tham chiếu" / "Benchmark"
- Trading frequency: "Tần suất giao dịch" / "Kỳ giao dịch" / "T+1" / "hàng ngày"
- Cut-off time: "Giờ chốt lệnh" / "Thời điểm chốt lệnh"
- NAV calculation/publication: "Tần suất tính NAV" / "Công bố NAV"
- Settlement: "Chu kỳ thanh toán" / "T+1" / "T+2"
- License/regulator: "Số giấy phép" / "Quyết định" / "UBCKNN"
- Valuation: "Định giá" / "Phương pháp định giá" / "Nguồn giá"
- Distribution: "Đại lý phân phối" / "Kênh phân phối" / "Điểm giao dịch"
- Restrictions/limits: "Hạn chế đầu tư" / "Hạn mức vay" / "Đòn bẩy"
- Expenses: "phí lưu ký" / "phí kiểm toán" / "phí giám sát" / "chi phí khác"
- Risk factors: look for "Rủi ro" sections and extract the relevant paragraphs

Extract fees EXACTLY as written (keep "5,0%", "Tối đa 1,5%/năm", "N/A", etc.)

Now extract the data from the provided document."""
class MistralOCRService:
    """Service for OCR using Mistral AI (Large model with text extraction)"""
    
    def __init__(self):
        # Configure Mistral API
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"
        # Model used for the JSON extraction (chat) step
        self.extraction_model = self.model
    
    def extract_structured_data(self, pdf_path: str) -> dict:
        try:
            logger.info(f"Uploading PDF to Mistral OCR: {pdf_path}")
            
            # BƯỚC 1: Upload file lên Mistral (Bắt buộc cho OCR API)
            with open(pdf_path, "rb") as f:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": os.path.basename(pdf_path),
                        "content": f,
                    },
                    purpose="ocr"
                )
            
            # Lấy signed URL để xử lý
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # BƯỚC 2: Gọi Native OCR API (Dùng hàm ocr.process)
            # Lưu ý: Không cần convert sang ảnh thủ công, Mistral tự đọc PDF
            logger.info("Running Mistral OCR...")
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                include_image_base64=False 
            )
            
            # Gộp kết quả Markdown từ các trang
            full_markdown = ""
            for i, page in enumerate(ocr_response.pages):
                full_markdown += f"\n\n--- PAGE {i+1} ---\n{page.markdown}"
            
            logger.info(f"OCR Success. Extracted {len(full_markdown)} characters.")

            # BƯỚC 3: Gửi Markdown sang Mistral Small để lấy JSON
            # (Lúc này mới dùng chat.complete)
            prompt = self._get_extraction_prompt()
            
            chat_response = self.client.chat.complete(
                model=self.extraction_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data assistant. Extract data from the provided Markdown content into valid JSON."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n--- DOCUMENT CONTENT (Markdown) ---\n{full_markdown}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0
            )

            response_content = chat_response.choices[0].message.content
            
            return json.loads(response_content)

        except Exception as e:
            logger.error(f"Error in Mistral OCR+Small pipeline: {str(e)}")
            # Nếu lỗi này là do chưa add thẻ (400/403), hãy check log
            if "Invalid model" in str(e) or "400" in str(e):
                logger.error("HINT: Ensure your Mistral account has Billing enabled for OCR models.")
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
                "fund_type": {
                    "type": "string",
                    "description": "Fund type (open-end/closed-end/ETF)"
                },
                "legal_structure": {
                    "type": "string",
                    "description": "Legal structure of the fund"
                },
                "license_number": {
                    "type": "string",
                    "description": "License / decision number"
                },
                "regulator": {
                    "type": "string",
                    "description": "Regulatory authority"
                },
                "management_company": {
                    "type": "string",
                    "description": "Name of the fund management company"
                },
                "custodian_bank": {
                    "type": "string",
                    "description": "Name of the custodian bank"
                },
                "fund_supervisor": {
                    "type": "string",
                    "description": "Fund supervisor/representative if applicable"
                },
                "auditor": {
                    "type": "string",
                    "description": "Auditor of the fund (e.g. Big4)"
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
                "investment_strategy": {
                    "type": "string",
                    "description": "Investment strategy / policy"
                },
                "investment_style": {
                    "type": "string",
                    "description": "Active or passive (chủ động / thụ động)"
                },
                "sector_focus": {
                    "type": "string",
                    "description": "Sector focus or diversification statement"
                },
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark / index referenced by the fund"
                },
                "investment_restrictions": {
                    "type": "string",
                    "description": "Key investment restrictions"
                },
                "borrowing_limit": {
                    "type": "string",
                    "description": "Borrowing limit"
                },
                "leverage_limit": {
                    "type": "string",
                    "description": "Leverage limit"
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
                        },
                        "total_expense_ratio": {
                            "type": "string",
                            "description": "Total Expense Ratio (TER) / Total costs if present"
                        },
                        "custody_fee": {
                            "type": "string",
                            "description": "Custody fee"
                        },
                        "audit_fee": {
                            "type": "string",
                            "description": "Audit fee"
                        },
                        "supervisory_fee": {
                            "type": "string",
                            "description": "Supervisory/custodian bank supervision fee"
                        },
                        "other_expenses": {
                            "type": "string",
                            "description": "Other expenses"
                        }
                    }
                },
                "risk_factors": {
                    "type": "object",
                    "description": "Key risk factors",
                    "properties": {
                        "concentration_risk": {"type": "string"},
                        "liquidity_risk": {"type": "string"},
                        "interest_rate_risk": {"type": "string"}
                    }
                },
                "operational_details": {
                    "type": "object",
                    "description": "Operational/trading details",
                    "properties": {
                        "trading_frequency": {"type": "string"},
                        "cut_off_time": {"type": "string"},
                        "nav_calculation_frequency": {"type": "string"},
                        "nav_publication": {"type": "string"},
                        "settlement_cycle": {"type": "string"}
                    }
                },
                "valuation": {
                    "type": "object",
                    "description": "Valuation and pricing",
                    "properties": {
                        "valuation_method": {"type": "string"},
                        "pricing_source": {"type": "string"}
                    }
                },
                "investor_rights": {
                    "type": "string",
                    "description": "Key investor rights"
                },
                "distribution_agent": {
                    "type": "string",
                    "description": "Distribution agent"
                },
                "sales_channels": {
                    "type": "string",
                    "description": "Sales channels"
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
    - Total Expense Ratio (TER) / tổng chi phí
    - Fee breakdown (custody/audit/supervisory/other)
   - Minimum investment requirements
    - Trading frequency and cut-off time
    - NAV calculation/publication and settlement cycle
   - Asset allocation percentages
    - Benchmark and investment style
    - Legal/regulatory identifiers (license number, regulator)
    - Valuation method and pricing source
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
        self._mistral_ocr_small_service = None
    
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
    
    def _get_mistral_ocr_small_service(self):
        """Lazy initialization of Mistral OCR + Small service"""
        if self._mistral_ocr_small_service is None:
            self._mistral_ocr_small_service = MistralOCRSmallService()
        return self._mistral_ocr_small_service
    
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
            import time
            start_time = time.time()
            
            try:
                logger.info(f"Starting extraction with model: {document.ocr_model}")
                if document.ocr_model == 'mistral':
                    extracted_data = self._get_mistral_service().extract_structured_data(optimized_pdf_path)
                elif document.ocr_model == 'mistral-ocr':
                    extracted_data = self._get_mistral_ocr_small_service().extract_structured_data(optimized_pdf_path)
                else:
                    extracted_data = self._get_gemini_service().extract_structured_data(optimized_pdf_path)
                
                extraction_time = time.time() - start_time
                logger.info(f"✓ Extraction completed with {document.ocr_model} in {extraction_time:.2f} seconds")
                
            except Exception as e:
                # If optimization caused an issue (e.g. 400 error), try original file as fallback
                if optimized_pdf_path != document.file.path:
                    logger.warning(f"Extraction failed with optimized PDF, retrying with original: {e}")
                    start_time = time.time()
                    if document.ocr_model == 'mistral':
                        extracted_data = self._get_mistral_service().extract_structured_data(document.file.path)
                    elif document.ocr_model == 'mistral-ocr':
                        extracted_data = self._get_mistral_ocr_small_service().extract_structured_data(document.file.path)
                    else:
                        extracted_data = self._get_gemini_service().extract_structured_data(document.file.path)
                    extraction_time = time.time() - start_time
                    logger.info(f"✓ Extraction completed (fallback) with {document.ocr_model} in {extraction_time:.2f} seconds")
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
            
            # Handle case where fees might be a list instead of dict (API error)
            if isinstance(fees_obj, list):
                logger.warning(f"Fees returned as list instead of dict: {fees_obj}")
                fees_obj = {}
            
            if isinstance(fees_obj, dict) and fees_obj:
                management_fee = fees_obj.get('management_fee')
                subscription_fee = fees_obj.get('subscription_fee')
                redemption_fee = fees_obj.get('redemption_fee')
                switching_fee = fees_obj.get('switching_fee')
            else:
                # Fallback to top-level fields
                management_fee = extracted_data.get('management_fee')
                subscription_fee = extracted_data.get('subscription_fee')
                redemption_fee = extracted_data.get('redemption_fee')
                switching_fee = extracted_data.get('switching_fee')
            
            # Log extracted fees for debugging
            logger.info(f"Extracted Fees - Management: {management_fee}, Subscription: {subscription_fee}, Redemption: {redemption_fee}, Switching: {switching_fee}")

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
            
            # Log key extracted fields for comparison
            logger.info(f"📊 Model: {document.ocr_model} | Extraction Summary:")
            logger.info(f"  ├─ Fund Name: {extracted_data.get('fund_name', 'NOT FOUND')}")
            logger.info(f"  ├─ Fund Code: {extracted_data.get('fund_code', 'NOT FOUND')}")
            logger.info(f"  ├─ Management Company: {extracted_data.get('management_company', 'NOT FOUND')}")
            logger.info(f"  ├─ Custodian Bank: {extracted_data.get('custodian_bank', 'NOT FOUND')}")
            logger.info(f"  ├─ Portfolio Items: {len(extracted_data.get('portfolio', []))}")
            logger.info(f"  ├─ NAV History: {len(extracted_data.get('nav_history', []))}")
            logger.info(f"  └─ Dividend History: {len(extracted_data.get('dividend_history', []))}")

            # Helper function to extract value from either structured or flat format
            def get_value(field_data):
                """Extract value from either {value, page, bbox} object or plain string/number"""
                if isinstance(field_data, dict) and 'value' in field_data:
                    return field_data['value']  # New Gemini structured format
                return field_data  # Old flat format (Mistral or legacy data)

            def get_nested_value(data: dict, *path, default=None):
                cur = data
                for key in path:
                    if not isinstance(cur, dict):
                        return default
                    cur = cur.get(key)
                return get_value(cur) if cur is not None else default

            def truncate_defaults_for_model(model_cls, defaults: dict) -> dict:
                """Ensure values fit DB column limits (e.g., CharField max_length).

                This prevents crashes when the LLM returns long paragraphs for short fields.
                The full (untruncated) value is still preserved in Document.extracted_data.
                """
                sanitized = dict(defaults)
                for field_name, field_value in sanitized.items():
                    if not isinstance(field_value, str) or field_value is None:
                        continue
                    try:
                        model_field = model_cls._meta.get_field(field_name)
                    except Exception:
                        continue

                    max_len = getattr(model_field, 'max_length', None)
                    if max_len and len(field_value) > max_len:
                        logger.warning(
                            f"Truncating {model_cls.__name__}.{field_name}: {len(field_value)} -> {max_len} chars"
                        )
                        sanitized[field_name] = field_value[:max_len]
                return sanitized

            fund_defaults = {
                    'fund_name': get_value(extracted_data.get('fund_name')),
                    'fund_code': get_value(extracted_data.get('fund_code')),
                    'fund_type': get_nested_value(extracted_data, 'fund_type'),
                    'legal_structure': get_nested_value(extracted_data, 'legal_structure'),
                    'license_number': get_nested_value(extracted_data, 'license_number') or get_nested_value(extracted_data, 'license') or get_nested_value(extracted_data, 'license_no'),
                    'regulator': get_nested_value(extracted_data, 'regulator'),
                    'management_company': get_value(extracted_data.get('management_company')),
                    'custodian_bank': get_value(extracted_data.get('custodian_bank')),
                    'fund_supervisor': get_nested_value(extracted_data, 'fund_supervisor') or get_nested_value(extracted_data, 'governance', 'fund_supervisor'),
                    'management_fee': str(get_value(management_fee)) if management_fee is not None else None,
                    'subscription_fee': str(get_value(subscription_fee)) if subscription_fee is not None else None,
                    'redemption_fee': str(get_value(redemption_fee)) if redemption_fee is not None else None,
                    'switching_fee': str(get_value(switching_fee)) if switching_fee is not None else None,
                    'total_expense_ratio': str(get_nested_value(extracted_data, 'fees', 'total_expense_ratio')) if get_nested_value(extracted_data, 'fees', 'total_expense_ratio') is not None else None,
                    'custody_fee': str(get_nested_value(extracted_data, 'fees', 'custody_fee')) if get_nested_value(extracted_data, 'fees', 'custody_fee') is not None else None,
                    'audit_fee': str(get_nested_value(extracted_data, 'fees', 'audit_fee')) if get_nested_value(extracted_data, 'fees', 'audit_fee') is not None else None,
                    'supervisory_fee': str(get_nested_value(extracted_data, 'fees', 'supervisory_fee')) if get_nested_value(extracted_data, 'fees', 'supervisory_fee') is not None else None,
                    'other_expenses': get_nested_value(extracted_data, 'fees', 'other_expenses') or get_nested_value(extracted_data, 'other_expenses'),

                    'investment_objective': get_nested_value(extracted_data, 'investment_objective') or get_nested_value(extracted_data, 'objective') or get_nested_value(extracted_data, 'investment', 'objective'),
                    'investment_strategy': get_nested_value(extracted_data, 'investment_strategy') or get_nested_value(extracted_data, 'strategy') or get_nested_value(extracted_data, 'investment', 'strategy'),
                    'investment_style': get_nested_value(extracted_data, 'investment_style') or get_nested_value(extracted_data, 'style'),
                    'sector_focus': get_nested_value(extracted_data, 'sector_focus') or get_nested_value(extracted_data, 'sector'),
                    'benchmark': get_nested_value(extracted_data, 'benchmark'),

                    'investment_restrictions': get_nested_value(extracted_data, 'investment_restrictions') or get_nested_value(extracted_data, 'investment', 'restrictions'),
                    'borrowing_limit': get_nested_value(extracted_data, 'borrowing_limit') or get_nested_value(extracted_data, 'investment', 'borrowing_limit'),
                    'leverage_limit': get_nested_value(extracted_data, 'leverage_limit') or get_nested_value(extracted_data, 'investment', 'leverage_limit'),

                    'concentration_risk': get_nested_value(extracted_data, 'risk_factors', 'concentration_risk') or get_nested_value(extracted_data, 'concentration_risk'),
                    'liquidity_risk': get_nested_value(extracted_data, 'risk_factors', 'liquidity_risk') or get_nested_value(extracted_data, 'liquidity_risk'),
                    'interest_rate_risk': get_nested_value(extracted_data, 'risk_factors', 'interest_rate_risk') or get_nested_value(extracted_data, 'interest_rate_risk'),

                    'trading_frequency': get_nested_value(extracted_data, 'operational_details', 'trading_frequency') or get_nested_value(extracted_data, 'trading_frequency'),
                    'cut_off_time': get_nested_value(extracted_data, 'operational_details', 'cut_off_time') or get_nested_value(extracted_data, 'cut_off_time'),
                    'nav_calculation_frequency': get_nested_value(extracted_data, 'operational_details', 'nav_calculation_frequency') or get_nested_value(extracted_data, 'nav_calculation_frequency'),
                    'nav_publication': get_nested_value(extracted_data, 'operational_details', 'nav_publication') or get_nested_value(extracted_data, 'nav_publication'),
                    'settlement_cycle': get_nested_value(extracted_data, 'operational_details', 'settlement_cycle') or get_nested_value(extracted_data, 'settlement_cycle'),

                    'valuation_method': get_nested_value(extracted_data, 'valuation', 'valuation_method') or get_nested_value(extracted_data, 'valuation_method'),
                    'pricing_source': get_nested_value(extracted_data, 'valuation', 'pricing_source') or get_nested_value(extracted_data, 'pricing_source'),

                    'investor_rights': get_nested_value(extracted_data, 'investor_rights') or get_nested_value(extracted_data, 'investor', 'rights'),
                    'distribution_agent': get_nested_value(extracted_data, 'distribution_agent') or get_nested_value(extracted_data, 'distribution', 'agent'),
                    'sales_channels': get_nested_value(extracted_data, 'sales_channels') or get_nested_value(extracted_data, 'distribution', 'sales_channels'),

                    'auditor': get_nested_value(extracted_data, 'governance', 'auditor') or get_nested_value(extracted_data, 'auditor'),

                    'asset_allocation': extracted_data.get('asset_allocation') if isinstance(extracted_data.get('asset_allocation'), dict) else {},
                    'minimum_investment': extracted_data.get('minimum_investment') if isinstance(extracted_data.get('minimum_investment'), dict) else {},
                    'portfolio': portfolio_value,
                    'nav_history': nav_history_value,
                    'dividend_history': dividend_history_value,

            }

            fund_defaults = truncate_defaults_for_model(ExtractedFundData, fund_defaults)

            ExtractedFundData.objects.update_or_create(
                document=document,
                defaults=fund_defaults,
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

class RAGService:
    """
    Service for Retrieval-Augmented Generation (Chat with PDF).
    Handles:
    1. Extracting full text/markdown from documents.
    2. Chunking text and generating embeddings.
    3. Storing vectors in PostgreSQL.
    4. Retrieving relevant chunks and answering user queries.
    """

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        self.chat_model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def ingest_document(self, document_id: int) -> bool:
        """
        Process a document into vector chunks for RAG.
        Returns True if successful.
        """
        try:
            document = Document.objects.get(id=document_id)
            logger.info(f"Starting RAG ingestion for Doc {document_id}")

            # 1. Check if already ingested to avoid duplicates
            if document.chunks.exists():
                logger.info(f"Document {document_id} already ingested. Deleting old chunks...")
                document.chunks.all().delete()

            # 2. Extract Raw Content (optimized for Scanned PDFs)
            # Since Mistral/Gemini services return JSON, we might not have the full text saved.
            # We call a helper to get the raw markdown representation.
            full_text = self._extract_content_for_rag(document)
            
            if not full_text:
                raise ValueError("Could not extract text content from document")

            # 3. Chunking Strategy
            # Parse page markers (supports both formats: "--- PAGE X ---" and "=== PAGE X ===")
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

            # 4. Generate Embeddings & Save (Batch Processing)
            batch_size = 10  # Reduced batch size to avoid SSL timeouts
            chunks_to_create = []
            
            for i in range(0, len(all_chunks_with_pages), batch_size):
                batch = all_chunks_with_pages[i:i + batch_size]
                batch_texts = [d.page_content for d in batch]
                
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
                            import time
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
            
            return True

        except Exception as e:
            logger.error(f"RAG Ingestion Error: {str(e)}")
            raise

    def chat(self, document_id: int, user_query: str, history: list = None) -> str:
        """
        Answer a user question using RAG.
        """
        try:
            document = Document.objects.get(id=document_id)

            # 1. Lấy dữ liệu cấu trúc đã trích xuất ("Phao cứu sinh" cho câu hỏi về phí, tên, mã...)
            structured_info = ""

            extracted_data = document.extracted_data or {}
            minimum_investment = extracted_data.get('minimum_investment')
            investment_objective = extracted_data.get('investment_objective')
            asset_allocation = extracted_data.get('asset_allocation')
            inception_date = extracted_data.get('inception_date')
            effective_date = extracted_data.get('effective_date')

            try:
                fund_data = ExtractedFundData.objects.get(document_id=document_id)
                structured_info = f"""
THÔNG TIN CƠ BẢN ĐÃ ĐƯỢC TRÍCH XUẤT (ƯU TIÊN DÙNG CHO CÂU HỎI VỀ PHÍ / TÊN / MÃ / NGÂN HÀNG):
- Tên quỹ: {fund_data.fund_name}
- Mã quỹ: {fund_data.fund_code}
- Công ty quản lý: {fund_data.management_company}
- Ngân hàng giám sát: {fund_data.custodian_bank}
- Phí quản lý: {fund_data.management_fee}
- Phí phát hành (mua): {fund_data.subscription_fee}
- Phí mua lại (bán): {fund_data.redemption_fee}
- Phí chuyển đổi: {fund_data.switching_fee}
- Ngày thành lập/quỹ bắt đầu hoạt động (inception_date): {inception_date or 'Không có'}
- Ngày hiệu lực (effective_date): {effective_date or 'Không có'}
- Mục tiêu đầu tư: {investment_objective or 'Không có'}
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
- Số tiền đầu tư tối thiểu (ban đầu / bổ sung): {json.dumps(minimum_investment or {}, ensure_ascii=False)}
- Cơ cấu phân bổ tài sản: {json.dumps(asset_allocation or {}, ensure_ascii=False)}
""".strip()

            # 2. Vector Search (Semantic Retrieval) cho câu hỏi giải thích / chiến lược / rủi ro...
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

            # 3. Tổng hợp Prompt: dùng cả JSON + Vector
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
        Fixed: Processes ALL pages (no limit) and handles batching correctly.
        """
        import time 
        
        try:
            # Use ORIGINAL uploaded PDF for RAG extraction (per product requirement)
            # Note: this may be slower than using an optimized/segmented version.
            file_path = document.file.path
            if hasattr(document, 'optimized_file') and document.optimized_file:
                logger.info(
                    "RAG Extraction: optimized_file exists but will be ignored; using original upload instead."
                )

            logger.info(f"Extracting raw content for RAG from ORIGINAL PDF: {file_path}")
            
            doc = fitz.open(file_path)
            full_text = ""
            
            # Process pages in batches of 15 to be safe
            batch_size = 15
            total_pages = len(doc)
            
            logger.info(f"Total pages to ingest: {total_pages}")
            
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"Processing RAG batch: Pages {batch_start + 1} to {batch_end}")
                
                # QUAN TRỌNG: Reset model_inputs cho mỗi batch (để không bị cộng dồn ảnh cũ)
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
                
                # Convert pages in this batch to images
                for i in range(batch_start, batch_end):
                    page = doc[i]
                    # Matrix 1.5 (~108 DPI) đủ nét để đọc chữ nhỏ
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    img_data = pix.tobytes("png")
                    image = PIL.Image.open(io.BytesIO(img_data))
                    # Provide a deterministic page marker right before the image
                    model_inputs.append(f"=== PAGE {i + 1} ===")
                    model_inputs.append(image)
                
                # Gọi Gemini với cơ chế Retry (thử lại nếu lỗi mạng)
                batch_text = ""
                for attempt in range(3):
                    try:
                        response = self.chat_model.generate_content(model_inputs)
                        batch_text = response.text
                        break # Thành công -> thoát vòng lặp retry
                    except Exception as e:
                        wait = (attempt + 1) * 5
                        logger.warning(f"Batch {batch_start}-{batch_end} failed (Attempt {attempt+1}): {e}. Retrying in {wait}s...")
                        time.sleep(wait)
                
                if batch_text:
                    full_text += batch_text + "\n\n"
                else:
                    logger.error(f"Failed to extract text for pages {batch_start+1}-{batch_end} after 3 attempts.")

                # Nghỉ 2 giây giữa các batch để tránh lỗi 429 (Rate Limit)
                time.sleep(2)
            
            doc.close()
            
            if not full_text.strip():
                raise ValueError("Extracted text is empty")
                
            return full_text

        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            # Trả về chuỗi rỗng thay vì crash để luồng chính xử lý tiếp
            return ""
        
    