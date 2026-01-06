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
                "management_company": field_with_loc,
                "custodian_bank": field_with_loc,
                "inception_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Fund inception/establishment date (YYYY-MM-DD)"
                },
                "investment_objective": {
                    "type": "string",
                    "description": "Main investment objective of the fund"
                },
                
                # Fees object with location for each fee type
                "fees": {
                    "type": "object",
                    "properties": {
                        "management_fee": field_with_loc,
                        "subscription_fee": field_with_loc,
                        "redemption_fee": field_with_loc,
                        "switching_fee": field_with_loc
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
   - Minimum investment requirements
   - Asset allocation percentages
   - Performance data
   - Risk classifications

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
  "management_company": "Công ty quản lý quỹ",
  "custodian_bank": "Ngân hàng giám sát / Ngân hàng lưu ký",
  "fees": {
    "management_fee": "Phí quản lý (%/năm)",
    "subscription_fee": "Phí mua/đăng ký",
    "redemption_fee": "Phí bán/hoàn mua",
    "switching_fee": "Phí chuyển đổi"
  },
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

            ExtractedFundData.objects.update_or_create(
                document=document,
                defaults={
                    'fund_name': get_value(extracted_data.get('fund_name')),
                    'fund_code': get_value(extracted_data.get('fund_code')),
                    'management_company': get_value(extracted_data.get('management_company')),
                    'custodian_bank': get_value(extracted_data.get('custodian_bank')),
                    'management_fee': str(get_value(management_fee)) if management_fee is not None else None,
                    'subscription_fee': str(get_value(subscription_fee)) if subscription_fee is not None else None,
                    'redemption_fee': str(get_value(redemption_fee)) if redemption_fee is not None else None,
                    'switching_fee': str(get_value(switching_fee)) if switching_fee is not None else None,
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
