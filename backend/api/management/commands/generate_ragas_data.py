import json
import pandas as pd
from django.core.management.base import BaseCommand
from api.models import Document
from api.services import RAGService

class Command(BaseCommand):
    help = 'Generates evaluation dataset for RAGAS'

    def add_arguments(self, parser):
        parser.add_argument('document_id', type=int, help='ID of the document to test')

    def handle(self, *args, **options):
        doc_id = options['document_id']
        rag_service = RAGService()

        # 1. Define your Test Questions (Ground Truth is optional but recommended)
        test_cases = test_cases = [
    {
        "question": "Tên đầy đủ tiếng Việt và tiếng Anh của quỹ TCSME là gì?",
        "ground_truth": "Tên tiếng Việt: Quỹ Đầu tư Cổ phiếu Doanh nghiệp vừa và nhỏ Techcom. Tên tiếng Anh: Techcom Small and Medium Enterprise Equity Fund."
    },
    {
        "question": "Mục tiêu đầu tư chính của Quỹ TCSME được quy định như thế nào trong Bản cáo bạch?",
        "ground_truth": "Mục tiêu đầu tư là mang lại lợi nhuận dài hạn thông qua tăng trưởng vốn gốc và thu nhập trên cơ sở đánh giá, lựa chọn tài sản tốt, phân bổ danh mục hợp lý và tối thiểu hóa rủi ro."
    },
    {
        "question": "Chiến lược đầu tư của Quỹ tập trung vào nhóm cổ phiếu nào?",
        "ground_truth": "Chiến lược đầu tư chính là đầu tư năng động vào cổ phiếu của các Công ty hàng đầu trong rổ cổ phiếu VNMID và VNSML."
    },
    {
        "question": "Vốn điều lệ ban đầu và mệnh giá một chứng chỉ quỹ trong đợt phát hành lần đầu là bao nhiêu?",
        "ground_truth": "Vốn điều lệ huy động lần đầu là 50.000.000.000 VNĐ. Mệnh giá là 10.000 VNĐ/chứng chỉ quỹ."
    },
    {
        "question": "Mức phí dịch vụ quản lý quỹ (Management Fee) tối đa là bao nhiêu phần trăm một năm?",
        "ground_truth": "Giá dịch vụ Quản Lý tối đa là 1,2% NAV/năm."
    },
    {
        "question": "Biểu phí dịch vụ mua lại (Redemption Fee) áp dụng cho nhà đầu tư nắm giữ dưới 6 tháng và từ 12 tháng trở lên là bao nhiêu?",
        "ground_truth": "0 đến dưới 6 tháng: 1,00%; Từ trên 12 tháng trở lên: 0,00%."
    },
    {
        "question": "Phí dịch vụ chuyển đổi (Switching Fee) giữa các quỹ mở của TechcomCapital được tính như thế nào?",
        "ground_truth": "Giá dịch vụ Chuyển Đổi Quỹ tối đa 3%. Biểu phí hiện tại: 0 đến dưới 6 tháng: 1,00%; trên 12 tháng: 0,00%."
    },
    {
        "question": "Nhà đầu tư phải trả bao nhiêu phí cho một lần chuyển nhượng chứng chỉ quỹ (Transfer Fee)?",
        "ground_truth": "Giá dịch vụ Chuyển Nhượng là 300.000 đồng cho mỗi giao dịch."
    },
    {
        "question": "Phí dịch vụ giám sát và phí lưu ký mà Quỹ phải trả cho Ngân hàng giám sát là bao nhiêu?",
        "ground_truth": "Phí giám sát: 0,02% NAV/năm (tối thiểu 5.000.000 đồng/tháng). Phí lưu ký: 0,06% NAV/năm (tối thiểu 20.000.000 đồng/tháng)."
    },
    {
        "question": "Phí dịch vụ phát hành (Subscription Fee) hiện tại của Quỹ là bao nhiêu?",
        "ground_truth": "Mức giá dịch vụ phát hành của Quỹ là 0%."
    },
    {
        "question": "Phí dịch vụ quản trị quỹ (Fund Administration Fee) được tính theo tỷ lệ nào và mức tối thiểu là bao nhiêu?",
        "ground_truth": "Giá dịch vụ Quản trị quỹ là 0,03% NAV/năm, tối thiểu 15.000.000 đồng/tháng (chưa VAT)."
    },
    {
        "question": "Thời điểm đóng sổ lệnh (Cut-off time) đối với lệnh mua và lệnh bán chứng chỉ quỹ là khi nào?",
        "ground_truth": "Thời điểm đóng sổ lệnh là 14h45 ngày T-1 (trước ngày giao dịch)."
    },
    {
        "question": "Giá trị đặt lệnh mua tối thiểu (Minimum Subscription) đối với nhà đầu tư là bao nhiêu?",
        "ground_truth": "Mức đầu tư tối thiểu là 10.000 VNĐ."
    },
    {
        "question": "Số lượng chứng chỉ quỹ tối thiểu phải bán trong một lệnh bán (Minimum Redemption) là bao nhiêu?",
        "ground_truth": "Lệnh Bán tối thiểu là 10 (mười) Đơn Vị Quỹ."
    },
    {
        "question": "Trong trường hợp lệnh bán bị thực hiện một phần, nhà đầu tư cần làm gì nếu số lượng CCQ còn lại nhỏ hơn số lượng tối thiểu?",
        "ground_truth": "Nhà Đầu tư cần đặt bán toàn bộ để giảm số Đơn vị Quỹ nắm giữ về 0."
    },
    {
        "question": "Quỹ xác định Giá trị tài sản ròng (NAV) với tần suất như thế nào và công bố ở đâu?",
        "ground_truth": "NAV được xác định tại mỗi Ngày Giao Dịch (thứ Hai đến thứ Sáu) và công bố vào ngày làm việc tiếp theo."
    },
    {
        "question": "Thời gian thanh toán tiền bán chứng chỉ quỹ cho nhà đầu tư là trong vòng bao lâu?",
        "ground_truth": "Trong thời hạn 5 ngày làm việc sau ngày giao dịch Chứng chỉ quỹ."
    },
    {
        "question": "Quỹ TCSME không được đầu tư quá bao nhiêu phần trăm tổng giá trị tài sản vào chứng khoán của một tổ chức phát hành?",
        "ground_truth": "Không được đầu tư quá 10% tổng giá trị chứng khoán đang lưu hành của tổ chức đó (trừ công cụ nợ Chính phủ) và không quá 20% tổng tài sản quỹ vào một tổ chức."
    },
    {
        "question": "Tổng giá trị các hạng mục đầu tư lớn (chiếm từ 5% tài sản quỹ trở lên) không được vượt quá tỷ lệ nào?",
        "ground_truth": "Không được vượt quá 40% tổng giá trị tài sản của quỹ."
    },
    {
        "question": "Quỹ có được phép đầu tư vào chứng chỉ quỹ của chính mình hoặc đầu tư trực tiếp vào bất động sản không?",
        "ground_truth": "Không được đầu tư vào chứng chỉ quỹ của chính quỹ đó. Không được đầu tư trực tiếp vào bất động sản, đá quý, kim loại quý hiếm."
    },
    {
        "question": "Giới hạn đầu tư vào nhóm công ty có quan hệ sở hữu (công ty mẹ, công ty con) là bao nhiêu phần trăm tổng giá trị tài sản quỹ?",
        "ground_truth": "Không được đầu tư quá 30% tổng giá trị tài sản của quỹ vào các công ty trong cùng một nhóm công ty có quan hệ sở hữu."
    },
    {
        "question": "Ngân hàng giám sát của Quỹ TCSME là ngân hàng nào và chi nhánh nào?",
        "ground_truth": "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam (BIDV) - Chi nhánh Hà Thành."
    },
    {
        "question": "Đại lý phân phối chứng chỉ quỹ (Distributor) và Đại lý chuyển nhượng (Transfer Agent) là những tổ chức nào?",
        "ground_truth": "Đại lý phân phối: Công ty CP Chứng khoán Kỹ thương (TCBS). Đại lý chuyển nhượng: Trung tâm Lưu ký Chứng khoán Việt Nam (VSD)."
    },
    {
        "question": "Trong trường hợp nào việc thực hiện lệnh bán của nhà đầu tư có thể bị thực hiện một phần (prorated)?",
        "ground_truth": "Khi tổng giá trị lệnh bán ròng > 10% NAV hoặc việc thực hiện lệnh làm NAV Quỹ < 50 tỷ đồng."
    },
    {
        "question": "Chương trình Đầu tư Định kỳ (SIP) sẽ tự động chấm dứt trong trường hợp nào?",
        "ground_truth": "Khi Nhà Đầu Tư thông báo dừng hoặc không nộp tiền/không nộp đủ tiền mua trong 05 kỳ liên tiếp."
    },
    {
        "question": "Rủi ro tái đầu tư (Reinvestment risk) được mô tả như thế nào trong Bản cáo bạch?",
        "ground_truth": "Là rủi ro khi lãi suất thị trường giảm, tiền lãi hoặc gốc nhận được phải tái đầu tư với mức sinh lợi thấp hơn."
    },
    {
        "question": "Nhà đầu tư nước ngoài cần thực hiện giao dịch đầu tư qua loại tài khoản vốn nào?",
        "ground_truth": "Nhà đầu tư nước ngoài phải thực hiện qua Tài khoản vốn đầu tư gián tiếp (IICA) tại một ngân hàng thương mại ở Việt Nam."
    },
    {
        "question": "Ai là những người chịu trách nhiệm chính về nội dung Bản cáo bạch từ phía Công ty quản lý quỹ?",
        "ground_truth": "Bà Nguyễn Thị Thu Hiền (Chủ tịch HĐQT), Ông Đặng Lưu Dũng (Tổng Giám đốc), Bà Phan Thị Thu Hằng (Kế toán trưởng)."
    },
    {
        "question": "Ban đại diện Quỹ bao gồm những thành viên nào?",
        "ground_truth": "Ông Nhâm Hà Hải, Ông Đào Kiên Trung, Ông Trần Viết Thỏa."
    },
    {
        "question": "Nhà đầu tư có những lựa chọn nào về việc nhận phân phối lợi nhuận (cổ tức)?",
        "ground_truth": "Lựa chọn Nhận Cổ Tức Bằng Tiền (DPP) hoặc Lựa chọn Tái Đầu tư Cổ tức (DRIP)."
    },
    {
        "question": "Nếu nhà đầu tư không chọn phương thức nhận cổ tức cụ thể, Quỹ sẽ áp dụng phương thức mặc định nào?",
        "ground_truth": "Lựa chọn Tái Đầu tư Cổ tức (DRIP) sẽ được tự động áp dụng."
    },
    {
        "question": "Công ty quản lý quỹ có được phép sử dụng vốn của Quỹ để cho vay không?",
        "ground_truth": "Không. Công ty Quản Lý Quỹ không được sử dụng vốn và tài sản của Quỹ để cho vay hoặc bảo lãnh."
    }
]

        results = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        print(f"--- Generating RAGAS Data for Doc ID {doc_id} ---")

        for case in test_cases:
            q = case["question"]
            print(f"Processing: {q}")

            # Call the service with return_sources=True
            response_data = rag_service.chat(doc_id, q, return_sources=True)

            # Append to lists
            results["question"].append(q)
            results["answer"].append(response_data["text"])
            results["contexts"].append(response_data["contexts"]) # List of strings
            results["ground_truth"].append(case["ground_truth"])

        # Convert to Pandas DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        output_file = "ragas_dataset.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Success! Dataset saved to {output_file}")
        print("You can now load this CSV in your evaluation.py script.")