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
        test_cases = [
    # --- NHÓM 1: KIỂM THỬ KEYWORD (TÌM CON SỐ & MÃ ĐỊNH DANH) ---
    {
        "question": "Giấy chứng nhận đăng ký chào bán số 250/GCN-UBCK được cấp vào ngày nào?",
        "ground_truth": "Giấy chứng nhận đăng ký chào bán chứng chỉ quỹ số 250/GCN-UBCK do Chủ tịch UBCKNN cấp ngày 23/8/2022."
    },
    {
        "question": "Địa chỉ văn phòng của TechcomCapital tại Tầng 28, Tòa C5, số 119 Trần Duy Hưng là ở quận nào?",
        "ground_truth": "Địa chỉ tại Tầng 28, Tòa C5, số 119 Trần Duy Hưng, phường Trung Hòa, Quận Cầu Giấy, Hà Nội."
    },
    {
        "question": "Thông tin về Ngân hàng lưu ký có số giấy chứng nhận đăng ký hoạt động là 510/QĐ-ĐKHĐLK như thế nào?",
        "ground_truth": "Là Ngân hàng TMCP Đầu tư và Phát triển Việt Nam - Chi nhánh Hà Thành, có Giấy chứng nhận đăng ký hoạt động lưu ký chứng khoán số 510/QĐ-ĐKHĐLK ngày 01/08/2006 do UBCKNN cấp."
    },
    {
        "question": "Mức phí quản lý quỹ là 2,0%/năm áp dụng cho đối tượng nào?",
        "ground_truth": "Mức phí 2,0% NAV/năm là mức giá dịch vụ Quản Lý tối đa được quy định trong Bản cáo bạch (mức hiện tại áp dụng có thể thấp hơn tùy thông báo)."
    },
    {
        "question": "Số điện thoại liên hệ của Công ty Cổ phần Quản lý quỹ Kỹ thương (TechcomCapital) là số nào?",
        "ground_truth": "Số điện thoại: (84-24) 39446368."
    },
    {
        "question": "Mã số thuế hoặc số đăng ký kinh doanh của TechcomCapital được ghi nhận như thế nào?",
        "ground_truth": "Giấy phép thành lập và hoạt động số 10/GP-UBCK ngày 03/07/2006 do Chủ tịch Ủy ban Chứng khoán Nhà nước cấp."
    },

    # --- NHÓM 2: KIỂM THỬ SEMANTIC (TÌM THEO Ý NGHĨA/NGỮ CẢNH) ---
    {
        "question": "Nếu thị trường chứng khoán sụt giảm mạnh, quỹ có những biện pháp gì để bảo vệ tài sản của tôi?",
        "ground_truth": "Quỹ thực hiện quản trị rủi ro bằng cách đa dạng hóa danh mục đầu tư, tuân thủ các hạn mức đầu tư an toàn và thực hiện các biện pháp phòng ngừa rủi ro thị trường theo quy định pháp luật."
    },
    {
        "question": "Tôi có thể rút tiền ra khỏi quỹ bằng cách nào và mất bao lâu?",
        "ground_truth": "Nhà đầu tư thực hiện bằng cách đặt Lệnh Bán chứng chỉ quỹ. Tiền sẽ được thanh toán trong vòng 05 ngày làm việc kể từ Ngày giao dịch chứng chỉ quỹ."
    },
    {
        "question": "Quỹ này ưu tiên đầu tư vào các loại hình doanh nghiệp như thế nào?",
        "ground_truth": "Quỹ tập trung đầu tư năng động vào cổ phiếu của các Công ty hàng đầu trong rổ cổ phiếu VNMID (vốn hóa vừa) và VNSML (vốn hóa nhỏ) trên thị trường chứng khoán Việt Nam."
    },
    {
        "question": "Ai là người chịu trách nhiệm giám sát việc chi tiêu và sử dụng tiền của quỹ?",
        "ground_truth": "Ngân hàng giám sát (BIDV - Chi nhánh Hà Thành) có trách nhiệm giám sát hoạt động của Công ty quản lý quỹ nhằm đảm bảo tài sản của quỹ được quản lý đúng quy định."
    },
    {
        "question": "Làm thế nào để tôi biết được giá trị tài sản của mình mỗi ngày?",
        "ground_truth": "Nhà đầu tư có thể theo dõi Giá trị tài sản ròng (NAV) được công bố định kỳ tại mỗi Ngày Giao Dịch trên website của Công ty quản lý quỹ và các Đại lý phân phối."
    },

    # --- NHÓM 3: KIỂM THỬ HYBRID (SỨC MẠNH TỔNG HỢP) ---
    {
        "question": "Quy định về thời gian Cut-off time để thực hiện lệnh giao dịch trong ngày là khi nào?",
        "ground_truth": "Thời điểm đóng sổ lệnh (Cut-off time) là 14h45 ngày T-1 (trước Ngày giao dịch)."
    },
    {
        "question": "Các rủi ro liên quan đến biến động lãi suất ảnh hưởng như thế nào đến lợi nhuận của Quỹ TCSME?",
        "ground_truth": "Rủi ro lãi suất ảnh hưởng đến giá trị các công cụ nợ và tiền gửi. Khi lãi suất thị trường tăng, giá trị các công cụ nợ thường giảm, làm ảnh hưởng đến NAV của quỹ."
    },
    {
        "question": "Hạn mức đầu tư vào cổ phiếu chưa niêm yết của quỹ được quy định tối đa là bao nhiêu phần trăm?",
        "ground_truth": "Quỹ không được đầu tư quá 20% tổng giá trị tài sản ròng vào các chứng khoán chưa niêm yết, trừ công cụ nợ Chính phủ."
    },
    {
        "question": "Trong trường hợp xảy ra tranh chấp, cơ quan nào sẽ đứng ra giải quyết theo quy định của Bản cáo bạch?",
        "ground_truth": "Tranh chấp sẽ được ưu tiên giải quyết thông qua thương lượng, hòa giải. Nếu không thành, tranh chấp sẽ được đưa ra giải quyết tại Tòa án có thẩm quyền tại Việt Nam."
    },
    {
        "question": "TCSME có được phép vay nợ để đầu tư không và hạn mức vay là bao nhiêu?",
        "ground_truth": "Quỹ chỉ được vay ngắn hạn để trang trải các chi phí cần thiết hoặc thanh toán lệnh mua lại CCQ, hạn mức không quá 5% giá trị tài sản ròng và thời gian vay tối đa 30 ngày."
    },
    {
        "question": "Điều kiện để trở thành thành viên của Ban đại diện Quỹ là gì?",
        "ground_truth": "Thành viên Ban đại diện Quỹ phải đáp ứng các tiêu chuẩn về đạo đức, chuyên môn tài chính/kế toán và không thuộc các trường hợp bị cấm theo quy định của Luật Chứng khoán và Điều lệ Quỹ."
    },
    {
        "question": "Phí chuyển đổi từ quỹ TCSME sang các quỹ khác của TechcomCapital được tính như thế nào?",
        "ground_truth": "Mức phí chuyển đổi tối đa là 3% giá trị giao dịch. Biểu phí hiện tại được phân loại theo thời gian nắm giữ, miễn phí nếu nắm giữ trên 12 tháng."
    },
    {
        "question": "Giải thích quy trình xử lý nếu xảy ra sai sót trong việc định giá NAV vượt mức sai số cho phép?",
        "ground_truth": "Công ty quản lý quỹ phải bồi thường thiệt hại cho nhà đầu tư và Quỹ nếu sai sót vượt mức 0,75% NAV (đối với quỹ cổ phiếu) theo quy định của pháp luật."
    },
    {
        "question": "Ngân hàng BIDV - Chi nhánh Hà Thành thực hiện những nhiệm vụ lưu ký cụ thể nào cho quỹ?",
        "ground_truth": "Thực hiện bảo quản, lưu ký chứng khoán, tài liệu quyền sở hữu tài sản, thực hiện thu các khoản lãi/cổ tức và giám sát các hoạt động đầu tư của Công ty quản lý quỹ."
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