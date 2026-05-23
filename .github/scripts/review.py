import os
import sys
import anthropic

# Khởi tạo client với API Key từ môi trường
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Đọc file chứa nội dung code đã thay đổi (diff)
try:
    with open("pr_diff.txt", "r", encoding="utf-8") as f:
        diff_content = f.read()
except FileNotFoundError:
    print("Không tìm thấy file diff.")
    sys.exit(1)

# Bỏ qua nếu không có thay đổi nào
if not diff_content.strip():
    with open("review_result.txt", "w", encoding="utf-8") as f:
        f.write("Không có thay đổi mã nguồn nào đáng kể để review.")
    sys.exit(0)

# Định nghĩa "Intent" (Mục đích) cho AI
system_prompt = """Bạn là một kỹ sư phần mềm cấp cao chuyên đánh giá mã nguồn Python.
Nhiệm vụ của bạn là đánh giá đoạn mã bị thay đổi (Git Diff) trong Pull Request này.

Hãy tập trung vào:
1. Các lỗi logic tiềm ẩn, đặc biệt là trong việc xử lý dữ liệu (ví dụ: sử dụng Pandas, NumPy không tối ưu, hoặc lỗi logic trong các hàm tính toán).
2. Các điểm nghẽn về hiệu suất hoặc nguy cơ rò rỉ bộ nhớ.
3. Đưa ra các đoạn code (snippet) đề xuất để sửa lỗi hoặc tối ưu hóa trực tiếp.

Trình bày kết quả bằng tiếng Việt, định dạng Markdown rõ ràng, ngắn gọn và đi thẳng vào vấn đề.
"""

# Gửi yêu cầu tới mô hình Claude 3.5 Sonnet
try:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"Dưới đây là Git Diff:\n\n{diff_content}"}
        ]
    )
    
    # Lưu kết quả review ra một file text
    with open("review_result.txt", "w", encoding="utf-8") as f:
        f.write(response.content[0].text)
        
except Exception as e:
    with open("review_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Quá trình review bị lỗi: {str(e)}")
