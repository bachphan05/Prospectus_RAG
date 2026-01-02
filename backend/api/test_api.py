import os
from mistralai import Mistral

# Khai báo Key
api_key = "GJlqoCHFbhHNrnw6J5KIOz1SNpaWHrwj" # Lưu ý: Nên dùng biến môi trường để bảo mật

try:
    # Khởi tạo client theo SDK mới nhất
    client = Mistral(api_key=api_key)
    
    # Gửi tin nhắn test
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "Hello, are you there?",
            },
        ]
    )
    
    print("✅ API key is working!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"❌ API key test failed. Error: {e}")
    print("Check: 1. API Key đúng chưa? 2. Đã add thẻ thanh toán (Billing) chưa?")