from ollama import chat

# 指定模型名稱
model = "phi4"

# 生成回應
def getResponse(input):
    response = chat(model=model, messages=[{"role": "user", "content": "請將下文翻譯成簡短的中文，一句一行"+input}])
    return response["message"]["content"]


# 輸出結果
if __name__ == "__main__":
    print(getResponse("hello how are you"))
