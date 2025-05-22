from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
# Chỉnh lại model
model = OllamaLLM(model="hf.co/uonlp/Vistral-7B-Chat-gguf:Q4_0")
# Prompt template (Chỉnh lại )
template = """
Bạn là một chuyên gia về luật An ninh quốc gia của Việt Nam.

Đây là các điều luật liên quan: {legal_articles}

Câu hỏi cần trả lời: {question}

Hãy trả lời dựa trên các điều luật được cung cấp ở trên. Nếu câu trả lời không thể tìm thấy trong các điều luật được cung cấp, hãy nói rõ điều đó. Trích dẫn cụ thể điều luật khi trả lời.
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    # Lấy kết quả từ retriever
    legal_articles = retriever.invoke(question)
    
    # Invoke chain với đúng tên biến
    result = chain.invoke({
        "legal_articles": legal_articles, 
        "question": question
    })
    
    print(result)