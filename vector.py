from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Đọc file văn bản luật
with open("ANQG.txt", 'r', encoding='utf-8') as file:
    law_text = file.read()

# Xử lý văn bản thành DataFrame
lines = law_text.split('\n')
chapters = []
articles = []
contents = []

current_chapter = ""
current_article = ""
current_content = []

for line in lines:
    line = line.strip()
    if not line:
        continue
        
    if line.startswith('Chương'):
        current_chapter = line
    elif line.startswith('Điều'):
        if current_article and current_content:
            chapters.append(current_chapter)
            articles.append(current_article)
            contents.append('\n'.join(current_content))
            current_content = []
        current_article = line
    else:
        current_content.append(line)

if current_article and current_content:
    chapters.append(current_chapter)
    articles.append(current_article)
    contents.append('\n'.join(current_content))

df = pd.DataFrame({
    'Chapter': chapters,
    'Article': articles,
    'Content': contents
})

# Sử dụng model embedding phù hợp với tiếng Việt
embeddings = OllamaEmbeddings(model="bge-m3:latest")

# Đổi tên database
db_location = "./chrome_law_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    # Xử lý văn bản luật thành các chunk
    for i, (chapter, article, content) in enumerate(zip(df['Chapter'], df['Article'], df['Content'])):
        # Tạo nội dung có cấu trúc
        page_content = f"""
        {chapter}
        {article}
        {content}
        """.strip()
        
        document = Document(
            page_content=page_content,
            metadata={
                "chapter": chapter,
                "article": article,
                "article_number": i + 1,  # Số thứ tự điều luật
                "type": "law_document"    # Loại văn bản
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="law_articles",  # Đổi tên collection
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Lấy 5 điều luật liên quan nhất
)
