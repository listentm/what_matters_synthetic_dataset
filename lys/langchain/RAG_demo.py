import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
os.environ['HF_HUB_BASE_URL']="https://www.hf-mirror.com"
# 配置OpenAI API
openai.api_key = "sk-dUXHtWYlEgQOhnNC82opf5HhVrfQjHUzcTDRnOpwdDI01Jev"
openai.api_base = "https://api.keya.pw/v1"

# **Step 1: 数据准备**
documents = [
    "机器学习是一种通过数据训练模型的技术。",
    "深度学习是机器学习的一个子领域，利用神经网络。",
    "RAG结合了检索系统和生成模型。",
    "知识检索系统可以帮助模型生成更加精确的答案。",
]

# **Step 2: 嵌入模型**
# 使用Sentence-BERT生成文档嵌入
embedder = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedder.encode(documents, convert_to_tensor=True)

# **Step 3: 构建向量检索器**
# 使用FAISS存储嵌入向量
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings.cpu().detach().numpy())


# **Step 4: 编写查询函数**
def rag_query_with_gpt4(query, top_k=2):
    # 1. 将查询转为嵌入向量
    query_embedding = embedder.encode([query], convert_to_tensor=True)

    # 2. 从FAISS中检索与查询最相关的文档
    _, indices = index.search(query_embedding.cpu().detach().numpy(), top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]

    # 3. 将检索到的文档作为上下文，并调用GPT-4生成答案
    context = "\n".join(retrieved_docs)
    prompt = f"""
你是一个智能助手，能够回答用户的问题。以下是一些相关的背景知识：
{context}

根据以上内容，请回答以下问题：{query}
    """

    # 调用OpenAI GPT-4 API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一个智能助手，回答应该简明扼要。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    # 提取生成的答案
    answer = response['choices'][0]['message']['content']
    return answer, retrieved_docs


# **Step 5: 测试RAG系统**
query = "什么是RAG？"
answer, retrieved_docs = rag_query_with_gpt4(query)

print(f"问题: {query}")
print(f"检索到的文档: {retrieved_docs}")
print(f"生成的答案: {answer}")
