import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI
import pickle
import hashlib


from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# -------------------------
# 1. PDF 읽기
# -------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n".join(texts)

# -------------------------
# 2. 텍스트 Chunking
# -------------------------
def chunk_text(text, max_tokens=200):
    """
    긴 텍스트를 max_tokens 단위로 나눔.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    current_chunk = []

    current_tokens = 0
    for word in words:
        tokens = len(enc.encode(word + " "))
        if current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = tokens
        else:
            current_chunk.append(word)
            current_tokens += tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------------------------
# 3. Embedding 생성
# -------------------------
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [d.embedding for d in response.data]
    return embeddings

# -------------------------
# 4. 검색
# -------------------------
class VectorStore:
    def __init__(self, embeddings, texts):
        self.texts = texts
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query_embedding, k=3):
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        results = [self.texts[i] for i in I[0]]
        return results

# -------------------------
# 5. GPT 답변 생성
# -------------------------
def generate_answer(question, retrieved_chunks, prompt: str = "당신은 친절한 PDF 문서 질문 답변 도우미입니다.", model = "gpt-3.5-turbo"):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
다음 정보를 참고하여 질문에 답하세요.

정보:
{context}

질문:
{question}
"""
    completion = client.chat.completions.create(
        model=model,
        # model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    
    return completion.choices[0].message.content.strip()

# -------------------------
# PDF 해시 생성
# -------------------------
def file_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# -------------------------
# 메인 실행
# -------------------------

def rag_question(pdf_path: str, question: str,prompt:str = "당신은 친절한 PDF 문서 질문 답변 도우미입니다.",model = "gpt-3.5-turbo", max_tokens: int = 200, k: int = 3):
    # 1. PDF 읽기
    hash_id = file_hash(pdf_path)
    cache_file = f"./datas/cache_{hash_id}.pkl"
    # 2. Chunking
    if os.path.exists(cache_file):
        print(f"📂 캐시 로드: {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        chunks = data["chunks"]
        chunk_embeddings = data["embeddings"]
    else:
        print("🚀 PDF 처리 중...")
        raw_text = read_pdf(pdf_path)
        chunks = chunk_text(raw_text, max_tokens)
        chunk_embeddings = embed_texts(chunks)
        with open(cache_file, "wb") as f:
            pickle.dump({"chunks": chunks, "embeddings": chunk_embeddings}, f)
        print(f"✅ 캐시 저장 완료: {cache_file}")
    
    # 4. Vector Store 초기화
    store = VectorStore(chunk_embeddings, chunks)
    # 5. 질문 처리
    query_embedding = embed_texts([question])[0]
    relevant_chunks = store.search(query_embedding, k)
    answer = generate_answer(question, relevant_chunks,prompt,model)
    return answer

if __name__ == "__main__":
    pdf_path = "./doc/경주 관광 인플루언서 마케팅 3개년 전략 로드맵.pdf"
    answer = rag_question(pdf_path, 
                            "로컬 인플루언서 미나의 1월 계획을 참고하여 세부 계획을 설계해줘"
                          "당신은 pdf파일 분석 전문가 입니다. 질문을 pdf에서 찾아 잘 정리 하여 알려주는 것에 특화되어 있습니다.")
    print(answer)