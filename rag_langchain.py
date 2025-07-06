import sys
import io

# stdout의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수 로드
load_dotenv()

# PDF 파일 로드
loader = PyPDFLoader("G:\\Hackerton\\RAG\\doc\\경주 관광 인플루언서 마케팅 3개년 전략 로드맵.pdf")
docs = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 임베딩 및 벡터 저장소 생성
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 검색기 생성
retriever = vectorstore.as_retriever()

# 프롬프트 템플릿 생성
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM 생성
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# RAG 체인 생성
rag_chain = (
    ({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == '__main__':
    # 질문 및 답변
    question = "경주 관광 인플루언서 마케팅의 3개년 목표는 무엇인가요?"
    answer = rag_chain.invoke(question)
    print(answer)