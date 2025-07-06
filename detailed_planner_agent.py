import sys
import io
import os
import json
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 전역 설정 ---
GENERATED_PLANS_DIR = "generated_plans"
FAISS_INDEX_PATH = "faiss_index"

# --- 폴더 생성 ---
if not os.path.exists(GENERATED_PLANS_DIR):
    os.makedirs(GENERATED_PLANS_DIR)

# --- 모델 설정 (비용 최적화) ---
QUESTION_LLM_MODEL = "gpt-3.5-turbo"
FINAL_PLAN_LLM_MODEL = "gpt-4-turbo"
EVALUATION_LLM_MODEL = "gpt-3.5-turbo" # 평가용 LLM 모델

question_llm = ChatOpenAI(model_name=QUESTION_LLM_MODEL, temperature=0.1)
final_plan_llm = ChatOpenAI(model_name=FINAL_PLAN_LLM_MODEL, temperature=0.3)
evaluation_llm = ChatOpenAI(model_name=EVALUATION_LLM_MODEL, temperature=0.0) # 평가는 보수적으로

# --- FAISS Retriever 초기화 ---
embeddings = OpenAIEmbeddings()
retriever_tool = None # 전역 변수로 선언

def initialize_faiss_retriever():
    global retriever_tool
    if os.path.exists(FAISS_INDEX_PATH):
        print(">> 저장된 FAISS 인덱스를 불러옵니다...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print(">> FAISS 인덱스가 없어 새로 생성합니다...")
        loader = PyPDFLoader("G:\Hackerton\RAG\doc\경주 관광 인플루언서 마케팅 3개년 전략 로드맵.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print(">> 새로운 인덱스를 '{FAISS_INDEX_PATH}'에 저장합니다.")
        vectorstore.save_local(FAISS_INDEX_PATH)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "gyeongju_tourism_document_search",
        "경주 관광 인플루언서 마케팅 3개년 전략 로드맵 문서에서 정보를 검색합니다."
    )

# --- 헬퍼 함수 ---
def _extract_json_from_response(response_text):
    """LLM 응답에서 JSON 블록을 추출합니다."""
    # 정규 표현식을 사용하여 ```json ... ``` 블록 찾기
    match = re.search(r"```json\n(.*)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # ```json이 없는 경우, 가장 큰 JSON 객체/배열 찾기
    match = re.search(r"\{.*\}|\[.*\]", response_text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None

def _generate_question(year, month):
    """최적의 질문을 생성하는 헬퍼 함수 (저비용 모델 사용)"""
    print(f"\n>> {year}년 {month}월 계획 수립을 위한 최적의 질문을 생성합니다 (Model: {QUESTION_LLM_MODEL})...")
    template = """
    당신은 전문 마케팅 전략가입니다.
    주어진 연도와 월을 바탕으로, 매우 창의적이고 구체적인 관광 마케팅 실행 계획을 수립하기 위한 **상세한 프롬프트(질문)**를 생성해야 합니다.
    아래 항목들이 반드시 포함된, 실제 담당자가 감탄할 만한 수준의 상세하고 매력적인 액션 플랜을 요청하는 프롬프트를 한글로 작성해주세요:
    - 캠페인 명 및 슬로건
    - 페르소나 기반 인플루언서 섭외 전략
    - 주차별 디테일드 콘텐츠 캘린더
    - 플랫폼별 킬러 콘텐츠 기획
    - 창의적인 온/오프라인 연계 이벤트
    - 구체적인 정량/정성 KPI
    또한, 해당 월의 날씨, 사회적 트렌드, 특별한 이벤트(예: 추석) 등을 웹에서 검색하여 반영하라는 요구사항을 포함시켜야 합니다.
    연도: {year}, 월: {month}
    결과물 (프롬프트 텍스트만 출력):
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | question_llm | StrOutputParser()
    return chain.invoke({"year": year, "month": month})

def _get_monthly_events(year, month):
    """해당 월의 주요 공휴일, 축제, 특별한 이벤트를 LLM을 통해 파악하는 헬퍼 함수"""
    print(f"\n>> {year}년 {month}월의 주요 이벤트를 파악합니다...")
    event_prompt = PromptTemplate.from_template(
        """
        {year}년 {month}월 대한민국에서 예상되는 주요 공휴일, 축제, 사회적 이벤트, 계절적 특징(예: 단풍, 장마) 등을 콤마로 구분하여 나열해주세요.
        예시: 설날, 신년 해돋이, 벚꽃 축제, 장마, 추석, 단풍 시작, 크리스마스
        """
    )
    chain = event_prompt | question_llm | StrOutputParser()
    events = chain.invoke({"year": year, "month": month})
    print(f">> 파악된 이벤트: {events}")
    return events

def _get_context(year, month):
    """계획 수립에 필요한 컨텍스트를 수집하는 헬퍼 함수"""
    print(f"\n>> {year}년 {month}월 계획 수립에 필요한 컨텍스트를 수집합니다...")
    doc_context = retriever_tool.invoke({"query": f"{month}월 경주 관광 계획"})
    monthly_events = _get_monthly_events(year, month)
    search_tool = TavilySearchResults(description="최신 트렌드, 이벤트, 날씨 등 일반적인 정보를 웹에서 검색합니다.")
    web_context = search_tool.invoke({"query": f"{year}년 {month}월 한국 날씨, 사회 트렌드, {monthly_events}"})
    print(">> 컨텍스트 수집 완료.")
    return f"--- 문서 컨텍스트 ---\n{doc_context}\n\n--- 웹 검색 컨텍스트 ---\n{web_context}"

def _generate_final_plan(request, context):
    """최종 계획을 생성하는 헬퍼 함수 (고성능 모델 사용)"""
    print(f"\n>> 수집된 컨텍스트와 요청사항을 바탕으로 최종 계획안을 생성합니다 (Model: {FINAL_PLAN_LLM_MODEL})...")
    template = """
    당신은 대한민국 최고의 관광 마케팅 전략가입니다.
    아래의 **컨텍스트**와 **요청사항**을 종합하여, 최종 결과물로 완벽하고 창의적인 실행 계획 보고서를 작성해야 합니다.
    요약하지 말고, 요청사항의 모든 항목이 포함된, 담당자가 감탄할 만한 수준의 완전하고 상세한 계획을 한글로 작성해주세요.

    --- 컨텍스트 ---
    {context}
    --- 컨텍스트 끝 ---

    --- 요청사항 ---
    {request}
    --- 요청사항 끝 ---

    **최종 실행 계획 보고서:**
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | final_plan_llm | StrOutputParser()
    return chain.invoke({"request": request, "context": context})

def _evaluate_monthly_plan(generated_plan_content, original_request_question):
    """생성된 월간 계획을 평가하고 결과를 JSON으로 반환합니다."""
    print(f"\n>> 생성된 월간 계획을 평가합니다 (Model: {EVALUATION_LLM_MODEL})...")
    evaluation_prompt = PromptTemplate.from_template(
        """
        당신은 마케팅 계획 평가 전문가입니다.
        아래의 **원래 요청 질문**과 **생성된 월간 계획**을 바탕으로, 계획의 품질을 평가해주세요.
        
        **평가 기준:**
        1.  **요구사항 준수**: 원래 요청 질문에 명시된 모든 항목(캠페인 명, 인플루언서 섭외, 주차별 캘린더, 킬러 콘텐츠, 이벤트, KPI)이 포함되었는가? (점수: 0-5점)
        2.  **구체성 및 상세함**: 계획이 실제 실행 가능할 정도로 구체적이고 상세한가? (점수: 0-5점)
        3.  **창의성**: 독창적이고 매력적인 아이디어가 포함되어 있는가? (점수: 0-5점)
        4.  **형식 준수**: Markdown 형식 및 가독성이 좋은가? (점수: 0-5점)
        5.  **언어**: 한글로 명확하게 작성되었는가? (점수: 0-5점)

        **평가 결과는 다음 JSON 형식으로만 출력해주세요. 다른 설명은 절대 추가하지 마세요:**
        ```json
        {{
            "overall_score": <총점 (25점 만점)>,
            "feedback": "<상세 피드백>",
            "pass": <true/false (총점 15점 이상이면 true)>
        }}
        ```

        --- 원래 요청 질문 ---
        {original_request_question}
        --- 원래 요청 질문 끝 ---

        --- 생성된 월간 계획 ---
        {generated_plan_content}
        --- 생성된 월간 계획 끝 ---
        """
    )
    chain = evaluation_prompt | evaluation_llm | StrOutputParser()
    evaluation_result_str = chain.invoke({
        "generated_plan_content": generated_plan_content,
        "original_request_question": original_request_question
    })
    
    json_str = _extract_json_from_response(evaluation_result_str)
    
    if json_str:
        try:
            evaluation_result = json.loads(json_str)
            print(f"\n>> 월간 계획 평가 결과: 총점 {evaluation_result.get('overall_score', 'N/A')}점, 합격 여부: {evaluation_result.get('pass', 'N/A')}")
            print(f"   상세 피드백: {evaluation_result.get('feedback', 'N/A')}")
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"WARNING: 월간 계획 평가 결과 파싱 오류: {e}")
            print(f"LLM 출력 (파싱 시도): {json_str}")
    else:
        print("WARNING: 평가 결과에서 JSON을 찾을 수 없습니다.")
        print(f"LLM 전체 출력: {evaluation_result_str}")

    return {
        "overall_score": 0,
        "feedback": "평가 결과 파싱 실패 또는 JSON 없음",
        "pass": False,
        "raw_output": evaluation_result_str
    }

# --- 메인 월간 계획 생성 함수 ---
def generate_and_save_monthly_plan(year, month,dir_path):
    """월간 계획을 생성하고, 콘텐츠와 평가 결과를 반환합니다."""
    print(f"\n>> {year}년 {month}월 월간 계획 생성 프로세스를 시작합니다...")
    question = _generate_question(year, month)
    question_filename = os.path.join(dir_path, f"generated_question_{year}_{month:02d}.md")
    with open(question_filename, "w", encoding="utf-8") as f:
        f.write(f"# {year}년 {month}월 마케팅 계획 수립을 위해 생성된 질문\n\n{question}")
    print(f">> 생성된 질문을 '{question_filename}' 파일로 저장했습니다.")

    context = _get_context(year, month)
    final_plan_content = _generate_final_plan(question, context)

    monthly_plan_filename = os.path.join(dir_path, f"detailed_plan_{year}_{month:02d}_final.md")
    with open(monthly_plan_filename, "w", encoding="utf-8") as f:
        f.write(f"# {year}년 {month}월 경주 관광 인플루언서 마케팅 세부 실행 계획\n\n{final_plan_content}")
    print(f">> {year}년 {month}월 월간 계획이 '{monthly_plan_filename}' 파일로 저장되었습니다.")

    # 월간 계획 평가
    evaluation_result = _evaluate_monthly_plan(final_plan_content, question)
    
    # 평가 결과를 별도의 JSON 파일로 저장
    evaluation_filename = os.path.join(dir_path, f"evaluation_monthly_{year}_{month:02d}.json")
    with open(evaluation_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=4)
    print(f">> 월간 계획 평가 결과를 '{evaluation_filename}' 파일에 저장했습니다.")
    
    return final_plan_content, evaluation_result

if __name__ == '__main__':
    # 이 스크립트를 직접 실행할 때만 작동
    PLAN_YEAR = 2025
    PLAN_MONTH = 9
    initialize_faiss_retriever()
    # 반환값을 확인하기 위해 변수에 저장
    plan, evaluation = generate_and_save_monthly_plan(PLAN_YEAR, PLAN_MONTH)
    print("\n--- 최종 반환 값 ---")
    print("Plan Content:", plan[:200] + "...") # 일부만 출력
    print("Evaluation Result:", evaluation)
