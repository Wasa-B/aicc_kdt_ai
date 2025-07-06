

import sys
import io
import os
import json
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import get_week_number # get_week_number 함수 임포트

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 전역 설정 ---
WEEKLY_PLANS_DIR = "weekly_content_plans"
GENERATED_PLANS_DIR = "generated_plans" # 월간 계획 파일 경로를 위해 추가

# --- 폴더 생성 ---
if not os.path.exists(WEEKLY_PLANS_DIR):
    os.makedirs(WEEKLY_PLANS_DIR)

# --- LLM 설정 ---
WEEKLY_LLM_MODEL = "gpt-4-turbo"
EVALUATION_LLM_MODEL = "gpt-3.5-turbo" # 평가용 LLM 모델

weekly_llm = ChatOpenAI(model_name=WEEKLY_LLM_MODEL, temperature=0.4) # 창의적인 콘텐츠 아이디어를 위해 온도 조절
evaluation_llm = ChatOpenAI(model_name=EVALUATION_LLM_MODEL, temperature=0.0) # 평가는 보수적으로

def _extract_json_from_response(response_text):
    """LLM 응답에서 JSON 블록을 추출합니다."""
    match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{.*\}|\[.*\]", response_text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None

def _generate_weekly_plan(master_plan_content, week_number, month):
    """주간 상세 콘텐츠 계획을 생성하는 헬퍼 함수"""
    print(f"\n>> {month}월 {week_number}주차 상세 콘텐츠 계획을 생성합니다 (Model: {WEEKLY_LLM_MODEL})...")
    template = """
    당신은 가상 인플루언서의 유능한 콘텐츠 매니저입니다.
    아래의 **월간 마스터 플랜**을 바탕으로, **{week_number}주차**에 실행할 **일일 콘텐츠 계획**을 구체적으로 작성해야 합니다.

    **요구사항:**
    1.  결과는 **Markdown 테이블** 형식으로 작성해주세요.
    2.  테이블 컬럼은 `요일`, `날짜 ({month}월)`, `플랫폼`, `콘텐츠 상세 아이디어`, `핵심 메시지/해시태그`로 구성해주세요.
    3.  콘텐츠 아이디어는 매우 구체적이고 실행 가능해야 합니다. (예: '인스타 사진 올리기' (X) -> '첨성대와 핑크뮬리를 배경으로 한복을 입고 찍은 감성 사진 5장 캐러셀 포스팅' (O))
    4.  Vlog, 블로그 포스트, 인스타그램(사진, 릴스, 스토리), 유튜브(쇼츠, 일반 영상), 뉴스레터, X(구 트위터) 등 다양한 플랫폼을 활용해주세요.
    5.  **하루에 여러 개의 콘텐츠를 기획할 수 있으며, 이 경우 동일한 요일과 날짜에 대해 여러 행을 사용하여 각 콘텐츠를 구분해주세요.**
    6.  **매일 최소 한 개 이상의 콘텐츠를 반드시 포함해주세요.**

    **예시 (하루에 여러 콘텐츠가 있는 경우):**
    ```markdown
    | 요일   | 날짜 (9월) | 플랫폼       | 콘텐츠 상세 아이디어                                               | 핵심 메시지/해시태그                 |
    |--------|------------|--------------|-------------------------------------------------------------------|--------------------------------------|
    | 월요일 | 1일        | 인스타그램   | 첨성대와 핑크뮬리를 배경으로 한복을 입고 찍은 감성 사진 5장 캐러셀 포스팅 | #가을의경주 #첨성대핑크뮬리          |
    | 월요일 | 1일        | 유튜브 쇼츠  | 경주 황리단길 맛집 탐방 쇼츠: 짧은 클립으로 소개하는 로컬 음식       | #경주맛집 #황리단길 #가을여행         |
    | 화요일 | 2일        | 유튜브       | 경주 불국사 가을 풍경 Vlog: 역사적 배경 설명과 함께 가을 단풍 촬영 | #가을의마음 #불국사 #경주여행         |
    ```

    --- 월간 마스터 플랜 ---
    {master_plan}
    --- 월간 마스터 플랜 끝 ---

    **{week_number}주차 일일 콘텐츠 계획 (Markdown 테이블):**
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | weekly_llm | StrOutputParser()
    return chain.invoke({"master_plan": master_plan_content, "week_number": week_number, "month": month})

def _evaluate_weekly_plan(generated_weekly_plan_content, master_plan_content, week_number):
    """생성된 주간 계획을 평가하고 결과를 JSON으로 반환합니다."""
    print(f"\n>> 생성된 {week_number}주차 주간 계획을 평가합니다 (Model: {EVALUATION_LLM_MODEL})...")
    evaluation_prompt = PromptTemplate.from_template(
        """
        당신은 콘텐츠 계획 평가 전문가입니다.
        아래의 **월간 마스터 플랜**과 **생성된 주간 계획**을 바탕으로, 주간 계획의 품질을 평가해주세요.
        
        **평가 기준:**
        1.  **월간 계획과의 일관성**: 주간 계획이 월간 마스터 플랜의 테마와 목표에 부합하는가? (점수: 0-5점)
        2.  **구체성 및 상세함**: 각 콘텐츠 아이디어가 실행 가능할 정도로 구체적이고 상세한가? (점수: 0-5점)
        3.  **다양성**: 다양한 플랫폼과 콘텐츠 형식이 적절히 활용되었는가? (점수: 0-5점)
        4.  **요구사항 준수**: '하루에 여러 콘텐츠 가능', '매일 최소 1개 콘텐츠 포함' 등 모든 요구사항을 준수했는가? (점수: 0-5점)
        5.  **형식 준수**: Markdown 테이블 형식 및 가독성이 좋은가? (점수: 0-5점)

        **평가 결과는 다음 JSON 형식으로만 출력해주세요. 다른 설명은 절대 추가하지 마세요:**
        ```json
        {{
            "overall_score": <총점 (25점 만점)>,
            "feedback": "<상세 피드백>",
            "pass": <true/false (총점 18점 이상이면 true)>
        }}
        ```

        --- 월간 마스터 플랜 ---
        {master_plan}
        --- 월간 마스터 플랜 끝 ---

        --- 생성된 주간 계획 ---
        {generated_weekly_plan}
        --- 생성된 주간 계획 끝 ---
        """
    )
    chain = evaluation_prompt | evaluation_llm | StrOutputParser()
    evaluation_result_str = chain.invoke({"generated_weekly_plan": generated_weekly_plan_content, "master_plan": master_plan_content})
    
    json_str = _extract_json_from_response(evaluation_result_str)

    if json_str:
        try:
            evaluation_result = json.loads(json_str)
            print(f"\n>> 주간 계획 평가 결과: 총점 {evaluation_result.get('overall_score', 'N/A')}점, 합격 여부: {evaluation_result.get('pass', 'N/A')}")
            print(f"   상세 피드백: {evaluation_result.get('feedback', 'N/A')}")
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"WARNING: 주간 계획 평가 결과 파싱 오류: {e}")
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

# --- 메인 주간 계획 생성 함수 ---
def generate_and_save_weekly_plan(master_plan_content, week_number, year, month,dir_path):
    """주간 계획을 생성하고, 콘텐츠와 평가 결과를 반환합니다."""
    print(f"\n>> {year}년 {month}월 {week_number}주차 주간 계획 생성 프로세스를 시작합니다...")
    weekly_plan_content = _generate_weekly_plan(master_plan_content, week_number, month)
    
    weekly_plan_filename = os.path.join(dir_path, f"weekly_plan_{year}_{month:02d}_week{week_number}.md")
    with open(weekly_plan_filename, "w", encoding="utf-8") as f:
        f.write(f"# {year}년 {month}월 {week_number}주차 상세 콘텐츠 실행 계획\n\n")
        f.write(weekly_plan_content)
    print(f">> {week_number}주차 계획이 '{weekly_plan_filename}' 파일로 저장되었습니다.")

    # 주간 계획 평가
    evaluation_result = _evaluate_weekly_plan(weekly_plan_content, master_plan_content, week_number)
    
    # 평가 결과를 별도의 JSON 파일로 저장
    evaluation_filename = os.path.join(dir_path, f"evaluation_weekly_{year}_{month:02d}_week{week_number}.json")
    with open(evaluation_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=4)
    print(f">> 주간 계획 평가 결과를 '{evaluation_filename}' 파일에 저장했습니다.")

    return weekly_plan_content, evaluation_result

if __name__ == '__main__':
    # 이 스크립트를 직접 실행할 때만 작동
    PLAN_YEAR = 2025
    PLAN_MONTH = 9
    # 월간 계획 파일 읽기 (직접 실행 시 필요)
    INPUT_PLAN_FILE = os.path.join(GENERATED_PLANS_DIR, f"detailed_plan_{PLAN_YEAR}_{PLAN_MONTH:02d}_final.md")
    try:
        with open(INPUT_PLAN_FILE, "r", encoding="utf-8") as f:
            master_plan = f.read()
        print(f"성공적으로 '{INPUT_PLAN_FILE}' 파일을 읽었습니다.")
    except FileNotFoundError:
        print(f"오류: 월간 계획 파일인 '{INPUT_PLAN_FILE}'을 찾을 수 없습니다. 먼저 월간 계획을 생성해주세요.")
        sys.exit(1)

    # 1주차부터 4주차까지 주간 계획 생성 및 저장
    for week in range(1, 5):
        # 반환값 확인을 위해 변수에 저장
        plan, evaluation = generate_and_save_weekly_plan(master_plan, week, PLAN_YEAR, PLAN_MONTH)
        print(f"\n--- {week}주차 최종 반환 값 ---")
        print(f"Plan Content:", plan[:200] + "...") # 일부만 출력
        print(f"Evaluation Result:", evaluation)

    print("\n모든 주간 콘텐츠 계획 생성이 완료되었습니다.")
