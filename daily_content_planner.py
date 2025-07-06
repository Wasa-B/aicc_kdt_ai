

import sys
import io
import os
import json
import re
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import get_week_number # get_week_number 함수 임포트

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 전역 설정 ---
DAILY_PLANS_DIR = "daily_content_plans"
WEEKLY_PLANS_DIR = "weekly_content_plans" # 주간 계획 파일 경로를 위해 추가

# --- 폴더 생성 ---
if not os.path.exists(DAILY_PLANS_DIR):
    os.makedirs(DAILY_PLANS_DIR)

# --- LLM 설정 ---
DAILY_LLM_MODEL = "gpt-4-turbo"
EVALUATION_LLM_MODEL = "gpt-3.5-turbo" # 평가용 LLM 모델

daily_llm = ChatOpenAI(model_name=DAILY_LLM_MODEL, temperature=0.5) # 창의적인 일일 아이디어를 위해 온도 조절
evaluation_llm = ChatOpenAI(model_name=EVALUATION_LLM_MODEL, temperature=0.0) # 평가는 보수적으로

def _extract_json_from_response(response_text):
    """LLM 응답에서 JSON 블록을 추출합니다."""
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{.*\}|\[.*\]", response_text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None

def _generate_daily_plan(year, month, day, weekly_plan_content):
    """특정 날짜에 대한 상세 콘텐츠 계획을 생성하는 헬퍼 함수"""
    print(f"\n>> {year}년 {month}월 {day}일의 상세 콘텐츠 계획을 생성합니다 (Model: {DAILY_LLM_MODEL})...")
    template = """
    당신은 가상 인플루언서의 콘텐츠 매니저입니다.
    아래의 **주간 콘텐츠 계획**을 바탕으로, **{year}년 {month}월 {day}일**에 실행할 **매우 상세하고 구체적인 일일 콘텐츠 계획**을 작성해야 합니다.

    **요구사항:**
    1.  결과는 Markdown 테이블 형식으로 작성해주세요.
    2.  테이블 컬럼은 `플랫폼`, `콘텐츠 상세 아이디어`, `핵심 메시지/해시태그`로 구성해주세요.
    3.  **각 행은 반드시 하나의 독립적인 콘텐츠 아이템을 나타내야 합니다.**
    4.  콘텐츠 상세 아이디어는 너무 길지 않게, 핵심 내용을 요약하여 작성해주세요. 이 내용은 콘텐츠 초안 파일의 제목으로도 사용됩니다.
    5.  각 플랫폼(인스타그램, 유튜브, 블로그, X 등)별로 어떤 콘텐츠를 어떤 내용으로 발행할지, 필요한 시각/청각 요소, 예상 캡션/스크립트 아이디어, 참여 유도 전략, 해시태그 등을 상세하게 포함해주세요.
    6.  해당 날짜의 요일과 월간 계획의 테마를 고려하여 콘텐츠를 기획해주세요.

    **예시 (일일 콘텐츠 계획 테이블 - 하루에 여러 콘텐츠가 있는 경우):**
    ```markdown
    | 플랫폼       | 콘텐츠 상세 아이디어                                               | 핵심 메시지/해시태그                 |
    |--------------|-------------------------------------------------------------------|--------------------------------------|
    | 인스타그램   | 첨성대 핑크뮬리 한복 사진 (5장 캐러셀)                             | #가을의경주 #첨성대핑크뮬리          |
    | 유튜브 쇼츠  | 황리단길 맛집 탐방 (짧은 클립)                                     | #경주맛집 #황리단길 #가을여행         |
    | 블로그       | 경주 가을 꽃 명소 완벽 가이드 (첨성대, 동궁과 월지)                | #경주가을꽃 #여행가이드              |
    ```

    --- 주간 콘텐츠 계획 ---
    {weekly_plan}
    --- 주간 콘텐츠 계획 끝 ---

    **{year}년 {month}월 {day}일 일일 콘텐츠 계획 (Markdown 테이블):**
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | daily_llm | StrOutputParser()
    return chain.invoke({"year": year, "month": month, "day": day, "weekly_plan": weekly_plan_content})

def _evaluate_daily_plan(generated_daily_plan_content, weekly_plan_content, year, month, day):
    """생성된 일일 계획을 평가하고 결과를 JSON으로 반환합니다."""
    print(f"\n>> 생성된 {year}년 {month}월 {day}일 일일 계획을 평가합니다 (Model: {EVALUATION_LLM_MODEL})...")
    evaluation_prompt = PromptTemplate.from_template(
        """
        당신은 콘텐츠 계획 평가 전문가입니다.
        아래의 **주간 콘텐츠 계획**과 **생성된 일일 계획**을 바탕으로, 일일 계획의 품질을 평가해주세요.
        
        **평가 기준:**
        1.  **주간 계획과의 일관성**: 일일 계획이 주간 계획의 목표와 콘텐츠 아이디어에 부합하는가? (점수: 0-5점)
        2.  **구체성 및 상세함**: 각 콘텐츠 아이디어가 실행 가능할 정도로 구체적이고 상세한가? (점수: 0-5점)
        3.  **다양성**: 다양한 플랫폼과 콘텐츠 형식이 적절히 활용되었는가? (점수: 0-5점)
        4.  **요구사항 준수**: 'Markdown 테이블 형식', '각 행은 독립적인 콘텐츠 아이템', '콘텐츠 상세 아이디어 요약' 등 모든 요구사항을 준수했는가? (점수: 0-5점)
        5.  **형식 준수**: Markdown 테이블 형식 및 가독성이 좋은가? (점수: 0-5점)

        **평가 결과는 다음 JSON 형식으로만 출력해주세요. 다른 설명은 절대 추가하지 마세요:**
        ```json
        {{
            "overall_score": <총점 (25점 만점)>,
            "feedback": "<상세 피드백>",
            "pass": <true/false (총점 18점 이상이면 true)>
        }}
        ```

        --- 주간 콘텐츠 계획 ---
        {weekly_plan}
        --- 주간 콘텐츠 계획 끝 ---

        --- 생성된 일일 계획 ---
        {generated_daily_plan}
        --- 생성된 일일 계획 끝 ---
        """
    )
    chain = evaluation_prompt | evaluation_llm | StrOutputParser()
    evaluation_result_str = chain.invoke({"generated_daily_plan": generated_daily_plan_content, "weekly_plan": weekly_plan_content})
    
    json_str = _extract_json_from_response(evaluation_result_str)

    if json_str:
        try:
            evaluation_result = json.loads(json_str)
            print(f"\n>> 일일 계획 평가 결과: 총점 {evaluation_result.get('overall_score', 'N/A')}점, 합격 여부: {evaluation_result.get('pass', 'N/A')}")
            print(f"   상세 피드백: {evaluation_result.get('feedback', 'N/A')}")
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"WARNING: 일일 계획 평가 결과 파싱 오류: {e}")
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

# --- 메인 일일 계획 생성 함수 ---
def generate_and_save_daily_plan(year, month, day, weekly_plan_content, dir_path):
    """일일 계획을 생성하고, 콘텐츠와 평가 결과를 반환합니다."""
    print(f"\n>> {year}년 {month}월 {day}일 일일 계획 생성 프로세스를 시작합니다...")
    daily_plan_content = _generate_daily_plan(year, month, day, weekly_plan_content)
    
    daily_plan_filename = os.path.join(dir_path, f"daily_plan_{year}_{month:02d}_{day:02d}.md")
    with open(daily_plan_filename, "w", encoding="utf-8") as f:
        f.write(f"# {year}년 {month}월 {day}일 상세 콘텐츠 실행 계획\n\n")
        f.write(daily_plan_content)
    print(f">> {year}년 {month}월 {day}일 계획이 '{daily_plan_filename}' 파일로 저장되었습니다.")

    # 일일 계획 평가
    evaluation_result = _evaluate_daily_plan(daily_plan_content, weekly_plan_content, year, month, day)
    
    # 평가 결과를 별도의 JSON 파일로 저장
    evaluation_filename = os.path.join(dir_path, f"evaluation_daily_{year}_{month:02d}_{day:02d}.json")
    with open(evaluation_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=4)
    print(f">> 일일 계획 평가 결과를 '{evaluation_filename}' 파일에 저장했습니다.")

    return daily_plan_content, evaluation_result

if __name__ == '__main__':
    # 이 스크립트를 직접 실행할 때만 작동
    PLAN_YEAR = 2025
    PLAN_MONTH = 9
    PLAN_DAY = 5

    # 주간 계획 파일 읽기 (직접 실행 시 필요)
    week_number = get_week_number(PLAN_YEAR, PLAN_MONTH, PLAN_DAY)
    weekly_plan_file = os.path.join(WEEKLY_PLANS_DIR, f"weekly_plan_{PLAN_YEAR}_{PLAN_MONTH:02d}_week{week_number}.md")
    try:
        with open(weekly_plan_file, "r", encoding="utf-8") as f:
            weekly_plan = f.read()
        print(f"성공적으로 '{weekly_plan_file}' 파일을 읽었습니다.")
    except FileNotFoundError:
        print(f"오류: 주간 계획 파일인 '{weekly_plan_file}'을 찾을 수 없습니다. 먼저 주간 계획을 생성해주세요.")
        sys.exit(1)

    # 반환값 확인을 위해 변수에 저장
    plan, evaluation = generate_and_save_daily_plan(PLAN_YEAR, PLAN_MONTH, PLAN_DAY, weekly_plan)
    print("\n--- 최종 반환 값 ---")
    print("Plan Content:", plan[:200] + "...") # 일부만 출력
    print("Evaluation Result:", evaluation)

    print("\n일일 콘텐츠 계획 생성이 완료되었습니다.")
