

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

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 전역 설정 ---
CONTENT_DRAFTS_DIR = "content_drafts"
DAILY_PLANS_DIR = "daily_content_plans" # 일일 계획 파일 경로를 위해 추가

# --- 폴더 생성 ---
if not os.path.exists(CONTENT_DRAFTS_DIR):
    os.makedirs(CONTENT_DRAFTS_DIR)

# --- LLM 설정 ---
DRAFT_LLM_MODEL = "gpt-4-turbo"
EVALUATION_LLM_MODEL = "gpt-3.5-turbo" # 평가용 LLM 모델

draft_llm = ChatOpenAI(model_name=DRAFT_LLM_MODEL, temperature=0.6)
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

def _parse_daily_plan_content(daily_plan_text):
    """
    일일 계획 텍스트(Markdown 테이블)에서 개별 콘텐츠 아이템을 파싱합니다.
    반환 형식: [{"platform": "...", "idea": "...", "message": "..."}, ...]
    """
    print(">> 일일 계획 내용을 파싱합니다...")
    content_items = []
    lines = daily_plan_text.split('\n')
    
    table_started = False
    header = []
    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue

        parts = [p.strip() for p in line.split('|') if p.strip()]
        if not parts:
            continue

        if '---' in line:
            table_started = True
            continue

        if table_started:
            if len(parts) >= 3:
                platform = parts[0]
                idea = parts[1]
                message = parts[2]
                content_items.append({"platform": platform, "idea": idea, "message": message})
            else:
                print(f"WARNING: 파싱할 수 없는 라인 형식: {line}")

    print(f"DEBUG: 최종 파싱된 아이템: {content_items}")
    return content_items

def _generate_content_draft(platform, idea, message, year, month, day):
    """
    개별 콘텐츠 아이템에 대한 상세 초안/스크립트를 생성하는 헬퍼 함수.
    """
    print(f">> {platform} - '{idea}' 콘텐츠 초안을 생성합니다 (Model: {DRAFT_LLM_MODEL})...")
    draft_prompt = PromptTemplate.from_template(
        """
        당신은 인플루언서의 콘텐츠 제작 전문가입니다.
        다음 정보를 바탕으로 {year}년 {month}월 {day}일에 발행할 **{platform} 콘텐츠**의 상세 초안 또는 스크립트를 작성해주세요.
        **오직 {platform}에 대한 내용만 작성하고, 다른 플랫폼에 대한 내용은 절대 포함하지 마세요.**
        매우 구체적이고, 해당 플랫폼의 특성을 최대한 살려주세요. 불필요한 서론이나 결론 문구는 제외하고, 바로 콘텐츠 본문을 작성해주세요.

        --- 콘텐츠 정보 ---
        플랫폼: {platform}
        콘텐츠 아이디어: {idea}
        핵심 메시지/해시태그: {message}
        --- 콘텐츠 정보 끝 ---

        **요구사항:**
        - **인스타그램**: 매력적인 캡션, 이미지/릴스 아이디어 (구체적인 장면 묘사), 추천 해시태그, 참여 유도 질문.
        - **유튜브 (Vlog/쇼츠)**: 영상 스크립트 초안 (도입-전개-결론), 촬영 가이드 (장면 전환, BGM, 자막 아이디어), 핵심 메시지, 추천 해시태그.
        - **블로그**: 포스팅 제목, 서론, 본론(소제목 포함), 결론, 이미지/영상 삽입 위치 제안, SEO 키워드, 관련 링크.
        - **X (구 트위터)**: 간결하고 임팩트 있는 트윗 여러 개 (최대 280자), 이미지/GIF/영상 아이디어, 관련 해시태그.
        - **뉴스레터**: 제목, 도입부, 본문 요약, CTA (Call to Action), 이미지/링크 삽입 제안.

        **콘텐츠 초안/스크립트:**
        """
    )
    draft_chain = draft_prompt | draft_llm | StrOutputParser()
    return draft_chain.invoke({
        "platform": platform,
        "idea": idea,
        "message": message,
        "year": year,
        "month": month,
        "day": day
    })

def _evaluate_content_draft(generated_draft_content, platform, idea, message):
    """생성된 콘텐츠 초안을 평가하고 결과를 JSON으로 반환합니다."""
    print(f">> {platform} - '{idea}' 콘텐츠 초안을 평가합니다 (Model: {EVALUATION_LLM_MODEL})...")
    evaluation_prompt = PromptTemplate.from_template(
        """
        당신은 콘텐츠 초안 평가 전문가입니다.
        아래의 **콘텐츠 정보**와 **생성된 콘텐츠 초안**을 바탕으로, 초안의 품질을 평가해주세요.
        
        **평가 기준:**
        1.  **요구사항 준수**: 요청된 플랫폼에 대한 내용만 포함하고, 다른 플랫폼 내용은 없는가? (점수: 0-5점)
        2.  **구체성 및 상세함**: 콘텐츠 아이디어를 바탕으로 구체적이고 실행 가능한 초안이 생성되었는가? (점수: 0-5점)
        3.  **플랫폼 특성 반영**: 해당 플랫폼의 특성(캡션, 스크립트, 해시태그 등)이 잘 반영되었는가? (점수: 0-5점)
        4.  **불필요한 문구 제거**: 서론이나 결론 등 불필요한 문구가 제거되었는가? (점수: 0-5점)
        5.  **가독성 및 완성도**: 전체적인 가독성이 좋고, 바로 활용할 수 있을 정도로 완성도가 높은가? (점수: 0-5점)

        **평가 결과는 다음 JSON 형식으로만 출력해주세요. 다른 설명은 절대 추가하지 마세요:**
        ```json
        {{
            "overall_score": <총점 (25점 만점)>,
            "feedback": "<상세 피드백>",
            "pass": <true/false (총점 18점 이상이면 true)>
        }}
        ```

        --- 콘텐츠 정보 ---
        플랫폼: {platform}
        콘텐츠 아이디어: {idea}
        핵심 메시지/해시태그: {message}
        --- 콘텐츠 정보 끝 ---

        --- 생성된 콘텐츠 초안 ---
        {generated_draft_content}
        --- 생성된 콘텐츠 초안 끝 ---
        """
    )
    chain = evaluation_prompt | evaluation_llm | StrOutputParser()
    evaluation_result_str = chain.invoke({
        "generated_draft_content": generated_draft_content,
        "platform": platform,
        "idea": idea,
        "message": message
    })
    
    json_str = _extract_json_from_response(evaluation_result_str)

    if json_str:
        try:
            evaluation_result = json.loads(json_str)
            print(f"\n>> 콘텐츠 초안 평가 결과: 총점 {evaluation_result.get('overall_score', 'N/A')}점, 합격 여부: {evaluation_result.get('pass', 'N/A')}")
            print(f"   상세 피드백: {evaluation_result.get('feedback', 'N/A')}")
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"WARNING: 콘텐츠 초안 평가 결과 파싱 오류: {e}")
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

# --- 메인 콘텐츠 초안 생성 함수 ---
def generate_and_save_content_drafts(year, month, day, daily_plan_content,dir_path):
    """일일 계획에 따라 여러 콘텐츠 초안을 생성하고, 초안과 평가 결과를 반환합니다."""
    print(f"\n>> {year}년 {month}월 {day}일의 콘텐츠 초안 생성 프로세스를 시작합니다...")
    content_items = _parse_daily_plan_content(daily_plan_content)
    
    if not content_items:
        print("파싱된 콘텐츠 아이템이 없습니다. 일일 계획 파일 형식을 확인해주세요.")
        return [], [] # 빈 리스트 반환

    all_drafts = []
    all_evaluations = []

    for i, item in enumerate(content_items):
        platform = item.get("platform", "Unknown")
        idea = item.get("idea", "No Idea Provided")
        message = item.get("message", "")

        # 파일명에 부적합한 문자 제거
        safe_platform = re.sub(r'[\\/*?"<>|]', '_', platform)
        # safe_idea = re.sub(r'[\\/*?"<>|]', '_', idea)[:30] # 아이디어를 파일명에 일부 사용

        draft_content = _generate_content_draft(platform, idea, message, year, month, day)
        
        # 파일로 저장: YYYY_MM_DD_(번호)_플랫폼_아이디어.md 형식
        output_filename = os.path.join(dir_path, f"{year}_{month:02d}_{day:02d}_({i+1})_{safe_platform}.md")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"# {year}년 {month}월 {day}일 {platform} 콘텐츠 초안: {idea}\n\n")
            f.write(draft_content)
        print(f">> 콘텐츠 초안을 '{output_filename}' 파일로 저장했습니다.")

        # 콘텐츠 초안 평가
        evaluation_result = _evaluate_content_draft(draft_content, platform, idea, message)
        
        # 평가 결과를 별도의 JSON 파일로 저장
        evaluation_filename = os.path.join(dir_path, f"evaluation_draft_{year}_{month:02d}_{day:02d}_({i+1})_{safe_platform}.json")
        with open(evaluation_filename, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=4)
        print(f">> 콘텐츠 초안 평가 결과를 '{evaluation_filename}' 파일에 저장했습니다.")

        all_drafts.append(draft_content)
        all_evaluations.append(evaluation_result)

    print("\n모든 콘텐츠 초안 생성이 완료되었습니다.")
    return all_drafts, all_evaluations

if __name__ == '__main__':
    # 이 스크립트를 직접 실행할 때만 작동
    PLAN_YEAR = 2025
    PLAN_MONTH = 9
    PLAN_DAY = 5

    # 일일 계획 파일 읽기 (직접 실행 시 필요)
    daily_plan_file = os.path.join(DAILY_PLANS_DIR, f"daily_plan_{PLAN_YEAR}_{PLAN_MONTH:02d}_{PLAN_DAY:02d}.md")
    try:
        with open(daily_plan_file, "r", encoding="utf-8") as f:
            daily_plan = f.read()
        print(f"성공적으로 '{daily_plan_file}' 파일을 읽었습니다.")
    except FileNotFoundError:
        print(f"오류: 일일 계획 파일인 '{daily_plan_file}'을 찾을 수 없습니다. 먼저 일일 계획을 생성해주세요.")
        sys.exit(1)

    # 반환값 확인을 위해 변수에 저장
    drafts, evaluations = generate_and_save_content_drafts(PLAN_YEAR, PLAN_MONTH, PLAN_DAY, daily_plan)
    print("\n--- 최종 반환 값 ---")
    print(f"{len(drafts)}개의 초안과 {len(evaluations)}개의 평가 결과가 생성되었습니다.")
    if drafts:
        print("\n첫 번째 초안:", drafts[0][:200] + "...")
        print("첫 번째 평가:", evaluations[0])
