import sys
import io
import os
from dotenv import load_dotenv
from datetime import datetime

# 모듈 임포트
from detailed_planner_agent import initialize_faiss_retriever, generate_and_save_monthly_plan
from weekly_content_generator import generate_and_save_weekly_plan
from daily_content_planner import generate_and_save_daily_plan
from content_draft_generator import generate_and_save_content_drafts
from utils import get_week_number

# stdout의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 전역 설정 ---
PLAN_YEAR = 2025
PLAN_MONTH = 8
PLAN_DAY = 18  # <-- 여기에 원하는 날짜를 입력하세요!


MAX_RETRIES = 2  # 최대 재시도 횟수
MONTHLY_PLAN_PASS_SCORE = 15
WEEKLY_PLAN_PASS_SCORE = 15
DAILY_PLAN_PASS_SCORE = 15
DRAFT_PASS_SCORE = 15

# --- 디렉토리 설정 ---
BASE_DIR = "contents_data"
GENERATED_PLANS_DIR = os.path.join(BASE_DIR, "generated_plans")
WEEKLY_PLANS_DIR = os.path.join(BASE_DIR, "weekly_content_plans")
DAILY_PLANS_DIR = os.path.join(BASE_DIR, "daily_content_plans")
CONTENT_DRAFTS_DIR = os.path.join(BASE_DIR, "content_drafts")
FAISS_INDEX_PATH = "faiss_index"

# --- 폴더 생성 ---
for d in [GENERATED_PLANS_DIR, WEEKLY_PLANS_DIR, DAILY_PLANS_DIR, CONTENT_DRAFTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def run_generation_with_retry(generation_func, pass_score, *args):
    """평가 점수가 기준 미만일 경우 재시도하는 래퍼 함수"""
    for i in range(MAX_RETRIES + 1):
        content, evaluation = generation_func(*args)
        score = evaluation.get("overall_score", 0)
        is_pass = evaluation.get("pass", False)

        if is_pass and score >= pass_score:
            print(f">> {generation_func.__name__} 성공! (점수: {score}, 합격: {is_pass})\n")
            return content, evaluation
        else:
            print(f"WARNING: {generation_func.__name__} 평가 실패. (점수: {score}, 합격: {is_pass})")
            if i < MAX_RETRIES:
                print(f"   재시도합니다... ({i + 1}/{MAX_RETRIES})\n")
            else:
                print(f"   최대 재시도 횟수({MAX_RETRIES})를 초과하여 중단합니다.\n")
                return None, evaluation
    return None, None # Should not be reached

if __name__ == '__main__':
    print(f"\n--- {PLAN_YEAR}년 {PLAN_MONTH}월 {PLAN_DAY}일 콘텐츠 기획 및 초안 생성 프로세스 시작 ---")

    initialize_faiss_retriever()

    # 1. 월간 계획 생성 (재시도 로직 포함)
    monthly_plan_file = os.path.join(GENERATED_PLANS_DIR, f"detailed_plan_{PLAN_YEAR}_{PLAN_MONTH:02d}_final.md")
    if os.path.exists(monthly_plan_file):
        print(f"\n>> 기존 월간 계획 파일 '{monthly_plan_file}'을 불러옵니다.")
        with open(monthly_plan_file, "r", encoding="utf-8") as f:
            master_plan_content = f.read()
    else:
        master_plan_content, _ = run_generation_with_retry(
            generate_and_save_monthly_plan, 
            MONTHLY_PLAN_PASS_SCORE,
            PLAN_YEAR, 
            PLAN_MONTH,
            GENERATED_PLANS_DIR
        )
        if not master_plan_content:
            print("월간 계획 생성에 최종 실패하여 프로세스를 중단합니다.")
            sys.exit(1)

    # 2. 주간 계획 생성 (재시도 로직 포함)
    week_number = get_week_number(PLAN_YEAR, PLAN_MONTH, PLAN_DAY)
    weekly_plan_file = os.path.join(WEEKLY_PLANS_DIR, f"weekly_plan_{PLAN_YEAR}_{PLAN_MONTH:02d}_week{week_number}.md")
    if os.path.exists(weekly_plan_file):
        print(f"\n>> 기존 주간 계획 파일 '{weekly_plan_file}'을 불러옵니다.")
        with open(weekly_plan_file, "r", encoding="utf-8") as f:
            weekly_plan_content = f.read()
    else:
        weekly_plan_content, _ = run_generation_with_retry(
            generate_and_save_weekly_plan,
            WEEKLY_PLAN_PASS_SCORE,
            master_plan_content,
            week_number,
            PLAN_YEAR,
            PLAN_MONTH,
            WEEKLY_PLANS_DIR
        )
        if not weekly_plan_content:
            print("주간 계획 생성에 최종 실패하여 프로세스를 중단합니다.")
            sys.exit(1)

    # 3. 일일 계획 생성 (재시도 로직 포함)
    daily_plan_file = os.path.join(DAILY_PLANS_DIR, f"daily_plan_{PLAN_YEAR}_{PLAN_MONTH:02d}_{PLAN_DAY:02d}.md")
    if os.path.exists(daily_plan_file):
        print(f"\n>> 기존 일일 계획 파일 '{daily_plan_file}'을 불러옵니다.")
        with open(daily_plan_file, "r", encoding="utf-8") as f:
            daily_plan_content = f.read()
    else:
        daily_plan_content, _ = run_generation_with_retry(
            generate_and_save_daily_plan,
            DAILY_PLAN_PASS_SCORE,
            PLAN_YEAR,
            PLAN_MONTH,
            PLAN_DAY,
            weekly_plan_content,
            DAILY_PLANS_DIR
        )
        if not daily_plan_content:
            print("일일 계획 생성에 최종 실패하여 프로세스를 중단합니다.")
            sys.exit(1)

    # 4. 콘텐츠 초안 생성 (개별 초안에 대한 재시도 로직은 content_draft_generator에 이미 있음)
    print(f"\n>> {PLAN_YEAR}년 {PLAN_MONTH}월 {PLAN_DAY}일의 콘텐츠 초안을 생성합니다...")
    # 여기서는 재시도 로직을 적용하지 않고, 개별 초안의 성공 여부는 로그로 확인
    draft_contents, evaluation_results = generate_and_save_content_drafts(PLAN_YEAR, PLAN_MONTH, PLAN_DAY, daily_plan_content, CONTENT_DRAFTS_DIR)
    
    successful_drafts = sum(1 for e in evaluation_results if e.get('pass'))
    print(f"\n>> 콘텐츠 초안 생성 완료: 총 {len(draft_contents)}개 중 {successful_drafts}개 성공.")

    print(f"\n--- {PLAN_YEAR}년 {PLAN_MONTH}월 {PLAN_DAY}일 콘텐츠 기획 및 초안 생성 프로세스 완료 ---")
