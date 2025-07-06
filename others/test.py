import os
from rag_pdf import rag_question

pdf_path = './doc/경주 관광 인플루언서 마케팅 3개년 전략 로드맵.pdf'


def month_plan(year,month):
    request =str(year)+"년 " + str(month) + """
월달 액션플랜, 마케팅플랜, 전략 계획, 테마 분석해서 최대한 공간, 시간 정보를 포함하여 자세히 알려줘
            """
    prompt = """
            당신은 pdf파일을 분석하여 자세히 알려주는 설명 전문가입니다.
            """
    answer = rag_question(pdf_path, request, prompt)
    print(answer)
    return answer

def week_plan(question):
    req = question + """\n
    을 참고 해서 한달 동안의 매주 간 콘텐츠 업로드 계획을 설계해줘 (1주, 2주, 3주 ...)
    테마와 시간, 공간, 행사 정보들을 자세히 포함해서 알려줘. 
    """
    answer = rag_question(pdf_path,question, "당신은 pdf와 요청을 분석하여 주간 계획을 짜주는 전문가입니다.", max_tokens=1000)
    return answer

def month_detail_paln(year,month):
    request =str(year)+"년 " + str(month) + """
월달 액션플랜, 마케팅플랜, 전략 계획을 분석해서 한달 전체의 일단위 콘텐츠 업로드 플랜을 설계해줘, 공간과 시간 또는 행사 이름을 포함하여 디테일하게 하게.
콘텐츠 타입은 : instagram, streaming, twit, youtube, blog, facebook, tiktok, news_letter, web_event
결과물은 JSON 형태로 주며,
{title, thema, plans : {day, contents_type, plan}} 
구조로 답변한다.
            """
    prompt = """
            당신은 pdf파일을 분석하여 세부 일정 계획을 설계해주는 설계 전문가입니다.
            """
    answer1 = rag_question(pdf_path, request, prompt)
    print(answer1)
    if os.path.exists("./month_plan/"+str(year)+"/") == False:
        os.makedirs("./month_plan/"+str(year))
    with open("./month_plan/"+str(year)+"/"+str(month)+".txt", "w", encoding="utf-8") as f:
        f.write(answer1)
        pass
    pass


# for i in range(1,13):
    # month_detail_paln(2025, i)
# month_detail_paln(2025, 8)

answer = month_plan(2025, 8)
answer2 = week_plan(answer)
print(answer2)


