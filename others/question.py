from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def ask_question(question: str, prompt: str = "당신은 유용한 질문 답변 도우미입니다.") -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,  # 창의성 조절
    )

    answer = response.choices[0].message.content.strip()
    return answer

if __name__ == "__main__":
    user_question = input("질문을 입력하세요: ")
    answer = ask_question(user_question)
    print("\n답변:", answer)