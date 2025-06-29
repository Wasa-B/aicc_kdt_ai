import os
import json
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END    # 행그래프 상태 표시 관련 모듈
from langgraph.graph.message import add_messages      # 연결되는 대화 추가 모듈

from langchain_core.messages import ToolMessage     # LLM이 호출한 도구로 나타난 결과를 다시 모델에게 입력 메시지로 제공하는 모듈

from langgraph.graph import StateGraph, START, END    # 행그래프 상태 표시 관련 모듈
from langgraph.graph.message import add_messages      # 연결되는 대화 추가 모듈

from typing import Annotated                          # 타입 힌트에 주석 추가 모듈
from typing_extensions import TypedDict               # 키-값 타입을 정의하는 모듈

from langchain_core.messages import ToolMessage     # LLM이 호출한 도구로 나타난 결과를 다시 모델에게 입력 메시지로 제공하는 모듈

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily_client = TavilyClient()

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm = ChatOpenAI(model="gpt-3.5-turbo")

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):                            # 그래프 상태 정의
  messages: Annotated[list, add_messages]          # 리스트에 메시지 누적

graph_builder = StateGraph(State)             # 대화 흐름을 관리하는 변수

def chatbot(state: State) :
  return {"messages":[llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)


class BasicToolNode:
  """
    마지막 AIMessage에서 요청된 도구를 실행하는 노드
  """
  def __init__(self, tools: list)->None:

    self.tools_by_name = {tool.name: tool for tool in tools} # ["tavily_search_results_json" : TavilySearchResults()]
  def __call__(self, inputs:dict):
    if messages := inputs.get("messages", []):
      message = messages[-1] # 마지막 메세지
    else:
      raise ValueError("No message found in inputs")

    outputs=[]
    for tool_call in message.tool_calls: # message에서 호출된 도구를 불러옴
      tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call['args'])
      outputs.append(
          ToolMessage(
              content=json.dumps(tool_result),
              name=tool_call["name"],
              tool_call_id=tool_call["id"]
          )
      )
    return {"messages":outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


def route_tools(
    state: State, # 현재 상태 개체
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0: # 만약 마지막 메시지에 tool_calls가 존재하고 1개 이상이면
        return "tools"
    return END


# 조건부 엣지 연결
graph_builder.add_conditional_edges(
    "chatbot", # 시작 노드
    route_tools,
    {"tools": "tools", END: END}, # 반환값이 "tools" 면 "tools" 노드로, END 면 END로 이동
)
# 엣지 연결
graph_builder.add_edge("tools", "chatbot") # 도구가 호출될 때마다 챗봇으로 돌아가 다음 단계를 결정
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}): # graph 노드 호출 결과 받아옴
        for value in event.values():
            # print("Assistant:", value["messages"][-1].contents) # AI 답변 출력
            print("Assistant:", value["messages"][-1].content) # AI 답변 출력


# stream_graph_updates("인플루언서 콘텐츠 업로드를 위한 2025년 8월달 경주의 볼거리를 알려줘")
stream_graph_updates("August 2025 tourist destinations to utilize influencer content to promote the Gyeongju region.")

# result = llm_with_tools.invoke("ai agent는 무엇인가요?") # llm 자체에서 답변 생성

# llm_with_tools = llm.bind_tools(tools)
# print(result)