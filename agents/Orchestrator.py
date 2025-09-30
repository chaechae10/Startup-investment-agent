import json
from typing import List

from graph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def Orchestrator(state: GraphState) -> GraphState:
    """
    Orchestrator Agent:
    - 유저 질문을 LLM에 던져 기업 리스트를 JSON으로 추출
    - 결과를 state["company_list"]에 저장
    """
    question = state["question"]

    # [1] 기업 리스트 추출 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 JSON만 출력하는 도우미다. "
                   "사용자가 언급한 기업명을 JSON 배열로 추출해서 {{\"companies\": [...]}} 형식으로 출력해."),
        ("user", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": question})

    # [2] JSON 파싱
    # [2] JSON 파싱
    try:
        data = json.loads(response.content)
        companies: List[str] = data.get("companies", []) if isinstance(data, dict) else []
    except Exception:
        companies = []


    # [3] state에 저장
    # state["company_list"] = companies
    print(state)
    return {"company_list": companies}