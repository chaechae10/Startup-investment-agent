from graph_state import GraphState

from dotenv import load_dotenv
from langchain_teddynote import logging
from typing import Annotated, Sequence, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

load_dotenv()

def InvestmentAdvisor(state, prompt_path="prompts/InvestmentAdvisor_sys_prompt.txt"):
    sys_prompt = Path(prompt_path).read_text(encoding="utf-8")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt
            ),
            ("human",
            "아래는 전체 입력입니다. 이를 바탕으로 스키마에 맞는 결론만 산출하세요. \n"
            "- 경쟁사 비교: {{ competitor }}\n"
            "- 시장 평가: {{ market_eval }}\n"
            )
        ],
        template_format="jinja2"
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "competitor": state.get("CompetitorAnalyzer", ""),
        "market_eval": state.get("MarketEvaluator", "")
    })


    report = response.content if hasattr(response, "content") else str(response)
    print("투자조언 ~~~~~~~~~~~~~~~~~~")
    print(state)
    return {
        "InvestmentAdvisor": report
    }
