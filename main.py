from typing import Annotated, Literal, Sequence, TypedDict, List
from langchain_teddynote.graphs import visualize_graph


from dataproc.loader import prepare_data
from graph_state import GraphState

from agents.Orchestrator import Orchestrator
from agents.TechScribe import make_techscribe_agent
from agents.MarketEvaluator import MarketEvaluator
from agents.CompetitorAnalyzer import CompetitorAnalyzer
from agents.InvestmentAdvisor import InvestmentAdvisor
from agents.ReportGenerator import ReportGenerator


from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()



def build_pdf_retriever(pdf_paths: list[str]):
    """여러 PDF 파일을 로드하고 하나의 retriever로 반환"""
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    # 벡터 임베딩
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # FAISS 벡터스토어
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # retriever로 변환
    return vectorstore.as_retriever(search_kwargs={"k": 4})



# def build_graph(company_list: list[str]):
#     workflow = StateGraph(GraphState)

#     # --- 기업별 TechScribe 노드 (직렬 실행) ---
#     prev_node = START
#     for idx, company in enumerate(company_list):
#         node_name = f"techscribe_{idx}"
#         workflow.add_node(node_name, make_techscribe_agent(idx))
#         workflow.add_edge(prev_node, node_name)   # ✅ 직렬 연결
#         prev_node = node_name

#     # 마지막 TechScribe 노드 → competitor, report
#     workflow.add_edge(prev_node, "competitor")
#     workflow.add_edge(prev_node, "report")

#     # 나머지 에이전트 노드
#     workflow.add_node("market_eval", MarketEvaluator)
#     workflow.add_node("competitor", CompetitorAnalyzer)
#     workflow.add_node("investment", InvestmentAdvisor)
#     workflow.add_node("report", ReportGenerator)

#     # --- 엣지 정의 ---
#     workflow.add_edge(START, "market_eval")

#     # 경쟁사 비교 + 시장성 평가 → 투자 판단
#     workflow.add_edge("competitor", "investment")
#     workflow.add_edge("market_eval", "investment")

#     # 투자 판단 → 보고서
#     workflow.add_edge("investment", "report")

#     # 시장성 평가도 보고서로 직접 연결
#     workflow.add_edge("market_eval", "report")
#     workflow.add_edge("competitor", "report")

#     # 보고서 → 종료
#     workflow.add_edge("report", END)

#     return workflow.compile()
def build_graph(company_list: list[str]):
    workflow = StateGraph(GraphState)

    # --- 노드 정의 ---
    workflow.add_node("techscribe", make_techscribe_agent)   # 기업별 기술요약
    workflow.add_node("market_eval", MarketEvaluator)        # 시장성 평가
    workflow.add_node("competitor", CompetitorAnalyzer)      # 경쟁사 비교
    workflow.add_node("investment", InvestmentAdvisor)       # 투자 판단
    workflow.add_node("report", ReportGenerator)             # 최종 보고서

    # --- 직렬 엣지 정의 ---
    workflow.add_edge(START, "techscribe")
    workflow.add_edge("techscribe", "market_eval")
    workflow.add_edge("market_eval", "competitor")
    workflow.add_edge("competitor", "investment")
    workflow.add_edge("investment", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def make_init_state(question: str) -> GraphState:
    """GraphState 구조에 맞는 안전한 초기 state 생성"""
    return {
        "question": question,
        "company_list": [],
        "search_context": "",
        "rag_context": "",
        "TechScribe": "",
        "MarketEvaluator": "",
        "CompetitorAnalyzer": "",
        "InvestmentAdvisor": "",
        "answer": "",
        "chat_history": []
    }

def main():
    vs = prepare_data()
    pdf_files = glob.glob("data/raw/*.pdf")

    retriever = build_pdf_retriever(pdf_files)

    # 초기 state 생성
    state = make_init_state("레몬 헬스케어, 큐라코 비교해줘")
    state["retriever"] = retriever 

    # [2] Orchestrator 실행 (company_list 추출)
    state = Orchestrator(state)
    companies = state["company_list"]
    print("Orchestrator ok", companies)

    # [3] 그래프 빌드
    graph = build_graph(companies)

    # [4] 실행
    result = graph.invoke(state)

    # [5] 결과 저장
    with open("outputs/final_report.md", "w", encoding="utf-8") as f:
        f.write(result["answer"])
    print(result["answer"])
    print("보고서 생성 완료: outputs/final_report.md")

if __name__ == "__main__":
    main()