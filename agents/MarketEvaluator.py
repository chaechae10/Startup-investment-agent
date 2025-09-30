import os
from typing import List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from graph_state import GraphState
# --- 환경 변수 로드 ---
load_dotenv()

# --- PyTorch 전용 환경 설정 ---
os.environ["USE_TF"] = "0"   # TensorFlow 비활성화
os.environ["USE_JAX"] = "0"  # JAX 비활성화

# --- LLM 설정 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --- 문서 포맷터 ---
def format_docs(docs: List[Any], max_chars: int = 6000) -> str:
    parts, total_len = [], 0
    for d in docs:
        text = getattr(d, "page_content", "")
        src = (getattr(d, "metadata", {}) or {}).get("source")
        entry = f"[source] {src}\n{text}" if src else text
        total_len += len(entry)
        if total_len > max_chars:
            break
        parts.append(entry)
    return "\n\n".join(parts)

# --- 프롬프트 ---
market_report_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
당신은 스타트업 투자 심사역입니다.
주어진 문맥을 활용하여 '헬스케어' 도메인의 시장성을 분석하세요.
## 1. 선택한 산업 분석 *(시장성 평가 결과)*
- **등장 배경**:
    - (핵심 요인 bullet)
- **시장 흐름과 전망**:
    - (CAGR, 지역별 특징 bullet)
- **시장 규모**:
    - (정량 수치, 연도 기준 bullet)
- **세부 분류**:
    - (산업 내 주요 카테고리 bullet)
[문맥]
{context}
"""
)

# --- 체인 생성 ---
def create_market_eval_agent(docs):
    # :흰색_확인_표시: HuggingFace 임베딩 사용 (PyTorch 전용)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    # 벡터스토어 구성
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # RetrievalQA 체인 반환
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": market_report_prompt},
        return_source_documents=False,
    )

# --- 웹검색 ---
def web_search_docs(query: str, num_results: int = 5):
    try:
        urls = [item.get("link") for item in DuckDuckGoSearchAPIWrapper().results(query, max_results=num_results)]
    except Exception:
        urls = []
    if not urls:
        return []
    docs = WebBaseLoader(urls).load()
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)

# --- LangGraph 노드용 함수 ---
def MarketEvaluator(state: GraphState) -> GraphState:
    query = "헬스케어 시장 분석"
    docs = web_search_docs(query, num_results=5)

    if not docs:
        state["MarketEvaluator"] = "검색 결과 없음"
        return state

    agent = create_market_eval_agent(docs)
    result = agent.invoke({"query": query})
    report = result.get("result") or result.get("output_text", "")

    state["search_context"] = format_docs(docs)
    state["MarketEvaluator"] = report

    print("마켓평가 ~~~~~~~~~~~~~~~~~~~")
    print(state)

    return state

