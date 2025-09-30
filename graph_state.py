from typing import Annotated, Literal, Sequence, TypedDict, List


class GraphState(TypedDict):
    question: Annotated[str, "Question"]                  # 질문
    search_context: Annotated[str, "Context"]             # 검색 결과 (예: Tavily, PDF 등)
    rag_context: Annotated[str, "Context"]                # RAG 결과
    company_list: Annotated[List[str], "Companies"]       # 추출된 기업 리스트
    TechScribe: Annotated[str, "Context"]                 # 기술 요약 결과
    MarketEvaluator: Annotated[str, "Context"]            # 시장성 평가 결과
    CompetitorAnalyzer: Annotated[str, "Context"]         # 경쟁사 비교 결과
    InvestmentAdvisor: Annotated[str, "Context"]          # 투자 판단 결과
    answer: Annotated[str, "Answer"]                      # 최종 답변 / 보고서
    chat_history: Annotated[list, "Messages"]             # 누적 대화 로그