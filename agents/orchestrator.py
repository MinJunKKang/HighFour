# 2026-01-23 송진우
"""
orchestrator.py

역할:
- 프로젝트 전체 파이프라인의 '흐름'만 제어한다.
- 판단(ML), 생성(LLM), 규칙(Safety)을 직접 수행하지 않는다.
- 각 Agent / Tool / Pipeline을 올바른 순서로 호출한다.

중요 설계 원칙:
- Orchestrator는 '지휘자'이지 '연주자'가 아니다.
- 모든 실제 작업은 다른 모듈에 위임한다.
"""

from typing import Dict, Any

# Agents
from agents.symptom_agent import SymptomAgent
from agents.explain_agent import ExplainAgent
from agents.safety_agent import SafetyAgent

# Tools
from tools.ml_predict_tool import MLPredictTool
from tools.web_search_tool import WebSearchTool
from tools.hospital_lookup_tool import HospitalLookupTool

# Pipelines
from pipelines.symptom_to_vector import symptom_to_vector
from pipelines.topk_postprocess import postprocess_topk
from pipelines.response_formatter import ResponseFormatter


class Orchestrator:
    """
    Orchestrator

    입력:
    - 사용자 자연어 증상 텍스트

    출력:
    - UI(Streamlit)가 바로 사용할 수 있는 최종 응답 dict
    """

    def __init__(
        self,
        symptom_agent: SymptomAgent,
        explain_agent: ExplainAgent,
        safety_agent: SafetyAgent,
        ml_predict_tool: MLPredictTool,
        web_search_tool: WebSearchTool,
        hospital_lookup_tool: HospitalLookupTool,
        response_formatter: ResponseFormatter,
    ):
        """
        모든 의존성은 main.py에서 생성되어 주입된다.
        Orchestrator는 생성 책임을 가지지 않는다.
        """

        self.symptom_agent = symptom_agent
        self.explain_agent = explain_agent
        self.safety_agent = safety_agent

        self.ml_predict_tool = ml_predict_tool
        self.web_search_tool = web_search_tool
        self.hospital_lookup_tool = hospital_lookup_tool

        self.response_formatter = response_formatter

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        전체 파이프라인 실행 진입점

        Parameters
        ----------
        user_input : str
            사용자가 입력한 자연어 증상 설명

        Returns
        -------
        Dict[str, Any]
            최종 사용자 응답 (UI 친화적 포맷)
        """

        # =========================================================
        # 1️⃣ 자연어 → 정규화된 증상 리스트 (LLM Agent)
        # =========================================================
        normalized_symptoms = self.symptom_agent.run(user_input)

        # 증상이 거의 없는 경우 조기 종료 (Fail-safe)
        if not normalized_symptoms:
            return self.response_formatter.empty_input()

        # =========================================================
        # 2️⃣ 응급 여부 판단 (Rule + Safety Agent)
        # =========================================================
        emergency_flag = self.safety_agent.check(normalized_symptoms)

        # =========================================================
        # 3️⃣ 증상 → 멀티핫 벡터 변환 (Pipeline)
        # =========================================================
        symptom_vector = symptom_to_vector(normalized_symptoms)

        # =========================================================
        # 4️⃣ ML 질병 예측 (XGBoost Tool)
        # =========================================================
        raw_predictions = self.ml_predict_tool.predict(symptom_vector)

        # =========================================================
        # 5️⃣ Top-K 후처리 (정렬 / 임계값 / 불확실 처리)
        # =========================================================
        topk_results = postprocess_topk(raw_predictions)

        # =========================================================
        # 6️⃣ 질병 설명 생성 (GPT-5.2 ExplainAgent)
        # =========================================================
        explanation = self.explain_agent.run({
            "normalized_symptoms": normalized_symptoms,
            "topk": topk_results,
            "emergency": emergency_flag
        })

        # =========================================================
        # 7️⃣ 조건부 Web Search / 병원 정보
        # =========================================================
        web_info = None
        hospital_info = None

        # 응급 상황에서는 외부 검색보다 즉각 대응 우선
        if not emergency_flag:
            web_info = self.web_search_tool.search(topk_results)
            hospital_info = self.hospital_lookup_tool.lookup(topk_results)

        # =========================================================
        # 8️⃣ 최종 응답 조립 (Response Formatter)
        # =========================================================
        final_response = self.response_formatter.format(
            symptoms=normalized_symptoms,
            topk=topk_results,
            explanation=explanation,
            web_info=web_info,
            hospital_info=hospital_info,
            emergency=emergency_flag
        )

        return final_response
