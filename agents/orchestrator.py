"""
orchestrator.py

전체 파이프라인의 흐름을 제어하는 중앙 조정자(Coordinator).

책임:
- 사용자 입력 → 각 Agent / Tool 호출 순서 제어
- 응급 여부에 따른 분기 처리
- Agent 간 데이터 전달
- 최종 응답 구조 조립

❌ 하지 않는 것
- GPT 호출
- 프롬프트 작성
- 의료 판단
- 병원 추천 로직
"""

from typing import Dict, Any, Optional

from agents import (
    SymptomAgent,
    ExplainAgent,
    SafetyAgent,
)

from tools import (
    MLPredictTool,
    HospitalLookupTool,
)

from pipelines.symptom_to_vector import symptom_to_vector
from pipelines.topk_postprocess import postprocess_topk
from pipelines.response_formatter import format_response


class Orchestrator:
    """
    Orchestrator는 상태를 거의 가지지 않는 Stateless Coordinator이다.
    (세션 상태는 상위 레이어—FastAPI / Streamlit—에서 관리)
    """

    def __init__(
        self,
        symptom_agent: SymptomAgent,
        explain_agent: ExplainAgent,
        safety_agent: SafetyAgent,
        ml_predict_tool: MLPredictTool,
        hospital_tool: HospitalLookupTool,
    ):
        """
        main.py에서 생성한 Agent / Tool 인스턴스를 주입받는다.

        이렇게 하는 이유:
        - LLM client 공유
        - 테스트 용이성
        - 의존성 역전 (DI)
        """
        self.symptom_agent = symptom_agent
        self.explain_agent = explain_agent
        self.safety_agent = safety_agent
        self.ml_predict_tool = ml_predict_tool
        self.hospital_tool = hospital_tool

    # =========================================================
    # 1️⃣ 1차 사용자 입력 처리 (메인 플로우)
    # =========================================================
    def handle_user_input(
        self,
        user_input: str,
        user_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        최초 사용자 증상 입력 처리

        Flow:
        1. 증상 추출
        2. 벡터화
        3. ML Top-K 예측
        4. 응급 여부 판단
        5. 분기 처리
        """

        # 1️⃣ 자연어 → 증상 리스트
        symptoms = self.symptom_agent.extract(user_input)

        # 2️⃣ 증상 → 멀티핫 벡터
        vector = symptom_to_vector(symptoms)

        # 3️⃣ XGBoost 예측 (Top-K)
        raw_topk = self.ml_predict_tool.predict(vector)
        topk = postprocess_topk(raw_topk)

        # 4️⃣ 응급 여부 판단
        emergency_result = self.safety_agent.check(symptoms)

        # =====================================================
        # 🚨 응급 상황
        # =====================================================
        if emergency_result.is_emergency:
            hospital_info = self.hospital_tool.lookup(
                location=user_location
            )

            return format_response(
                is_emergency=True,
                symptoms=symptoms,
                emergency_reason=emergency_result.reason,
                hospital_info=hospital_info,
            )

        # =====================================================
        # ✅ 비응급 상황
        # =====================================================
        explanation = self.explain_agent.generate(
            symptoms=symptoms,
            topk=topk,
            emergency=False,
        )

        return format_response(
            is_emergency=False,
            symptoms=symptoms,
            topk=topk,
            explanation=explanation,
            show_hospital_option=True,  # "원하면 병원 안내" 문구용
        )

    # =========================================================
    # 2️⃣ 사용자가 "병원 알려줘"라고 했을 때
    # =========================================================
    def handle_hospital_request(
        self,
        user_location: Optional[str] = None,
    ):
        raw_hospitals = self.hospital_tool.lookup(
            location=user_location
        )

        explanation = self.hospital_explain_agent.generate(
            hospitals=raw_hospitals
        )

        return {
            "type": "hospital_info",
            "explanation": explanation,
        }
