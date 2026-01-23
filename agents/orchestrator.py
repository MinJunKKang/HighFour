# agents/orchestrator.py

"""
Orchestrator
============

역할:
- 전체 파이프라인 순서 제어
- Agent / Tool 간 데이터 전달
- 응급 여부에 따른 분기 처리
- 최종 UI 응답용 dict 조립

❌ 하지 않는 것
- GPT 호출
- 프롬프트 작성
- 의료 판단
- 병원 검색 로직 직접 구현
"""

from typing import Dict, Any, Optional


class Orchestrator:
    """
    Stateless Coordinator
    세션 상태는 Streamlit / FastAPI 레이어에서 관리
    """

    def __init__(
        self,
        symptom_agent,
        safety_agent,
        explain_agent,
        hospital_search_agent,
        ml_predict_tool,
    ):
        """
        main.py에서 생성한 객체들을 DI로 주입

        - symptom_agent         : GPT (증상 추출)
        - safety_agent          : GPT (응급 판단)
        - explain_agent         : GPT (설명 생성)
        - hospital_search_agent : GPT + Web Search (병원 검색)
        - ml_predict_tool       : XGBoost 예측
        """
        self.symptom_agent = symptom_agent
        self.safety_agent = safety_agent
        self.explain_agent = explain_agent
        self.hospital_search_agent = hospital_search_agent
        self.ml_predict_tool = ml_predict_tool

    # =========================================================
    # 1️⃣ 최초 사용자 증상 입력 처리
    # =========================================================
    def handle_user_input(
        self,
        user_input: str,
        user_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Flow:
        1. 자연어 → 증상 feature 추출 (LLM)
        2. ML Top-K 질병 예측
        3. Safety Agent로 응급 여부 판단
        4. 응급 / 비응급 분기
        """

        # 1️⃣ 증상 추출
        normalized_symptoms = self.symptom_agent.run(user_input)

        # 2️⃣ ML 예측
        topk = self.ml_predict_tool.predict(normalized_symptoms)

        # 3️⃣ 응급 여부 판단
        safety_result = self.safety_agent.run(
            symptoms=normalized_symptoms,
            topk=topk,
        )

        # =====================================================
        # 🚨 응급 상황 → 병원 정보 즉시 제공
        # =====================================================
        if safety_result["is_emergency"]:
            hospital_info = self.hospital_search_agent.run(
                symptoms=normalized_symptoms,
                topk=topk,
                location=user_location,
                emergency=True,
            )

            return {
                "type": "emergency",
                "is_emergency": True,
                "reason": safety_result["reason"],
                "symptoms": normalized_symptoms,
                "topk": topk,
                "hospital_info": hospital_info,
            }

        # =====================================================
        # ✅ 비응급 → 설명 Agent로 전달
        # =====================================================
        explanation = self.explain_agent.run(
            input_data={
                "normalized_symptoms": normalized_symptoms,
                "topk": topk,
                "emergency": False,
            }
        )

        return {
            "type": "explanation",
            "is_emergency": False,
            "symptoms": normalized_symptoms,
            "topk": topk,
            "explanation": explanation,
            "can_request_hospital": True,  # UI에서 버튼 표시용
        }

    # =========================================================
    # 2️⃣ 사용자가 "병원 정보 알려줘"라고 요청했을 때
    # =========================================================
    def handle_hospital_request(
        self,
        symptoms,
        topk,
        user_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        이미 계산된 증상 / Top-K를 기반으로 병원 정보 제공
        """

        hospital_info = self.hospital_search_agent.run(
            symptoms=symptoms,
            topk=topk,
            location=user_location,
            emergency=False,
        )

        return {
            "type": "hospital_info",
            "hospital_info": hospital_info,
        }
