# agents/orchestrator.py

from typing import Dict, Any, Optional


class Orchestrator:
    """
    Stateless Coordinator
    - 파이프라인 제어만 담당
    - 판단/설명/검색 로직 없음
    """

    def __init__(
        self,
        symptom_agent,
        safety_agent,
        explain_agent,
        hospital_search_agent,
        ml_predict_tool,
        clarify_agent,
    ):
        self.symptom_agent = symptom_agent
        self.safety_agent = safety_agent
        self.explain_agent = explain_agent
        self.hospital_search_agent = hospital_search_agent
        self.ml_predict_tool = ml_predict_tool
        self.clarify_agent = clarify_agent

    # =========================================================
    # 1️⃣ 최초 사용자 입력 처리
    # =========================================================
    def handle_user_input(
        self,
        user_input: str,
        user_location: Optional[str] = None,
    ) -> Dict[str, Any]:

        # 1️⃣ 증상 추출 (LLM)
        normalized_symptoms = self.symptom_agent.run(user_input)

        print("=== [DEBUG] SymptomAgent Output ===")
        print(type(normalized_symptoms), normalized_symptoms)
        print("==================================")

        if not normalized_symptoms:
            cr = self.clarify_agent.run(user_input)
            return {
                "type": cr["route"],
                "is_emergency": False,
                "symptoms": [],
                "message": cr["message"],
                "questions": cr.get("questions", []),
                "can_request_hospital": False,
            }

        # 2️⃣ ML 질병 후보 예측 (label만 의미 있음)
        topk_raw = self.ml_predict_tool.predict(normalized_symptoms)

        # 👉 ExplainAgent용: label만 전달
        topk_labels = [d["label"] for d in topk_raw]

        # 3️⃣ Safety 판단 (GPT가 점수 계산)
        safety_result = self.safety_agent.run(
            symptoms=normalized_symptoms,
            topk=topk_labels,
        )

        # =====================================================
        # 🚨 응급 분기
        # =====================================================
        if safety_result["is_emergency"]:
            hospital_info = self.hospital_search_agent.run(
                symptoms=normalized_symptoms,
                topk=topk_labels,
                location=user_location,
                emergency=True,
            )

            return {
                "type": "emergency",
                "is_emergency": True,
                "reason": safety_result["reason"],
                "symptoms": normalized_symptoms,
                "topk": topk_labels,
                "hospital_info": hospital_info,
            }

        # =====================================================
        # ✅ 비응급 → ExplainAgent
        # =====================================================
        explanation = self.explain_agent.run(
            input_data={
                "symptoms": normalized_symptoms,
                "topk": topk_labels,  # 🔥 점수 없음
            }
        )

        return {
            "type": "explanation",
            "is_emergency": False,
            "symptoms": normalized_symptoms,
            "topk": topk_labels,
            "explanation": explanation,
            "can_request_hospital": True,
        }

    # =========================================================
    # 2️⃣ 병원 정보 요청
    # =========================================================
    def handle_hospital_request(
        self,
        symptoms,
        topk,
        user_location: Optional[str] = None,
    ) -> Dict[str, Any]:

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
