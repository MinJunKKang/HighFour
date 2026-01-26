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
        intent_guard_agent,
        symptom_agent,
        safety_agent,
        explain_agent,
        hospital_search_agent,
        ml_predict_tool,
    ):
        self.intent_guard_agent = intent_guard_agent
        self.symptom_agent = symptom_agent
        self.safety_agent = safety_agent
        self.explain_agent = explain_agent
        self.hospital_search_agent = hospital_search_agent
        self.ml_predict_tool = ml_predict_tool

    # =========================================================
    # 1️⃣ 최초 사용자 입력 처리
    # =========================================================
    def handle_user_input(
        self,
        user_input: str,
        user_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        
        # 0️⃣ Intent Guard (의도 먼저)
        ig = self.intent_guard_agent.run(user_input)
        intent = ig.get("intent")

        # 의료 의도 아님 → redirect (여기서 끝)
        if intent == "redirect":
            return {
                "type": "redirect",
                "is_emergency": False,
                "symptoms": [],
                "message": ig.get("message", ""),
                "questions": [],
                "can_request_hospital": False,
        }

        # 의료 의도는 있는데 너무 모호 → clarify (여기서 끝)
        if intent == "clarify":
            return {
                "type": "clarify",
                "is_emergency": False,
                "symptoms": [],
                "message": ig.get("message", ""),
                "questions": ig.get("questions", []),
                "can_request_hospital": False,
        }

        # 여기부터는 medical intent 확정
        # 1️⃣ 증상 추출 (LLM)
        normalized_symptoms = self.symptom_agent.run(user_input)

        print("=== [DEBUG] SymptomAgent Output ===")
        print(type(normalized_symptoms), normalized_symptoms)
        print("==================================")

        # medical인데도 증상 추출 실패하면 → clarify로 강제 전환 (여기서 끝)
        if not normalized_symptoms:
            return {
                "type": "clarify",
                "is_emergency": False,
                "symptoms": [],
                "message": "말씀해주신 내용만으로는 증상을 구체적으로 파악하기 어려워요. 아래 질문에 답해주면 더 정확히 안내할게요.",
                "questions": [
                    "어느 부위가 어떻게 아프신가요? (예: 팔/손목, 찌릿/욱신/쑤심)",
                    "언제부터 시작됐고, 다치거나 넘어지신 적이 있나요?",
                    "붓기/멍/변형/움직이기 어려움/저림이 있나요?"
            ],
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
                "reason": safety_result.get("user_reason", "응급 상황이 감지되었습니다."),
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
