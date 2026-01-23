from typing import List, Dict, Optional, Any
import logging
import json

# 파일 로거
logger = logging.getLogger(__name__)


class HospitalSearchAgent:
    def __init__(self, client):
        # main.py에서 생성한 OpenAI client 객체 주입
        self.client = client

    def run(
        self,
        symptoms: List[str],
        topk: List[Dict],
        location: Optional[str] = None,
        emergency: bool = False,  # orchestrator 호환용 (로직에는 미사용)
    ) -> Dict[str, Any]:

        # ❌ 위치 정보 없으면 검색 불가
        if not location:
            return {
                "status": "error",
                "message": "위치 정보가 없어 병원을 검색할 수 없습니다.",
                "hospitals": [],
            }

        # 1️⃣ Top-K 중 가장 확률 높은 질병 사용
        predicted_disease = topk[0]["label"] if topk else None

        # 2️⃣ 질병 → 진료과 추론
        department = None
        if predicted_disease:
            department = self._infer_department_from_disease(predicted_disease)

        # 3️⃣ 위치 + 진료과 기반 검색 쿼리 (JSON 출력 강제)
        dept_text = f"{department} " if department else ""

        query = (
            f"{location} 근처 {dept_text}병원 3곳을 찾아줘.\n"
            "반드시 아래 JSON 형식으로만 출력해. 다른 설명은 절대 하지 마.\n"
            "{\n"
            '  "hospitals": [\n'
            '    {"name": "...", "address": "...", "phone": "...", "department": "..."},\n'
            '    {"name": "...", "address": "...", "phone": "...", "department": "..."},\n'
            '    {"name": "...", "address": "...", "phone": "...", "department": "..."}\n'
            "  ]\n"
            "}"
        )

        logger.info(f"[HospitalSearchAgent] Query: {query}")

        try:
            # 4️⃣ Web Search Tool 호출
            response = self.client.responses.create(
                model="gpt-5.2",
                tools=[{"type": "web_search"}],
                input=query,
            )

            raw_text = response.output_text
            hospitals = self._parse_hospital_text(raw_text)

            return {
                "status": "ok",
                "emergency": emergency,
                "predicted_disease": predicted_disease,
                "department": department,
                "hospitals": hospitals,
            }

        except Exception as e:
            logger.error(f"[HospitalSearchAgent] Search failed: {e}")
            return {
                "status": "error",
                "message": "병원 검색 중 오류가 발생했습니다.",
                "hospitals": [],
            }

    def _infer_department_from_disease(self, disease: str) -> Optional[str]:
        prompt = (
            f"'{disease}'라는 질병이 있을 때 가야 할 병원 진료과를 "
            "한 단어로만 답해줘.\n"
            "예: 내과, 신경과, 이비인후과, 피부과, 안과, 비뇨의학과 등\n\n"
            "진료과 이름만 출력해."
        )

        try:
            response = self.client.responses.create(
                model="gpt-5.2",
                input=prompt,
            )
            dept = response.output_text.strip()
            logger.info(f"[HospitalSearchAgent] Inferred dept: {dept}")
            return dept

        except Exception as e:
            logger.error(f"[HospitalSearchAgent] Dept inference failed: {e}")
            return None

    def _parse_hospital_text(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        # JSON 앞뒤에 다른 문장이 섞였을 경우 대비
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return [{"raw": text}]

        try:
            data = json.loads(text[start:end + 1])
            hospitals = data.get("hospitals", [])

            # 🔧 줄바꿈/공백 정리
            def _clean(v):
                if v is None:
                    return None
                return " ".join(str(v).split())

            cleaned = []
            for h in hospitals:
                cleaned.append({
                    "name": _clean(h.get("name")),
                    "address": _clean(h.get("address")),
                    "phone": _clean(h.get("phone")),
                    "department": _clean(h.get("department")),
                })

            return cleaned

        except Exception:
            # JSON 파싱 실패 시 원문 반환
            return [{"raw": text}]
