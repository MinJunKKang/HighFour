from typing import List, Dict, Optional, Any
import json


class HospitalSearchAgent:
    def __init__(self, client):
        self.client = client

    def run(
        self,
        symptoms: List[str],
        topk: List[str],
        location: Optional[str] = None,
        emergency: bool = False,
    ) -> Dict[str, Any]:

        if not location:
            return {
                "status": "error",
                "message": "위치 정보가 없습니다.",
                "hospitals": [],
            }

        predicted_disease = topk[0] if topk else None

        department = None
        if predicted_disease:
            department = self._infer_department(predicted_disease)

        query = (
            f"{location} 근처 {department or ''} 병원 3곳 찾아줘.\n"
            "아래 JSON 형식으로만 출력:\n"
            '{ "hospitals": [ { "name": "", "address": "", "phone": "" } ] }'
        )

        response = self.client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            input=query,
        )

        return self._parse(response.output_text)

    def _infer_department(self, disease: str) -> str:
        response = self.client.responses.create(
            model="gpt-5.2",
            input=f"{disease}에 가야 할 진료과 한 단어로 알려줘",
        )
        return response.output_text.strip()

    def _parse(self, text: str) -> Dict:
        try:
            return json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception:
            return {"hospitals": [], "raw": text}
