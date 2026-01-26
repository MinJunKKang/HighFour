from typing import List, Dict, Optional, Any, Tuple
import json
import re


class HospitalSearchAgent:
    """
    - Orchestrator/Streamlit 수정 없이 사용 (run 시그니처 유지)
    - symptoms 토큰이 다양해도(영/한, 문장형, 복합형) 정신과 신호를 robust하게 감지
    - 정신과면 '병원' 대신 '정신건강의학과의원/정신과의원/멘탈클리닉' 위주로 검색
    - emergency 판단은 하지 않음 (입력 emergency 플래그만 반영)
    - 출력 스키마는 Streamlit 렌더링에 맞춰 고정
    """

    # 정신건강 신호 패턴(부분일치)
    # - 토큰이 "depressive or psychotic symptoms" 같이 길어도 잡히도록 stem/substring 위주
    MENTAL_PATTERNS = [
        # EN
        r"\bdepress",         # depression, depressive...
        r"\banx",             # anxiety, anxious...
        r"\bpanic",           # panic
        r"\bptsd\b",
        r"\bocd\b",
        r"\bbipolar\b",
        r"\bschizo",          # schizophrenia...
        r"\bpsych",           # psychotic, psychiatry...
        r"\bsuicid",          # suicidal...
        r"\bself[- ]?harm\b",
        r"\binsomnia\b",
        r"\bstress\b",
        r"\bphobia\b",
        r"\badhd\b",
        r"\btrauma\b",

        # KO
        r"우울",
        r"불안",
        r"공황",
        r"대인",
        r"사회불안",
        r"강박",
        r"수면장애|불면",
        r"트라우마",
        r"자해",
        r"자살",
        r"환청|환각|망상",
        r"조울|양극성",
    ]

    # 정신과 “검색 키워드” (병원보다 의원/클리닉이 잘 잡힘)
    MENTAL_FACILITY_QUERY = "정신건강의학과의원 정신과의원 멘탈클리닉 마음클리닉 상담 클리닉"

    # 응급(입력 emergency=True일 때만) 검색 키워드
    EMERGENCY_FACILITY_QUERY = "응급실 24시간 응급의료센터 권역응급의료센터"

    # 일반 검색 키워드
    GENERAL_FACILITY_QUERY = "병원 의원"

    def __init__(self, client):
        self.client = client
        self._mental_regex = [re.compile(p, re.IGNORECASE) for p in self.MENTAL_PATTERNS]

    def run(
        self,
        symptoms: List[str],
        topk: List[str],  # 시그니처 유지(참고용으로만 사용)
        location: Optional[str] = None,
        emergency: bool = False,
    ) -> Dict[str, Any]:

        if not location or not str(location).strip():
            return {
                "status": "error",
                "message": "위치 정보가 없습니다. 예: '서울시 강남구'처럼 입력해주세요.",
                "hospitals": [],
                "raw": "",
            }

        # 1) 토큰 정규화 + 정신/비정신 분리
        mental_syms, other_syms = self._split_symptoms(symptoms)

        is_mental = bool(mental_syms)

        # 2) 검색 라우팅 (응급 판단 X, emergency 입력만 반영)
        if emergency:
            department = "응급의학과"
            facility_query = self.EMERGENCY_FACILITY_QUERY
            note = "emergency=True(외부 판단) → 응급실 중심 검색"
        elif is_mental:
            department = "정신건강의학과"
            facility_query = self.MENTAL_FACILITY_QUERY
            note = f"정신과 신호 토큰 감지 → 정신건강의학과(의원/클리닉) 우선"
        else:
            department = ""
            facility_query = self.GENERAL_FACILITY_QUERY
            note = "증상 토큰 기반 일반 의료기관 검색(관련 진료과 표기된 곳 우선)"

        # 3) 검색에 넣을 증상 키워드: 정신 토큰을 앞에 배치(검색 우선순위 강화)
        ordered_tokens = (mental_syms + other_syms)[:10]
        symptom_text = self._sanitize(", ".join(ordered_tokens)) if ordered_tokens else ""

        # (참고) topk도 조금만 힌트로 (매핑 X)
        topk_hint = self._sanitize(", ".join([str(x) for x in (topk or [])[:3]]))

        # 4) 프롬프트 (JSON만 출력 강제 + 정신과는 의원/클리닉 우선 명시)
        query = (
            f"{self._sanitize(location)} 근처에서 아래 증상/상황에 맞는 의료기관 3곳을 찾아줘.\n"
            f"- 증상 토큰(우선순위 반영): {symptom_text or '-'}\n"
            f"- (참고) 예측 후보: {topk_hint or '-'}\n"
            f"- 우선 검색 키워드: {facility_query}\n"
            f"- 참고: {note}\n\n"
            "조건:\n"
            "1) 반드시 실제 존재하는 의료기관만(가능하면 지도/홈페이지/전화번호 근거)\n"
            "2) 같은 곳 중복 금지\n"
            "3) address는 꼭 포함, phone 없으면 '-'로\n"
            "4) latitude/longitude는 가능하면 채우고 모르면 null\n"
            "5) 정신건강의학과 케이스면 '병원'보다 '의원/클리닉'을 우선 포함\n\n"
            "아래 JSON 한 덩어리만 출력(다른 텍스트 금지):\n"
            "{\n"
            '  "hospitals": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "address": "",\n'
            '      "phone": "-",\n'
            '      "latitude": null,\n'
            '      "longitude": null,\n'
            f'      "department": "{department}"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        response = self.client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            input=query,
        )

        parsed = self._parse(response.output_text)
        return self._postprocess(
            parsed,
            fallback_department=department or ("정신건강의학과" if is_mental else ""),
        )

    # =========================================================
    # Token handling
    # =========================================================
    def _split_symptoms(self, symptoms: List[str]) -> Tuple[List[str], List[str]]:
        """
        symptoms 토큰이 다양하게 들어와도:
        - 소문자/공백 정리
        - 불필요한 기호 제거
        - 정신과 신호(부분일치)면 mental로 분리
        """
        mental = []
        other = []

        for s in symptoms or []:
            raw = str(s).strip()
            if not raw:
                continue

            norm = self._normalize_token(raw)

            # 너무 짧은 토큰은 의미 없으니 패스(예: '-', 'a')
            if len(norm) < 2:
                continue

            if self._is_mental_token(norm):
                mental.append(norm)
            else:
                other.append(norm)

        # 중복 제거(순서 유지)
        mental = self._dedupe_keep_order(mental)
        other = self._dedupe_keep_order(other)

        return mental, other

    def _is_mental_token(self, token: str) -> bool:
        # token에 정신과 패턴이 하나라도 매치되면 mental
        for rx in self._mental_regex:
            if rx.search(token):
                return True
        return False

    def _normalize_token(self, token: str) -> str:
        # 대소문자 통일 + 공백/구두점 정리(부분일치 용)
        t = token.lower()
        t = t.replace("_", " ")
        t = re.sub(r"[/|]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        # JSON 깨뜨릴 수 있는 기호 제거(검색 품질 크게 영향 없음)
        t = t.replace("{", "").replace("}", "").replace("`", "").replace('"', "")
        return t

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            k = x
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    def _sanitize(self, s: str) -> str:
        s = str(s).replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = s.replace("{", "").replace("}", "").replace("`", "")
        return re.sub(r"\s+", " ", s).strip()

    # =========================================================
    # JSON parsing (robust)
    # =========================================================
    def _parse(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"status": "partial", "message": "빈 응답", "hospitals": [], "raw": ""}

        cleaned = re.sub(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            r"\1",
            text,
            flags=re.IGNORECASE,
        ).strip()

        obj = self._extract_first_json_object(cleaned)
        if isinstance(obj, dict):
            obj.setdefault("hospitals", [])
            obj.setdefault("raw", "")
            return obj

        return {"status": "partial", "message": "JSON 파싱 실패", "hospitals": [], "raw": text}

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        return None
        return None

    # =========================================================
    # Postprocess for Streamlit schema + dedupe
    # =========================================================
    def _postprocess(self, parsed: Dict[str, Any], fallback_department: str) -> Dict[str, Any]:
        hospitals = parsed.get("hospitals", [])
        raw = parsed.get("raw", "")

        out: List[Dict[str, Any]] = []
        seen = set()

        if not isinstance(hospitals, list):
            hospitals = []

        for h in hospitals:
            if not isinstance(h, dict):
                continue

            name = str(h.get("name") or "").strip()
            address = str(h.get("address") or "").strip()
            if not name or not address:
                continue

            key = (self._norm_key(name), self._norm_key(address))
            if key in seen:
                continue
            seen.add(key)

            out.append({
                "name": name,
                "address": address,
                "phone": str(h.get("phone") or "-").strip() or "-",
                "latitude": self._to_float_or_none(h.get("latitude")),
                "longitude": self._to_float_or_none(h.get("longitude")),
                "department": str(h.get("department") or fallback_department).strip(),
            })

        status = parsed.get("status") or ("ok" if out else "partial")
        message = parsed.get("message")
        if message is None:
            message = "" if out else "병원 정보를 충분히 찾지 못했습니다. 위치를 더 구체적으로 입력해 보세요."

        return {
            "status": status,
            "message": message,
            "hospitals": out[:3],
            "raw": raw if status != "ok" else "",
        }

    def _to_float_or_none(self, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _norm_key(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s가-힣]", "", s)
        return s.strip()
