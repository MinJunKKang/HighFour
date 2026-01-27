import streamlit as st
import pandas as pd
from app.main import create_orchestrator

# ================================
# 병원 정보 렌더링 유틸 (그대로 사용 가능)
# ================================
def render_hospitals(hospital_info):
    hospitals = hospital_info.get("hospitals", [])
    if not hospitals:
        st.warning("병원 정보를 찾지 못했습니다.")
        return

    st.subheader("🏥 인근 의료기관")
    for i, h in enumerate(hospitals, 1):
        with st.container(border=True):
            st.markdown(f"### {i}. {h.get('name', '이름 없음')}")
            st.write(f"📍 주소: {h.get('address', '-')}")
            st.write(f"📞 전화번호: {h.get('phone', '-')}")
            if h.get("department"):
                st.write(f"🩺 진료과: {h.get('department')}")

    map_rows = []
    for h in hospitals:
        if h.get("latitude") and h.get("longitude"):
            map_rows.append({"lat": h["latitude"], "lon": h["longitude"]})

    if map_rows:
        st.subheader("🗺️ 병원 위치 지도")
        st.map(pd.DataFrame(map_rows))


def init():
    # orchestrator 1회 생성
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = create_orchestrator()

    # 대화 기록
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 마지막 분석 결과(병원 요청 시 재사용)
    if "last_context" not in st.session_state:
        st.session_state.last_context = None


def add_message(role: str, content: str, payload=None):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "payload": payload or {}
    })


def run():
    st.set_page_config(page_title="AI 건강 정보 안내", page_icon="🩺")
    init()

    st.title("🩺 AI 건강 정보 안내 (비진단)")
    st.caption("증상을 입력하면 관련 정보를 안내합니다. 응급으로 의심되면 즉시 의료기관/119를 이용하세요.")

    # 위치는 사이드바에 두는 게 채팅 UX에 자연스러움
    with st.sidebar:
        st.header("설정")
        user_location = st.text_input("현재 위치(병원 검색용)", placeholder="예: 서울시 강남구")
        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.session_state.last_context = None
            st.rerun() # 페이지 새로고침

    # 기존 대화 렌더링
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

            # payload로 병원 정보가 들어온 메시지면 병원 카드/지도 렌더
            if m["payload"].get("hospital_info"):
                render_hospitals(m["payload"]["hospital_info"])

            # payload로 질문(clarify) 들어온 메시지면 질문 리스트 렌더
            qs = m["payload"].get("questions")
            if qs:
                st.write("아래 중 답할 수 있는 것만 편하게 알려줘 🙂")
                for q in qs:
                    st.write(f"- {q}")

    # 입력창 (채팅)
    user_text = st.chat_input("예: 어제부터 기침이 나고 가슴이 답답해요")

    if user_text:
        add_message("user", user_text)

        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                result = st.session_state.orchestrator.handle_user_input(
                    user_input=user_text,
                    user_location=user_location or None
                )

        # 분기 결과를 “assistant 메시지”로 저장
        if result["type"] in ("clarify", "redirect"):
            msg = result.get("message", "")
            add_message("assistant", msg, payload={
                "questions": result.get("questions", [])
            })
            st.session_state.last_context = None

        elif result.get("is_emergency") is True:
            msg = f"🚨 응급 가능성이 감지되었습니다.\n\n {result.get('reason','-')}\n\n가까운 의료기관 정보를 아래에 표시합니다."
            add_message("assistant", msg, payload={
                "hospital_info": result.get("hospital_info", {})
            })
            st.session_state.last_context = None

        else:
            # 비응급: 설명 + (병원 요청 버튼은 “다음 입력/버튼”으로 처리)
            add_message("assistant", result.get("explanation", ""))

            # 병원 요청을 위해 context 저장
            st.session_state.last_context = {
                "symptoms": result.get("symptoms", []),
                "topk": result.get("topk", []),
                "user_location": user_location or None,
            }

        st.rerun()

    # 채팅 하단에 “병원 보기” 버튼을 상시 두는 방식
    ctx = st.session_state.last_context
    if ctx and (ctx.get("user_location")):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📍 증상 관련 병원 보기", use_container_width=True):
                with st.spinner("병원 검색 중..."):
                    h = st.session_state.orchestrator.handle_hospital_request(
                        symptoms=ctx["symptoms"],
                        topk=ctx["topk"],
                        user_location=ctx["user_location"],
                    )
                add_message("assistant", "가까운 병원 정보를 가져왔어요.", payload={
                    "hospital_info": h.get("hospital_info", {})
                })
                st.session_state.last_context = None
                st.rerun()
        with col2:
            if st.button("계속 대화하기", use_container_width=True):
                pass
    elif ctx and not (ctx.get("user_location")):
        st.info("병원 정보를 보려면 사이드바에 위치를 입력해줘 📍")


if __name__ == "__main__":
    run()