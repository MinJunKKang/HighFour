import streamlit as st
import pandas as pd
from app.main import create_orchestrator


# ================================
# 🏥 병원 정보 렌더링 유틸
# ================================
def render_hospitals(hospital_info):
    hospitals = hospital_info.get("hospitals", [])

    if not hospitals:
        st.warning("병원 정보를 찾지 못했습니다.")
        return

    # ---------- 텍스트 카드 ----------
    st.subheader("🏥 인근 의료기관")

    for i, h in enumerate(hospitals, 1):
        with st.container(border=True):
            st.markdown(f"### {i}. {h.get('name', '이름 없음')}")
            st.write(f"📍 주소: {h.get('address', '-')}")
            st.write(f"📞 전화번호: {h.get('phone', '-')}")
            if h.get("department"):
                st.write(f"🩺 진료과: {h.get('department')}")

    # ---------- 지도 ----------
    map_rows = []
    for h in hospitals:
        if h.get("latitude") and h.get("longitude"):
            map_rows.append({
                "lat": h["latitude"],
                "lon": h["longitude"],
                "name": h.get("name", "")
            })

    if map_rows:
        st.subheader("🗺️ 병원 위치 지도")
        df = pd.DataFrame(map_rows)
        st.map(df)


# ================================
# 🚀 Streamlit App
# ================================
def run():
    # 1️⃣ 오케스트레이터 초기화
    if "orchestrator" not in st.session_state:
        with st.spinner("전문가 시스템을 연결 중입니다..."):
            st.session_state.orchestrator = create_orchestrator()

    if "page" not in st.session_state:
        st.session_state.page = "input"

    # ============================
    # 입력 화면
    # ============================
    if st.session_state.page == "input":
        st.title("🩺 AI 건강 정보 안내 (비진단)")
        st.write("알려주시는 증상을 바탕으로 AI가 관련 정보를 분석합니다.")

        user_input = st.text_area(
            "증상을 자연스럽게 입력해주세요",
            placeholder="예: 어제부터 왼쪽 가슴이 찌릿하고 숨쉬기가 불편해요",
            height=150
        )

        user_location = st.text_input(
            "현재 위치 (선택)",
            placeholder="예: 서울시 강남구"
        )

        if st.button("분석 시작", type="primary"):
            if not user_input.strip():
                st.warning("증상을 입력해주세요.")
            else:
                with st.spinner("AI 전문가 팀이 분석 중입니다..."):
                    result = st.session_state.orchestrator.handle_user_input(
                        user_input=user_input,
                        user_location=user_location or None
                    )
                    st.session_state.result = result
                    st.session_state.page = "result"
                    st.rerun()

    # ============================
    # 결과 화면
    # ============================
    elif st.session_state.page == "result":
        res = st.session_state.result
        st.title("📋 분석 결과")

        # 🚨 응급
        if res.get("is_emergency") is True:
            st.error("🚨 응급 가능성이 감지되었습니다!")
            st.markdown(f"### 판단 사유\n{res.get('reason')}")

            hospital_info = res.get("hospital_info", {})
            render_hospitals(hospital_info)

            st.warning("※ 본 정보는 의료 진단이 아닙니다. 즉시 의료기관을 방문하세요.")

        # ✅ 비응급
        else:
            st.success("✅ 건강 정보 분석이 완료되었습니다.")
            st.markdown(f"### 안내 내용\n{res.get('explanation')}")

            if res.get("can_request_hospital"):
                st.divider()
                if st.button("📍 관련 병원 정보 보기"):
                    with st.spinner("가까운 병원을 찾는 중입니다..."):
                        h_result = st.session_state.orchestrator.handle_hospital_request(
                            symptoms=res.get("symptoms"),
                            topk=res.get("topk"),
                            user_location=None
                        )
                        st.session_state.hospital_result = h_result
                        st.session_state.page = "hospital"
                        st.rerun()

        if st.button("처음으로 돌아가기"):
            st.session_state.page = "input"
            st.rerun()

    # ============================
    # 병원 전용 페이지
    # ============================
    elif st.session_state.page == "hospital":
        st.title("🏥 관련 병원 상세 정보")

        h_info = st.session_state.hospital_result.get("hospital_info", {})
        render_hospitals(h_info)

        if st.button("메인 화면으로"):
            st.session_state.page = "input"
            st.rerun()
