import streamlit as st
from app.main import create_orchestrator

def run():
    # =========================================================
    # 1. 오케스트레이터 세션 초기화
    # =========================================================
    if "orchestrator" not in st.session_state:
        with st.spinner("전문가 시스템을 연결 중입니다..."):
            st.session_state.orchestrator = create_orchestrator()

    # 페이지 전환 세션 초기화
    if "page" not in st.session_state:
        st.session_state.page = "input"

    # =========================================================
    # 2. 화면 로직
    # =========================================================

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

    elif st.session_state.page == "result":
        res = st.session_state.result
        st.title("📋 분석 결과")

        if res.get("is_emergency") is True:
            st.error("🚨 응급 가능성이 감지되었습니다!")
            st.markdown(f"### **판단 사유**\n{res.get('reason')}")

            st.subheader("🏥 인근 의료기관 정보")
            st.write(res.get("hospital_info"))

            st.warning("※ 위 정보는 AI의 분석 결과이며 실제 진단을 대체하지 않습니다.")

        else:
            st.success("✅ 건강 정보 분석이 완료되었습니다.")
            st.markdown(f"### **안내 내용**\n{res.get('explanation')}")

            if res.get("can_request_hospital"):
                st.divider()
                if st.button("📍 관련 병원 정보 보기"):
                    with st.spinner("가까운 병원을 찾는 중입니다..."):
                        h_result = st.session_state.orchestrator.handle_hospital_request(
                            symptoms=res.get("symptoms"),
                            topk=res.get("topk", 3)
                        )
                        st.session_state.hospital_result = h_result
                        st.session_state.page = "hospital"
                        st.rerun()

        if st.button("처음으로 돌아가기"):
            st.session_state.page = "input"
            st.rerun()

    elif st.session_state.page == "hospital":
        st.title("🏥 관련 병원 상세 정보")

        h_info = st.session_state.hospital_result.get("hospital_info")
        st.write(h_info)

        if st.button("메인 화면으로"):
            st.session_state.page = "input"
            st.rerun()
