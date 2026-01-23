import streamlit as st
import sys
import os

# =========================================================
# 1. 경로 설정 (Import Error 방지 최적화)
# =========================================================
# 현재 파일(streamlit_app.py)의 폴더와 프로젝트 루트(HighFour)를 탐색 경로에 등록합니다.
current_dir = os.path.dirname(os.path.abspath(__file__)) # ui 폴더
project_root = os.path.abspath(os.path.join(current_dir, "..")) # HighFour 폴더

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# =========================================================
# 2. 오케스트레이터(팀장님) 연결 및 세션 초기화
# =========================================================
try:
    # app.main에서 가져오되, 환경에 따라 main에서 직접 시도할 수 있도록 합니다.
    try:
        from app.main import create_orchestrator
    except ImportError:
        import sys
        sys.path.append(os.path.join(project_root, "app"))
        from main import create_orchestrator
    
    # 앱이 켜질 때 딱 한 번만 오케스트레이터를 생성하여 세션에 저장합니다.
    if "orchestrator" not in st.session_state:
        with st.spinner("전문가 시스템을 연결 중입니다..."):
            st.session_state.orchestrator = create_orchestrator()
            
except Exception as e:
    st.error(f"❌ 시스템 연결 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
    st.info("💡 app/main.py 파일이 최신 상태인지, 또는 __init__.py 파일이 있는지 확인해주세요.")
    st.stop()

# 페이지 전환을 위한 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "input"

# =========================================================
# 3️⃣ 화면 로직 (페이지 전환)
# =========================================================

# --- 1단계: 증상 입력 화면 ---
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
            with st.spinner("AI 전문가 팀이 분석 중입니다... 잠시만 기다려주세요."):
                # 팀장님(오케스트레이터)에게 분석 요청
                result = st.session_state.orchestrator.handle_user_input(
                    user_input=user_input,
                    user_location=user_location or None
                )
                st.session_state.result = result
                st.session_state.page = "result"
                st.rerun()

# --- 2단계: 분석 결과 화면 (응급/비응급 분기) ---
elif st.session_state.page == "result":
    res = st.session_state.result
    st.title("📋 분석 결과")

    # 🚨 응급 케이스 (is_emergency가 True인 경우)
    if res.get("is_emergency") is True:
        st.error("🚨 응급 가능성이 감지되었습니다!")
        st.markdown(f"### **판단 사유**\n{res.get('reason')}")
        
        st.subheader("🏥 인근 의료기관 정보")
        st.write(res.get("hospital_info"))
        
        st.warning("※ 위 정보는 AI의 분석 결과이며, 실제 의사의 진단을 대신할 수 없습니다. 즉시 119에 연락하거나 가까운 응급실을 방문하세요.")

    # ✅ 비응급 케이스
    else:
        st.success("✅ 건강 정보 분석이 완료되었습니다.")
        st.markdown(f"### **안내 내용**\n{res.get('explanation')}")

        # 병원 추천 버튼 활성화 여부 확인
        if res.get("can_request_hospital"):
            st.divider()
            st.info("💡 증상과 관련된 가까운 병원 정보를 확인하시겠습니까?")
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

# --- 3단계: 병원 상세 정보 화면 ---
elif st.session_state.page == "hospital":
    st.title("🏥 관련 병원 상세 정보")
    
    h_info = st.session_state.hospital_result.get("hospital_info")
    st.info("분석된 증상에 따라 방문을 권장드리는 의료기관입니다.")
    st.write(h_info)

    if st.button("메인 화면으로"):
        st.session_state.page = "input"
        st.rerun()