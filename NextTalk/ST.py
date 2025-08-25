import streamlit as st
import ollama
import re  # ← 추가

st.set_page_config(page_title="다음 멘트 추천 (Ollama)", page_icon="💬")
st.title("💬 다음 멘트 추천 (Ollama)")

with st.sidebar:
    st.markdown("### 설정")
    MODEL = st.text_input("모델명", value="llama3.1:8b")  # 대화형은 보통 :instruct 권장
    N = st.slider("제안 개수", 3, 5, 3, 1)
    TEMP = st.slider("temperature", 0.0, 1.5, 0.7, 0.1)
    if st.button("세션 초기화"):
        st.session_state.clear()
        st.rerun()

system_message = f"""
너는 '대화 이어주기 코치'야. 항상 한국어 반말로 짧고 자연스럽게 제안해.
출력 규칙:
- 딱 {N}개 제안만.
- 각 제안은 1문장, 30자 내외.
"""

if "turns" not in st.session_state:
    st.session_state.turns = []

def ensure_model_exists(name: str) -> bool:
    try:
        ollama.show(model=name)
        return True
    except Exception:
        st.error(f"❌ 모델이 없습니다: `{name}`\n- `ollama list`로 확인\n- 필요 시 `ollama pull {name}`")
        return False

# ✅ 로그를 "이름 - 내용"으로 정규화
def normalize_dialog(text: str) -> str:
    """
    다음과 같은 흔한 패턴들을 '이름 - 내용'으로 통일:
    1) [이름] [시간] 내용
    2) [이름] 내용
    3) 이름: 내용
    4) 이름 - 내용
    그 외 매치되지 않으면 공백 정리 후 원문 유지
    """
    patterns = [
        re.compile(r'^\s*\[(?P<name>[^\]]+)\]\s*\[(?P<time>[^\]]*)\]\s*(?P<content>.+)\s*$'),
        re.compile(r'^\s*\[(?P<name>[^\]]+)\]\s*(?P<content>.+)\s*$'),
        re.compile(r'^\s*(?P<name>[^:\-\[\]]+)\s*[:\-]\s*(?P<content>.+)\s*$'),
    ]
    out = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        normalized = None
        for p in patterns:
            m = p.match(line)
            if m:
                name = re.sub(r'\s+', ' ', m.group('name').strip())
                content = re.sub(r'\s+', ' ', m.group('content').strip())
                normalized = f"{name} - {content}"
                break
        out.append(normalized if normalized else re.sub(r'\s+', ' ', line))
    return "\n".join(out)

user_log = st.chat_input("여기에 '대화 로그'를 그대로 붙여넣고 Enter")
if user_log:
    # 1) 정규화해서 화면에 깔끔히 보여주기
    norm_log = normalize_dialog(user_log)
    with st.chat_message("user"):
        st.markdown("**정규화된 대화**")
        st.code(norm_log)

    # 2) 모델에도 정규화된 로그를 전달
    user_content = f"""
이 대화에서 상대방과 대화를 자연스럽게 이어가기 위해, 다음에 어떤 말을 하면 좋을지 한국어로 추천해줘.

[대화 로그 시작]
{norm_log}
[대화 로그 끝]

조건:
- 반말, 공감, 가벼운 유머 허용
- {N}개 제안, 각 1문장/30자 내외
"""

    with st.chat_message("assistant"):
        if not ensure_model_exists(MODEL.strip()):
            st.stop()

        try:
            res = ollama.chat(
                model=MODEL.strip(),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                options={"temperature": float(TEMP), "num_ctx": 4096},
            )
            text = res["message"]["content"].replace("\\n", "\n").strip()

            # 라인 정리
            lines = [
                ln.strip(" -•0123456789.").strip()
                for ln in text.splitlines() if ln.strip()
            ]
            lines = lines[:N] if len(lines) >= N else lines

            if not lines:
                st.markdown("생성된 제안이 없네. 로그를 조금 더 붙여줘! 😅")
            else:
                for i, ln in enumerate(lines, 1):
                    st.markdown(f"{i}. {ln}")

            st.caption(f"모델: **{res.get('model', MODEL)}** | temp={TEMP}")
            st.session_state.turns.append((norm_log, lines))

        except Exception as e:
            st.error(f"모델 호출 실패: {e}\n- `ollama serve`가 실행 중인지와 모델명을 확인해줘.")

# (선택) 이전 추천 보기
if st.session_state.turns:
    with st.expander("이전 추천 보기"):
        for i, (logs, suggs) in enumerate(reversed(st.session_state.turns), 1):
            st.markdown(f"**#{i} 입력 로그 (정규화)**")
            st.code(logs)
            if suggs:
                for j, ln in enumerate(suggs, 1):
                    st.markdown(f"{j}. {ln}")

#streamlit run ST.py 으로 실행하면 됨
