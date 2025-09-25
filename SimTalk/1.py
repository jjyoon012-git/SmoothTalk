# 소개팅 시뮬레이터 인데 왜 오류가 날까요? => 괜찮아졌습니다! (2025-09-25업뎃)
# 실행: streamlit run 1.py
# 필요: pip install streamlit, 그리고 로컬에서 ollama pull gemma3:4b (또는 원하는 모델)

import json
import re
import math # 인데 안 씀 아직
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st


try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    print("올라마없")
    OLLAMA_AVAILABLE = False

# UI 설정

st.set_page_config(page_title="Smooth Talk: 소개팅 시뮬레이터", page_icon="💘") # 아이콘 괜찮나요? 지피티가 추천해줌
st.title("💘 Smooth Talk: 소개팅 시뮬레이터")

with st.sidebar:
    st.markdown("### ⚙️ 설정")
    MODEL = st.text_input("모델명 (Ollama)", value="gemma3:4b", help="예: gemma3:4b, llama3.1:8b, qwen2.5:7b 등")
    TEMP = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1) # 해석: 0부터 쩜오까지, 중간값은 0.7, 스텝은 0.1
    ROUNDS = st.slider("라운드 수", 3, 12, 6, 1) # 해석: 3부터 12까지, 중간값은 6, 스텝은 1
    DIFF = st.selectbox("난이도", ["쉬움", "보통", "어려움"], index=1)
    SCENARIO = st.selectbox(
        "시나리오",
        [
            "첫 만남 (카페)",
            "첫 만남 (레스토랑/저녁 식사)",
            "첫 만남 (영화관/데이트 코스)",
            "첫 만남 (공원/야외 산책)", # 보통 소개팅 어디서 하나요? 전 모르겠습니다. 소개팅 전문가 추가해주시죠
        ],
    )
    EVAL_MODE = st.radio(
        "평가 엔진",
        ["자동", "LLM", "휴리스틱"],
        index=0,
        help="자동: LLM 사용, 실패 시 휴리스틱"
    )
    st.caption("※ Ollama가 실행 중이어야 모델 대화/평가가 동작합니다.")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    reset = st.button("🔄 새 시뮬레이션 시작", use_container_width=True)
with col_btn2:
    export = st.button("💾 기록 내보내기 (JSON)", use_container_width=True) 


# 세션 상태 초기화

def init_state():
    st.session_state.messages = []  # {"role": "user"/"assistant"/"system", "content": str}
    st.session_state.turn = 0
    st.session_state.max_rounds = ROUNDS
    st.session_state.scores: List[Dict[str, Any]] = []
    st.session_state.summary = None
    st.session_state.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.finished = False

if "messages" not in st.session_state or reset:
    init_state()

#
# 시나리오 힌트. 추후에 더 다양하게 늘려봅시다.
# 
def scenario_hint(s: str) -> str:
    if "카페" in s: return "첫 만남 특유의 가벼운 탐색, 취향·일상 질문 위주."
    if "레스토랑" in s: return "정중하고 차분한 톤, 감정·가치관 질문 1개 포함."
    if "영화관" in s: return "경험 공유와 감상 질문 1개 포함."
    if "산책" in s: return "자연스러운 관찰 코멘트 + 가벼운 질문."
    return ""

# 
# 시스템/시나리오 프롬프트
# 
def build_system_prompt():
    return f"""You are a dating simulation partner in Korean.
- Your role: 상대역(NPC)로 자연스럽게 대화한다.
- Style: 공손하고 따뜻하며, 솔직하고 유머는 가볍게.
- Length: 2~4문장 위주로 답하되, 질문 1개를 곁들여 대화가 이어지게 한다.
- Avoid: 장문 독백, 과도한 사담, 과한 칭찬, 모호한 답변.
- Keep boundaries: 과한 신체 접촉/사생활 침해성 질문은 부드럽게 선을 긋는다.
- Scenario: {SCENARIO} — Hint: {scenario_hint(SCENARIO)}
- Difficulty: {DIFF} (난이도가 높을수록 말을 아끼고, 되묻거나 테스트하는 질문을 섞는다)
- Language: 한국어로만 반응한다.
"""

def ensure_system_message():
    # 최초 1회만 시스템 메시지 삽입
    if not any(m.get("role") == "system" for m in st.session_state.messages):
        st.session_state.messages.insert(0, {"role": "system", "content": build_system_prompt()})

ensure_system_message()

# 
# NPC 응답 생성 (Ollama)
# 
def npc_reply(messages: List[Dict[str, str]]) -> str:
    """Ollama를 사용해 NPC 응답 생성. 실패 시 폴백 문구."""
    if not OLLAMA_AVAILABLE:
        return "저는 시뮬레이터 NPC예요. (Ollama 미동작) — 요즘 어떤 취미 즐기세요?"

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": TEMP},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        hint = f"모델이 로컬에 없을 수 있어요. 터미널에서 `ollama pull {MODEL}` 후 다시 시도해보세요."
        return f"(모델 오류) 간단히 이어갈게요. 요즘 어떻게 지내셨어요?\n\n— 에러: {e}\n— 힌트: {hint}"

# 
# 평가 프롬프트 (항상 사용자만 평가)
# 
RUBRIC = {
    "공감": 0.25,  # 높을수록 좋음
    "호기심": 0.20,  # 높을수록 좋음
    "명료성": 0.20,  # 높을수록 좋음
    "정중함": 0.20,   # 높을수록 좋음
    "레드플래그": 0.15 # 높을수록 나쁨
}

def build_eval_prompt(user_msg: str, npc_msg: str, difficulty: str) -> str:
    return f"""너는 소개팅 코치야. 아래 '평가대상'의 발화만 평가해.
오직 평가대상의 문장만 점수화하고 피드백을 작성해.
'상대 발화'는 맥락 참고용일 뿐이고, 강점/개선/팁에 인용하거나 근거로 사용하지 마.

[평가대상] 나(사용자)

[평가대상 발화]
{user_msg}

[상대 발화(참고용, 인용 금지)]
{npc_msg}

[평가 기준]
- 공감(0~10): 상대의 감정/내용을 이해하고 반영했는가?
- 호기심(0~10): 자연스러운 관심 질문이 있는가?
- 명료성(0~10): 구체적이고 분명한가?
- 경계선(0~10): 과하지 않고 예의 바른가?
- 레드플래그(0~10): 무례/과몰입/사생활침해/거짓말/셀프디스 등(높을수록 나쁨)

[난이도]
{difficulty}

[출력 형식(JSON strict)]
{{
  "scores": {{
    "공감": 0-10,
    "호기심": 0-10,
    "명료성": 0-10,
    "경계선": 0-10,
    "레드플래그": 0-10
  }},
  "feedback": {{
    "strengths": ["짧은 문장", "최대 3개"],
    "improvements": ["짧은 문장", "최대 3개"],
    "tip": "다음 턴에 바로 쓸 한 문장 코칭"
  }}
}}
반드시 유효한 JSON만 출력해. 주석/설명/추가 텍스트 금지.
"""

def safe_json_parse(text: str) -> Dict[str, Any]:
    # ```json ... ``` 블록 제거 및 트레일링 처리
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        cleaned = re.sub(r"(\d+)\s*-\s*10", "10", cleaned)  # "0-10" 오인치환 방지
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

# 휴리스틱 평가: 이거 거의 옛날 파파고 번역기 수준. 규칙기반 평가입니다. => 이거 자동 / LLM / 휴리스틱으로 설정해뒀는데, LLM 사용 권장합니다.

def heuristic_evaluate(user_msg: str) -> Dict[str, Any]:
    txt = (user_msg or "").strip()
    low = txt.lower()

    tokens = re.findall(r"[가-힣A-Za-z0-9]+", low)
    n_tokens = len(tokens)
    n_chars = len(txt)

    n_q = txt.count("?")
    n_exc = txt.count("!")
    n_ellipsis = txt.count("…") + txt.count("...")

    all_caps_ratio = 0.0
    cap_tokens = [t for t in re.findall(r"[A-Za-z]+", txt) if len(t) >= 3]
    if cap_tokens:
        all_caps = [t for t in cap_tokens if t.isupper()]
        all_caps_ratio = len(all_caps) / len(cap_tokens)

    emoji_like = bool(re.search(r"[😊😂🤣😍😘🥰🙌👍✨❤💕💘😉😅🙏]|[^\w\s][\)D]|[~]{2,}", txt))

    is_very_short = n_chars < 8
    is_long = n_chars > 120
    has_sentences = len(re.split(r"[.!?？！。…]+", txt)) >= 2

    softeners = ["혹시", "괜찮다면", "실례지만", "가능할까요", "바쁘시면", "천천히", "부탁", "고맙", "감사", "죄송", "미안"]
    empathy_pos = ["좋", "재밌", "대단", "멋지", "축하", "응원", "이해", "그렇군", "알겠", "수고", "고생"]
    empathy_reflect = ["말씀", "얘기", "이야기", "포인트", "공감", "맞아요", "맞다", "그렇죠"]

    curiosity_words = ["왜", "언제", "어디", "무엇", "무슨", "어떤", "어떻게", "어때", "가능할까요", "물어봐도"]
    question_suffix = bool(re.search(r"(나요|니요|죠\?|지요\?)$", low))

    boundary_bad = ["카톡아이디", "카카오톡", "집주소", "만나자지금", "지금만나", "번호줘", "연락처줘", "숙소", "방잡", "술자리 강요"]
    red_flags = ["싫", "꺼져", "닥쳐", "미친", "뭐래", "한남", "김치년", "멍청", "병신", "야 ", "딱딱", "담배냄새", "살쪘", "돈자랑"]

    def clamp01(x): return max(0.0, min(1.0, x))
    def to10(x): return round(10 * clamp01(x), 1)

    # 공감
    emp_score = 0.0
    emp_score += 0.6 * sum(1 for k in empathy_pos if k in low)
    emp_score += 0.7 * sum(1 for k in empathy_reflect if k in low)
    emp_score += 0.5 * sum(1 for k in softeners if k in low)
    emp_score += 0.4 if emoji_like else 0.0
    emp_score -= 0.2 * max(0, n_exc - 2)

    # 호기심
    cur_score = 0.0
    cur_score += 0.9 if n_q >= 1 else 0.0
    cur_score += 0.6 * sum(1 for k in curiosity_words if k in low)
    cur_score += 0.6 if question_suffix else 0.0
    cur_score -= 0.2 * max(0, n_q - 2)

    # 명료성
    cla_score = 0.0
    if is_very_short: cla_score += 0.3
    elif is_long:     cla_score += 0.6
    else:             cla_score += 1.0
    cla_score += 0.4 if has_sentences else 0.0
    cla_score -= 0.3 * (1 if n_ellipsis >= 1 else 0)
    cla_score -= 0.3 * max(0, n_exc - 1)
    cla_score -= 1.0 * clamp01(all_caps_ratio)

    # 경계선
    bnd_score = 1.5
    bnd_score += 0.2 * sum(1 for k in softeners if k in low)
    bnd_score -= 1.2 * sum(1 for k in boundary_bad if k in low)
    if re.search(r"(지금|바로|당장).*(만나|오|보자)", low): bnd_score -= 1.0
    if re.search(r"(우리집|내방|호텔|모텔)", low):          bnd_score -= 1.0

    # 레드플래그
    red_score = 0.0
    red_score += 1.5 * sum(1 for k in red_flags if k in low)
    red_score += 0.8 * max(0, n_exc - 2)
    red_score += 1.0 * clamp01(all_caps_ratio * 2)

    # 0~10 정규화
    emp = to10(emp_score / 4.0)
    cur = to10(cur_score / 3.0)
    cla = to10(cla_score / 2.2)
    bnd = to10(bnd_score / 2.0)
    red = to10(red_score / 4.0)

    scores = {"공감": emp, "호기심": cur, "명료성": cla, "경계선": bnd, "레드플래그": red}

    # 피드백
    strengths, improvements = [], []
    if emp >= 7: strengths.append("상대의 포인트를 인정·반영하는 표현이 좋아요.")
    if cur >= 7: strengths.append("대화를 확장하는 질문이 자연스럽습니다.")
    if cla >= 7: strengths.append("문장이 간결하고 읽기 쉬워요.")
    if bnd >= 7: strengths.append("경계선을 잘 지키며 예의를 갖췄어요.")

    if emp < 7:
        improvements.append("공감 1구(“말씀 듣고 보니 공감돼요”) 후 관련 질문 1개로 이어보세요.")
    if cur < 7:
        improvements.append("문장 끝에 한 개의 구체 질문만 붙여주세요(‘???’ 남용 금지).")
    if cla < 7:
        if is_long and not has_sentences:
            improvements.append("길다면 문장을 나누고 생략부호/느낌표를 줄여 가독성을 높이세요.")
        elif is_very_short:
            improvements.append("핵심 정보(언제/어디/무엇)를 1–2개만 보강해 주세요.")
        else:
            improvements.append("짧은 문장 1–2개로 정리하고 생략부호/느낌표를 줄여보세요.")
    if bnd < 7:
        improvements.append("연락처/집/즉시 만남 등 사적 요구는 신뢰 형성 후로 미루세요.")
    if red >= 4:
        improvements.append("강한 단어·올캡·느낌표 남용을 피하고 톤을 부드럽게 하세요.")

    candidates = [t for t in tokens if 2 <= len(t) <= 10]
    keyword = candidates[0] if candidates else "이야기"
    tip = f"'{keyword}'를 받아 한 문장으로: 공감 1구 → 관련 질문 1개."
    rewrite_example = f"“{keyword}” 얘기 들으니 공감돼요. 혹시 {keyword}에서 가장 좋았던 점은 뭐였나요?"

    return {
        "scores": scores,
        "feedback": {
            "strengths": strengths[:3],
            "improvements": improvements[:3],
            "tip": tip,
            "rewrite_example": rewrite_example,
        },
        "signals": {
            "question_marks": n_q,
            "exclamations": n_exc,
            "ellipsis": n_ellipsis,
            "all_caps_ratio": round(all_caps_ratio, 2),
            "emoji_like": emoji_like,
            "length_chars": n_chars,
            "length_tokens": n_tokens,
        },
    }

# 가중 총점
def weighted_total(scores: Dict[str, float]) -> float:
    total = 0.0
    for k, w in RUBRIC.items():
        val = scores.get(k, 0)
        if k == "레드플래그":
            val = 10 - val  # 낮을수록 가산
        total += val * w
    return round(total, 2)

# 턴 평가 (LLM / 휴리스틱 / 자동)
def evaluate_turn(user_msg: str, npc_msg: str, difficulty: str) -> Dict[str, Any]:
    use_llm = (EVAL_MODE == "LLM") or (EVAL_MODE == "자동" and OLLAMA_AVAILABLE)

    if use_llm:
        try:
            prompt = build_eval_prompt(user_msg, npc_msg, difficulty)
            resp = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
            )
            data = safe_json_parse(resp["message"]["content"])
            if "scores" in data and all(k in data["scores"] for k in RUBRIC.keys()):
                data["total"] = weighted_total(data["scores"])
                data["__eval_target"] = "나(사용자)"
                data["__evaluated_text"] = user_msg
                data["__engine"] = "LLM"
                return data
        except Exception:
            pass

    data = heuristic_evaluate(user_msg)
    data["total"] = weighted_total(data["scores"])
    data["__eval_target"] = "나(사용자)"
    data["__evaluated_text"] = user_msg
    data["__engine"] = "휴리스틱"
    return data

# 종합 요약
def summarize_overall(score_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not score_list:
        return {"avg_total": 0.0, "strengths": [], "improvements": [], "tip": ""}
    avg_total = round(sum(d["total"] for d in score_list) / len(score_list), 2)
    strengths, improvements = [], []
    for d in score_list:
        strengths += d.get("feedback", {}).get("strengths", [])
        improvements += d.get("feedback", {}).get("improvements", [])
    strengths = list(dict.fromkeys(strengths))[:3]
    improvements = list(dict.fromkeys(improvements))[:3]
    tip = "상대의 마지막 문장에서 키워드 1개를 골라 공감+구체 질문으로 이어가세요."
    return {"avg_total": avg_total, "strengths": strengths, "improvements": improvements, "tip": tip}


# 대화 표시
def render_chat():
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])

render_chat()

# 입력창 (하나만!)
user_input = st.chat_input("메시지를 입력하세요…")


# 전송 처리
def on_user_message(content: str):
    if st.session_state.finished:
        st.info("이 시뮬레이션은 종료되었습니다. 🔄 새 시뮬레이션을 시작해 주세요.")
        return

    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": content})
    with st.chat_message("user"):
        st.markdown(content)

    # NPC 응답 생성
    npc_msg = npc_reply(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": npc_msg})
    with st.chat_message("assistant"):
        st.markdown(npc_msg)

    # 평가 (항상 '나(사용자)'만)
    eval_result = evaluate_turn(content, npc_msg, DIFF)
    st.session_state.scores.append(eval_result)

    # 점수 표시
    with st.expander(f"턴 {st.session_state.turn + 1} 평가 보기", expanded=False):
        st.caption(f"평가 대상: **{eval_result.get('__eval_target', '나(사용자)')}** · 엔진: {eval_result.get('__engine','?')}")
        st.markdown("**평가한 발화(원문):**")
        st.code(eval_result.get("__evaluated_text", ""), language="text")

        cols = st.columns(5)
        for i, k in enumerate(["공감", "호기심", "명료성", "경계선", "레드플래그"]):
            with cols[i]:
                st.metric(k, eval_result["scores"][k])

        st.progress(eval_result["total"] / 10)
        st.caption(f"가중 총점: **{eval_result['total']} / 10**")

        fb = eval_result.get("feedback", {})
        if fb.get("strengths"):
            st.markdown("**👍 강점**")
            for s in fb["strengths"]:
                st.markdown(f"- {s}")
        if fb.get("improvements"):
            st.markdown("**🛠 개선 포인트**")
            for s in fb["improvements"]:
                st.markdown(f"- {s}")
        if fb.get("tip"):
            st.markdown(f"**🎯 다음 턴 팁:** {fb['tip']}")
        if fb.get("rewrite_example"):
            st.markdown("**✍️ 바로 쓸 문장 예시**")
            st.markdown(f"> {fb['rewrite_example']}")

    # 라운드 증가 및 종료 체크
    st.session_state.turn += 1
    if st.session_state.turn >= st.session_state.max_rounds:
        st.session_state.finished = True
        st.session_state.summary = summarize_overall(st.session_state.scores)
        st.success("✅ 시뮬레이션 종료!")
        show_summary()

def show_summary():
    if not st.session_state.summary:
        return
    s = st.session_state.summary
    st.subheader("📊 종합 리포트")
    st.metric("평균 가중 총점", s["avg_total"])
    st.progress(s["avg_total"] / 10)
    if s["strengths"]:
        st.markdown("**👍 전체 대화 강점**")
        for x in s["strengths"]:
            st.markdown(f"- {x}")
    if s["improvements"]:
        st.markdown("**🛠 전체 개선 포인트**")
        for x in s["improvements"]:
            st.markdown(f"- {x}")
    if s["tip"]:
        st.markdown(f"**🎯 최종 코칭:** {s['tip']}")

# 사용자 입력 처리 (빈문자 방지)
if user_input and user_input.strip():
    on_user_message(user_input.strip())

# 종료 상태면 종합 요약 표시
if st.session_state.finished and st.session_state.summary:
    show_summary()

# json 내보내기
def export_json():
    data = {
        "meta": {
            "started_at": st.session_state.started_at,
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": MODEL,
            "temperature": TEMP,
            "rounds": st.session_state.max_rounds,
            "difficulty": DIFF,
            "scenario": SCENARIO,
            "eval_mode": EVAL_MODE,
        },
        "messages": st.session_state.messages,
        "scores": st.session_state.scores,
        "summary": st.session_state.summary or summarize_overall(st.session_state.scores),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)

if export:
    st.download_button(
        label="JSON 다운로드",
        data=export_json().encode("utf-8"),
        file_name=f"date_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )
