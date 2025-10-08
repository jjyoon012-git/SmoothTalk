# 1.py
# 실행: streamlit run 1.py
# 필요: pip install streamlit, 그리고 로컬에서 `ollama pull gemma3:4b` (또는 원하는 모델)

import json
import re
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st

# ================== 상황별 아이콘/타이틀(초기 설정) ==================
SCENARIO_META = {
    "소개팅":                           {"icon": "💘", "title": "Smooth Talk: 소개팅 시뮬레이터"},
    "첫 만남(직장)":                    {"icon": "💼", "title": "Smooth Talk: 첫 만남(직장) 시뮬레이터"},
    "동아리/동호회":                    {"icon": "🎯", "title": "Smooth Talk: 동아리/동호회 시뮬레이터"},
    "친구의 친구 모임":                 {"icon": "👥", "title": "Smooth Talk: 친구의 친구 모임 시뮬레이터"},
    "면접(캐주얼)":                     {"icon": "🧑‍💻", "title": "Smooth Talk: 면접(캐주얼) 시뮬레이터"},
    "첫 만남 (카페)":                   {"icon": "☕", "title": "Smooth Talk: 카페 첫 만남"},
    "첫 만남 (레스토랑/저녁 식사)":     {"icon": "🍽️", "title": "Smooth Talk: 저녁 식사 첫 만남"},
    "첫 만남 (영화관/데이트 코스)":     {"icon": "🎬", "title": "Smooth Talk: 영화관 데이트"},
    "첫 만남 (공원/야외 산책)":          {"icon": "🌿", "title": "Smooth Talk: 야외 산책 첫 만남"},
}
_DEFAULT_ICON = "💬"
_DEFAULT_TITLE = "Smooth Talk: 시뮬레이터"

# set_page_config는 최상단 1회만!
st.set_page_config(page_title=_DEFAULT_TITLE, page_icon=_DEFAULT_ICON)

# ================== Ollama 체크 ==================
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    print("올라마없")
    OLLAMA_AVAILABLE = False

# ================== 사이드바 설정 ==================
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    MODEL = st.text_input("모델명 (Ollama)", value="gemma3:4b", help="예: gemma3:4b, llama3.1:8b, qwen2.5:7b 등")
    TEMP = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    ROUNDS = st.slider("라운드 수", 3, 12, 6, 1)
    DIFF = st.selectbox("난이도", ["쉬움", "보통", "어려움"], index=1)

    SCENARIO = st.selectbox(
        "상황",
        [
            "소개팅",
            "첫 만남(직장)",
            "동아리/동호회",
            "친구의 친구 모임",
            "면접(캐주얼)",
            "첫 만남 (카페)",
            "첫 만남 (레스토랑/저녁 식사)",
            "첫 만남 (영화관/데이트 코스)",
            "첫 만남 (공원/야외 산책)",
        ],
        index=0,
    )

    EVAL_MODE = st.radio(
        "평가 엔진",
        ["자동", "LLM", "휴리스틱"],
        index=0,
        help="자동: LLM 사용, 실패 시 휴리스틱"
    )
    st.caption("※ Ollama가 실행 중이어야 모델 대화/평가가 동작합니다.")

# 상황별 타이틀/아이콘 적용
_meta = SCENARIO_META.get(SCENARIO, {"icon": _DEFAULT_ICON, "title": _DEFAULT_TITLE})
st.title(f"{_meta['icon']} {_meta['title']}")
st.caption("상황을 선택하고 프로필을 입력하면, 해당 상황에 맞는 톤과 맥락으로 시뮬레이션이 진행됩니다.")

# ================== 세션 상태 ==================
def init_state():
    st.session_state.messages: List[Dict[str, str]] = []
    st.session_state.turn = 0
    st.session_state.max_rounds = ROUNDS
    st.session_state.scores: List[Dict[str, Any]] = []
    st.session_state.summary = None
    st.session_state.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.finished = False
    if "profile" not in st.session_state:
        st.session_state.profile = {}

if ("messages" not in st.session_state) or st.sidebar.button("🔄 새 시뮬레이션 시작", use_container_width=True):
    init_state()

export = st.sidebar.button("💾 기록 내보내기 (JSON)", use_container_width=True)

# ================== 시나리오 힌트 ==================
def scenario_hint(s: str) -> str:
    if s == "소개팅": return "가볍게 서로를 탐색, 예의 바르고 부담 적게."
    if s == "첫 만남(직장)": return "정중·차분, 업무/업무외 밸런스. 사생활 과도 침투 금지."
    if s == "동아리/동호회": return "공통 취미 중심으로 라포 형성, 활동 경험 공유."
    if s == "친구의 친구 모임": return "공통분모(공통친구)로 안전한 주제 확장."
    if s == "면접(캐주얼)": return "라이트톤 + 전문성, 구체 사례 위주. 공격적 질문 지양."
    if "카페" in s: return "첫 만남 특유의 가벼운 탐색, 취향·일상 질문 위주."
    if "레스토랑" in s: return "정중하고 차분한 톤, 감정·가치관 질문 1개 포함."
    if "영화관" in s: return "경험 공유와 감상 질문 1개 포함."
    if "산책" in s: return "자연스러운 관찰 코멘트 + 가벼운 질문."
    return ""

# ================== 상황별 프로필 입력(심플) ==================
with st.expander("🧾 상황별 프로필", expanded=False):
    prof = st.session_state.get("profile", {})
    if SCENARIO == "소개팅":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["상대 이름"] = st.text_input("상대 이름", prof.get("상대 이름", ""))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "영화, 카페"))
    elif SCENARIO == "첫 만남(직장)":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["팀"] = st.text_input("팀/파트", prof.get("팀", ""))
        prof["역할"] = st.text_input("역할", prof.get("역할", ""))
    elif SCENARIO == "동아리/동호회":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["동아리"] = st.text_input("동아리명", prof.get("동아리", ""))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "등산, 보드게임"))
    elif SCENARIO == "친구의 친구 모임":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["공통 친구"] = st.text_input("공통 친구", prof.get("공통 친구", ""))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "여행, 음악"))
    elif SCENARIO == "면접(캐주얼)":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["기술스택"] = st.text_input("기술스택", prof.get("기술스택", "Python, React"))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "스타트업, 오픈소스"))
    elif SCENARIO == "첫 만남 (카페)":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["상대 이름"] = st.text_input("상대 이름", prof.get("상대 이름", ""))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "영화, 카페"))
    elif SCENARIO == "첫 만남 (레스토랑/저녁 식사)":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["상대 이름"] = st.text_input("상대 이름", prof.get("상대 이름", ""))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "맛집, 여행"))
    elif SCENARIO == "첫 만남 (영화관/데이트 코스)":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["상대 이름"] = st.text_input("상대 이름", prof.get("상대 이름", ""))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "영화, 드라마"))
    elif SCENARIO == "첫 만남 (공원/야외 산책)":
        prof["내 이름"] = st.text_input("내 이름", prof.get("내 이름", ""))
        prof["상대 이름"] = st.text_input("상대 이름", prof.get("상대 이름", ""))
        prof["관심사"] = st.text_input("관심사(쉼표)", prof.get("관심사", "산책, 반려동물"))

    # 관심사 쉼표 분해(있으면)
    if "관심사" in prof:
        prof["_관심사_list"] = [s.strip() for s in (prof["관심사"] or "").split(",") if s.strip()]

    st.session_state.profile = prof
    if scenario_hint(SCENARIO):
        st.caption(f"시나리오 힌트: {scenario_hint(SCENARIO)}")

# ================== 시스템 프롬프트 ==================
def profile_summary_kr(profile: Dict[str, Any]) -> str:
    parts = []
    if profile.get("내 이름"): parts.append(f"내 이름: {profile['내 이름']}")
    if profile.get("상대 이름"): parts.append(f"상대 이름: {profile['상대 이름']}")
    if profile.get("_관심사_list"): parts.append(f"관심사: {', '.join(profile['_관심사_list'])}")
    if profile.get("팀"): parts.append(f"팀/파트: {profile['팀']}")
    if profile.get("역할"): parts.append(f"역할: {profile['역할']}")
    if profile.get("동아리"): parts.append(f"동아리: {profile['동아리']}")
    if profile.get("공통 친구"): parts.append(f"공통 친구: {profile['공통 친구']}")
    if profile.get("기술스택"): parts.append(f"기술스택: {profile['기술스택']}")
    return " · ".join(parts) if parts else "추가 프로필 없음"

def build_system_prompt():
    prof = st.session_state.get("profile", {})
    ctx = profile_summary_kr(prof)
    return f"""You are a Korean-speaking NPC for a conversation simulator.

- Role: 상대역(NPC)로 자연스럽게 대화한다. 사용자는 '{prof.get('내 이름') or '사용자'}'.
- Style: 공손하고 따뜻하며, 솔직하고 유머는 가볍게.
- Length: 2~4문장 위주로 답하고, 마지막에 짧은 질문 1개를 덧붙인다.
- Avoid: 장문 독백, 과도한 칭찬/사담, 모호한 답변, 신상침해성 요구.
- Boundaries: 과한 신체/사생활 침해성 질문은 부드럽게 선 긋기.
- Scenario: {SCENARIO} — Hint: {scenario_hint(SCENARIO)}
- Difficulty: {DIFF} (난이도가 높을수록 말을 아끼고, 되묻거나 작은 테스트 질문을 섞는다)
- Politeness: 존댓말 사용을 권장한다. 예의 없는 반말/명령형/호칭 무시는 사용자 평가의 '정중함'에서 감점 요인이다.
- Context Summary: {ctx}
- Language: 한국어만 사용한다.
"""

def ensure_system_message():
    sys_prompt = build_system_prompt()
    if not any(m.get("role") == "system" for m in st.session_state.messages):
        st.session_state.messages.insert(0, {"role": "system", "content": sys_prompt})
    else:
        if st.session_state.messages[0]["content"] != sys_prompt:
            st.session_state.messages[0]["content"] = sys_prompt

ensure_system_message()

# ================== NPC 응답 생성 ==================
def npc_reply(messages: List[Dict[str, str]]) -> str:
    if not OLLAMA_AVAILABLE:
        return "저는 시뮬레이터 NPC예요. (Ollama 미동작) — 요즘 어떤 취미 즐기세요?"

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": float(TEMP)},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        hint = f"모델이 로컬에 없을 수 있어요. 터미널에서 `ollama pull {MODEL}` 후 다시 시도해보세요."
        return f"(모델 오류) 간단히 이어갈게요. 요즘 어떻게 지내셨어요?\n\n— 에러: {e}\n— 힌트: {hint}"

# ================== 평가 로직(반말/무례 톤 페널티 강화) ==================
RUBRIC = {
    "공감": 0.25,
    "호기심": 0.20,
    "명료성": 0.20,
    "정중함": 0.20,
    "레드플래그": 0.15
}

def build_eval_prompt(user_msg: str, npc_msg: str, difficulty: str) -> str:
    return f"""너는 소개팅/사회적 대화 코치야. 아래 '평가대상' 발화만 평가해.
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
- 정중함(0~10): 예의를 갖추었는가? (반말/명령형/호칭 무시/무례어/비격식 강한 슬랭 사용 시 크게 감점)
- 레드플래그(0~10): 무례/과몰입/사생활침해/거짓말/셀프디스 등(높을수록 나쁨)

[난이도]
{difficulty}

[출력 형식(JSON strict)]
{{
  "scores": {{
    "공감": 0-10,
    "호기심": 0-10,
    "명료성": 0-10,
    "정중함": 0-10,
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
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        cleaned = re.sub(r"(\d+)\s*-\s*10", "10", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

def heuristic_evaluate(user_msg: str) -> Dict[str, Any]:
    txt = (user_msg or "").strip()
    low = txt.lower()

    tokens = re.findall(r"[가-힣A-Za-z0-9]+", low)
    n_tokens = len(tokens)
    n_chars = len(txt)

    n_q = txt.count("?")
    n_exc = txt.count("!")
    n_ellipsis = txt.count("…") + txt.count("...")

    cap_tokens = [t for t in re.findall(r"[A-Za-z]+", txt) if len(t) >= 3]
    all_caps_ratio = (len([t for t in cap_tokens if t.isupper()]) / len(cap_tokens)) if cap_tokens else 0.0

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
    red_flags = ["싫", "꺼져", "닥쳐", "미친", "뭐래", "멍청", "병신", "야 ", "돈자랑"]

    # ---------- 반말/무례 톤 감지(정중함 감점용) ----------
    honorific_markers = ["요", "입니다", "합니다", "하세요", "십시오", "합니까", "해요", "드립니다"]
    honorific_hits = sum(1 for m in honorific_markers if m in txt)

    banmal_patterns = [
        r"[가-힣A-Za-z0-9]+다$", r"[가-힣]+해$", r"[가-힣]+해\?", r"[가-힣]+해라$", r"[가-힣]+해봐$",
        r"[가-힣]+해줘$", r"[가-힣]+해줄래", r"[가-힣]+냐\?$", r"[가-힣]+니\?$", r"[가-힣]+해라\?$",
        r".*빨리.*", r".*지금.*해$", r".*와라$", r".*보자$"
    ]
    banmal_hit = any(re.search(p, txt) for p in banmal_patterns)

    # ✅ '야' 단독/변형(야?, 야!! 등) 강력 감지 + 문장 내 포함 케이스
    rude_vocative_regex = r"^\s*야+[!?.]?\s*$"
    rude_vocatives_inline = ["야 ", "야,", "야?", "야!"]
    rude_vocative_hit = bool(re.search(rude_vocative_regex, txt)) or any(rv in txt for rv in rude_vocatives_inline)

    # ---------- 점수 산정 ----------
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

    # 정중함 (기본치)
    polite_score = 1.5
    polite_score += 0.2 * sum(1 for k in softeners if k in low)
    polite_score -= 1.2 * sum(1 for k in boundary_bad if k in low)
    if re.search(r"(지금|바로|당장).*(만나|오|보자)", low): polite_score -= 1.0
    if re.search(r"(우리집|내방|호텔|모텔)", low):          polite_score -= 1.0

    # ✅ 정중함 페널티 강화
    if rude_vocative_hit:
        polite_score -= 1.5            # '야' 단독 등 강한 감점
    if banmal_hit:
        polite_score -= 1.2
    if honorific_hits == 0 and n_chars <= 3:
        polite_score -= 0.9            # 아주 짧은 반말/명령형
    if honorific_hits == 0 and n_chars >= 8:
        polite_score -= 0.6            # 평문인데 존댓말 흔적 전무

    # 레드플래그
    red_score = 0.0
    red_score += 1.5 * sum(1 for k in red_flags if k in low)
    red_score += 0.8 * max(0, n_exc - 2)
    red_score += 1.0 * clamp01(all_caps_ratio * 2)

    emp = to10(emp_score / 4.0)
    cur = to10(cur_score / 3.0)
    cla = to10(cla_score / 2.2)
    polite = to10(polite_score / 2.0)
    red = to10(red_score / 4.0)

    scores = {"공감": emp, "호기심": cur, "명료성": cla, "정중함": polite, "레드플래그": red}

    strengths, improvements = [], []
    if emp >= 7: strengths.append("상대의 포인트를 인정·반영하는 표현이 좋아요.")
    if cur >= 7: strengths.append("대화를 확장하는 질문이 자연스럽습니다.")
    if cla >= 7: strengths.append("문장이 간결하고 읽기 쉬워요.")
    if polite >= 7: strengths.append("존중감 있는 어투로 예의를 잘 지켰어요.")

    if emp < 7:
        improvements.append("공감 1구(“말씀 듣고 보니 공감돼요”) 후 관련 질문 1개로 이어보세요.")
    if cur < 7:
        improvements.append("문장 끝에 구체 질문 1개만 덧붙여 대화를 확장해 보세요.")
    if cla < 7:
        if is_long and not has_sentences:
            improvements.append("길다면 문장을 나누고 생략부호/느낌표를 줄여 가독성을 높이세요.")
        elif is_very_short:
            improvements.append("핵심 정보(언제/어디/무엇)를 1–2개만 보강해 주세요.")
        else:
            improvements.append("짧은 문장 1–2개로 정리하고 생략부호/느낌표를 줄여보세요.")
    if polite < 7:
        improvements.append("존댓말(요/습니다)과 완곡한 표현을 사용해 톤을 부드럽게 해보세요.")
        if rude_vocative_hit or banmal_hit or honorific_hits == 0:
            improvements.append("반말/명령형을 피하고 “혹시…”, “괜찮으시면…” 같은 완곡어를 활용하세요.")
    if red >= 4:
        improvements.append("강한 단어·올캡·느낌표 남용을 피하고 톤을 부드럽게 하세요.")

    candidates = [t for t in tokens if 2 <= len(t) <= 10]
    keyword = candidates[0] if candidates else "이야기"
    tip = f"'{keyword}'를 받아 한 문장으로: 공감 1구 → 존댓말 질문 1개."
    rewrite_example = f"“{keyword}” 말씀 공감돼요. 혹시 {keyword}에서 가장 좋았던 점은 무엇이었나요?"

    return {
        "scores": scores,
        "feedback": {
            "strengths": strengths[:3],
            "improvements": improvements[:3],
            "tip": tip,
            "rewrite_example": rewrite_example,
        },
        "signals": {},
    }

def weighted_total(scores: Dict[str, float]) -> float:
    total = 0.0
    for k, w in RUBRIC.items():
        val = scores.get(k, 0)
        if k == "레드플래그":
            val = 10 - val
        total += val * w
    return round(total, 2)

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
    tip = "상대의 마지막 문장에서 키워드 1개를 골라 공감 + 존댓말 질문으로 이어가세요."
    return {"avg_total": avg_total, "strengths": strengths, "improvements": improvements, "tip": tip}

# ================== 채팅 표시 ==================
def render_chat():
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])

render_chat()
user_input = st.chat_input("메시지를 입력하세요…")

# ================== 전송 처리 ==================
def on_user_message(content: str):
    if st.session_state.finished:
        st.info("이 시뮬레이션은 종료되었습니다. 🔄 새 시뮬레이션을 시작해 주세요.")
        return

    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": content})
    with st.chat_message("user"):
        st.markdown(content)

    # NPC 응답
    npc_msg = npc_reply(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": npc_msg})
    with st.chat_message("assistant"):
        st.markdown(npc_msg)

    # 평가 (항상 사용자 발화만)
    eval_result = evaluate_turn(content, npc_msg, DIFF)
    st.session_state.scores.append(eval_result)

    # 점수 표시
    with st.expander(f"턴 {st.session_state.turn + 1} 평가 보기", expanded=False):
        st.caption(f"평가 대상: **{eval_result.get('__eval_target', '나(사용자)')}** · 엔진: {eval_result.get('__engine','?')}")
        st.markdown("**평가한 발화(원문):**")
        st.code(eval_result.get("__evaluated_text", ""), language="text")

        cols = st.columns(5)
        for i, k in enumerate(["공감", "호기심", "명료성", "정중함", "레드플래그"]):
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

    # 라운드/종료
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

# ================== 입력 처리 ==================
if user_input and user_input.strip():
    ensure_system_message()
    on_user_message(user_input.strip())

# 종료 상태면 종합 요약 표시
if st.session_state.finished and st.session_state.summary:
    show_summary()

# ================== JSON 내보내기 ==================
def export_json():
    data = {
        "meta": {
            "started_at": st.session_state.started_at,
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": MODEL,
            "temperature": float(TEMP),
            "rounds": st.session_state.max_rounds,
            "difficulty": DIFF,
            "scenario": SCENARIO,
            "eval_mode": EVAL_MODE,
        },
        "profile": st.session_state.get("profile", {}),
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
