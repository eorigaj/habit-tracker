# app.py
import os
import re
import json
from datetime import datetime, timedelta

import requests
import pandas as pd
import streamlit as st

# OpenAI (official SDK)
# pip install openai
from openai import OpenAI


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Sidebar: API Keys
# -----------------------------
st.sidebar.header("🔑 API 설정")

openai_key = st.sidebar.text_input(
    "OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password"
)
owm_key = st.sidebar.text_input(
    "OpenWeatherMap API Key", value=os.getenv("OPENWEATHERMAP_API_KEY", ""), type="password"
)

st.sidebar.caption("키는 브라우저 세션(session_state)에서만 사용돼요.")


# -----------------------------
# Utilities / API functions
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap 현재 날씨 (한국어, 섭씨)
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        weather_desc = (data.get("weather") or [{}])[0].get("description")
        main = data.get("main") or {}
        temp = main.get("temp")
        feels_like = main.get("feels_like")
        humidity = main.get("humidity")
        return {
            "city": city,
            "description": weather_desc,
            "temp": temp,
            "feels_like": feels_like,
            "humidity": humidity,
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_dog_image():
    """
    Dog CEO 랜덤 강아지 사진 URL + 품종 추출
    실패 시 None 반환, timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        img_url = data.get("message")
        if not img_url:
            return None

        # 품종 추출: .../breeds/<breed>/xxx.jpg
        breed = None
        m = re.search(r"/breeds/([^/]+)/", img_url)
        if m:
            breed = m.group(1).replace("-", " ").strip()

        return {"url": img_url, "breed": breed or "unknown"}
    except Exception:
        return None


def _coach_system_prompt(style: str) -> str:
    if style == "스파르타 코치":
        return (
            "당신은 엄격한 스파르타 코치다. 변명은 차단하고, 핵심만 찌르며, "
            "실행 가능한 액션을 강하게 지시한다. 다만 인신공격은 금지."
        )
    if style == "따뜻한 멘토":
        return (
            "당신은 따뜻한 멘토다. 공감과 격려를 기반으로, 작은 성취를 강화하고 "
            "현실적인 다음 कदम을 제시한다."
        )
    # 게임 마스터
    return (
        "당신은 RPG 게임 마스터다. 사용자를 플레이어로 보고, 오늘의 상태를 버프/디버프로 묘사하며 "
        "퀘스트 형태로 내일 미션을 제시한다. 유쾌하고 몰입감 있게."
    )


def generate_report(
    openai_api_key: str,
    coach_style: str,
    date_str: str,
    city: str,
    mood: int,
    habits_checked: list,
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 묶어 OpenAI에 전달
    모델: gpt-5-mini
    출력 형식:
      - 컨디션 등급(S~D)
      - 습관 분석
      - 날씨 코멘트
      - 내일 미션
      - 오늘의 한마디
    """
    if not openai_api_key:
        return None

    system = _coach_system_prompt(coach_style)

    weather_line = "날씨 정보 없음"
    if weather:
        weather_line = (
            f"{weather.get('city')} 현재 날씨: {weather.get('description')}, "
            f"{weather.get('temp')}°C (체감 {weather.get('feels_like')}°C), 습도 {weather.get('humidity')}%"
        )

    dog_line = "강아지 정보 없음"
    if dog:
        dog_line = f"오늘의 강아지 품종: {dog.get('breed')}"

    user_payload = {
        "date": date_str,
        "city": city,
        "mood_1_to_10": mood,
        "completed_habits": habits_checked,
        "weather": weather or None,
        "dog": dog or None,
    }

    user_msg = f"""
아래 사용자 데이터를 기반으로 'AI 습관 트래커' 컨디션 리포트를 작성해줘.

[요구 출력 형식]
1) 컨디션 등급: S/A/B/C/D 중 하나 (한 줄)
2) 습관 분석: 잘한 점 2가지 + 개선 1가지 (불릿)
3) 날씨 코멘트: 오늘 날씨에 맞춘 조언 1~2문장
4) 내일 미션: 3개의 구체적인 미션 (체크리스트 형태)
5) 오늘의 한마디: 1문장 (스타일에 맞게)

[참고]
- 달성률이 낮으면 원인 가설 + 최소 미션 전략을 제시해.
- 기분 점수(1~10)를 중요 신호로 활용해.
- 날씨/강아지 품종도 자연스럽게 한 번은 언급해.

[요약 텍스트]
- {weather_line}
- {dog_line}

[원본 데이터(JSON)]
{json.dumps(user_payload, ensure_ascii=False, indent=2)}
""".strip()

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.responses.create(
            model="gpt-5-mini",
            instructions=system,
            input=user_msg,
        )
        return (resp.output_text or "").strip() or None
    except Exception:
        return None


# -----------------------------
# Session state: history
# -----------------------------
def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _init_history_if_needed():
    if "history" in st.session_state:
        return

    # 데모용 6일 샘플 + 오늘(빈 값)
    today = datetime.now().date()
    sample = []
    # 최근 6일(오늘 제외)
    preset_rates = [40, 60, 80, 20, 100, 60]
    preset_moods = [5, 6, 7, 4, 8, 6]
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        sample.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "rate": preset_rates[6 - i],
                "completed": int(round(preset_rates[6 - i] / 100 * 5)),
                "mood": preset_moods[6 - i],
            }
        )

    # 오늘 엔트리(초기값)
    sample.append(
        {
            "date": today.strftime("%Y-%m-%d"),
            "rate": 0,
            "completed": 0,
            "mood": 5,
        }
    )

    st.session_state.history = sample


def _upsert_today(rate: int, completed: int, mood: int):
    today_str = datetime.now().date().strftime("%Y-%m-%d")
    hist = st.session_state.history
    idx = next((i for i, r in enumerate(hist) if r["date"] == today_str), None)
    row = {"date": today_str, "rate": int(rate), "completed": int(completed), "mood": int(mood)}
    if idx is None:
        hist.append(row)
    else:
        hist[idx] = row
    # 최근 7일 유지
    hist_sorted = sorted(hist, key=lambda x: x["date"])
    st.session_state.history = hist_sorted[-7:]


_init_history_if_needed()


# -----------------------------
# Main UI
# -----------------------------
st.title("📊 AI 습관 트래커")
st.caption("체크인 → 달성률 확인 → 날씨/강아지/AI 코칭 리포트까지 한 번에!")

st.subheader("✅ 오늘의 습관 체크인")

habits = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

colA, colB = st.columns(2)
checked = []

with colA:
    c1 = st.checkbox(f"{habits[0][0]} {habits[0][1]}", value=False)
    c2 = st.checkbox(f"{habits[1][0]} {habits[1][1]}", value=False)
    c3 = st.checkbox(f"{habits[2][0]} {habits[2][1]}", value=False)

with colB:
    c4 = st.checkbox(f"{habits[3][0]} {habits[3][1]}", value=False)
    c5 = st.checkbox(f"{habits[4][0]} {habits[4][1]}", value=False)

flags = [c1, c2, c3, c4, c5]
for (emoji, name), is_on in zip(habits, flags):
    if is_on:
        checked.append(name)

mood = st.slider("🙂 오늘 기분 점수", min_value=1, max_value=10, value=7)

cities = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Ulsan", "Suwon", "Seongnam", "Jeju",
]
city = st.selectbox("📍 도시 선택", cities, index=0)

coach_style = st.radio(
    "🧠 코치 스타일",
    ["스파르타 코치", "따뜻한 멘토", "게임 마스터"],
    horizontal=True,
)

completed_count = sum(flags)
rate = int(round(completed_count / 5 * 100))


# -----------------------------
# Metrics + chart
# -----------------------------
st.subheader("📈 오늘의 지표")

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{rate}%")
m2.metric("달성 습관", f"{completed_count}/5")
m3.metric("기분", f"{mood}/10")

# 오늘 기록을 history에 반영(세션 유지)
_upsert_today(rate=rate, completed=completed_count, mood=mood)

st.subheader("🗓️ 최근 7일 달성률")
df = pd.DataFrame(st.session_state.history)
# 보기 좋은 순서
df = df.sort_values("date")
chart_df = df.set_index("date")[["rate"]]
st.bar_chart(chart_df)


# -----------------------------
# Generate report
# -----------------------------
st.divider()
st.subheader("🧾 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", type="primary")

weather = None
dog = None
report = None

if btn:
    with st.spinner("날씨/강아지/리포트를 준비 중..."):
        weather = get_weather(city, owm_key) if owm_key else None
        dog = get_dog_image()

        today_str = datetime.now().strftime("%Y-%m-%d")
        report = generate_report(
            openai_api_key=openai_key,
            coach_style=coach_style,
            date_str=today_str,
            city=city,
            mood=mood,
            habits_checked=checked,
            weather=weather,
            dog=dog,
        )

    # 결과 표시
    left, right = st.columns(2)

    with left:
        st.markdown("### 🌦️ 오늘의 날씨")
        if weather:
            st.info(
                f"**{weather.get('city')}**\n\n"
                f"- 상태: {weather.get('description')}\n"
                f"- 기온: {weather.get('temp')}°C (체감 {weather.get('feels_like')}°C)\n"
                f"- 습도: {weather.get('humidity')}%"
            )
        else:
            st.warning("날씨 정보를 가져오지 못했어요. (API Key/네트워크를 확인해 주세요)")

    with right:
        st.markdown("### 🐶 오늘의 강아지")
        if dog and dog.get("url"):
            st.image(dog["url"], use_container_width=True)
            st.caption(f"품종: {dog.get('breed', 'unknown')}")
        else:
            st.warning("강아지 이미지를 가져오지 못했어요.")

    st.markdown("### 🧠 AI 코치 리포트")
    if report:
        st.write(report)
    else:
        if not openai_key:
            st.error("OpenAI API Key가 필요해요. 사이드바에서 입력해 주세요.")
        else:
            st.error("리포트 생성에 실패했어요. 잠시 후 다시 시도해 주세요.")

    # 공유용 텍스트
    habit_line = ", ".join(checked) if checked else "없음"
    weather_short = (
        f"{weather.get('description')} / {weather.get('temp')}°C" if weather else "날씨 없음"
    )
    dog_short = dog.get("breed") if dog else "강아지 없음"

    share_text = f"""[AI 습관 트래커 공유]
- 날짜: {datetime.now().strftime("%Y-%m-%d")}
- 도시: {city}
- 달성률: {rate}% ({completed_count}/5)
- 완료 습관: {habit_line}
- 기분: {mood}/10
- 날씨: {weather_short}
- 오늘의 강아지: {dog_short}

[AI 코치 리포트]
{report or "(리포트 없음)"}
"""
    st.markdown("### 📣 공유용 텍스트")
    st.code(share_text, language="text")


# -----------------------------
# API 안내 (Expander)
# -----------------------------
with st.expander("ℹ️ API 안내 / 트러블슈팅"):
    st.markdown(
        """
- **OpenAI API Key**: OpenAI 플랫폼에서 발급한 키가 필요합니다.  
  - 이 앱은 **OpenAI Python SDK**의 **Responses API**로 `gpt-5-mini` 모델을 호출합니다. :contentReference[oaicite:0]{index=0}
- **OpenWeatherMap API Key**: OpenWeatherMap에서 발급한 키가 필요합니다.  
  - 현재 날씨 API를 `units=metric(섭씨)`, `lang=kr(한국어)`로 호출합니다. :contentReference[oaicite:1]{index=1}
- **Dog CEO API**: 키 없이 사용 가능합니다. 랜덤 강아지 이미지를 가져옵니다. :contentReference[oaicite:2]{index=2}

**자주 발생하는 문제**
- 날씨가 `None`: OpenWeatherMap 키가 없거나, 호출 제한/도시명 오타/네트워크 문제일 수 있어요.
- 리포트 실패: OpenAI 키가 없거나, 네트워크/권한 문제일 수 있어요.
        """.strip()
    )
