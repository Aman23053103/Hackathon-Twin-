# app.py
import streamlit as st
import google.generativeai as genai
import os
import json
from typing import List, Dict, Tuple
from collections import defaultdict

# ---------------------------
# CONFIGURE GEMINI (Flash)
# ---------------------------
# ==== YOUR API KEY ====
API_KEY = "AIzaSyASJrZ6FgGoSf2YBOo-C1xIvhP8qfib74c"
# ======================
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# ---------------------------
# Utilities
# ---------------------------
def call_gemini_with_text(prompt: str, max_output_tokens: int = 512, temperature: float = 0.2) -> Tuple[bool, str]:
    """Call Gemini Flash using the SDK wrapper and return (success, text_or_error)."""
    try:
        resp = model.generate_content(prompt)
        # The SDK typically exposes `text` on the returned object
        text = getattr(resp, "text", None)
        # fallback to resp.candidates handling if SDK returns that structure
        if not text and hasattr(resp, "candidates"):
            cands = getattr(resp, "candidates", [])
            if cands:
                # try navigate nested shape
                try:
                    text = cands[0]["content"]["parts"][0]["text"]
                except Exception:
                    text = str(cands[0])
        if text is None:
            return False, f"Empty response object: {resp}"
        return True, text.strip()
    except Exception as e:
        return False, str(e)

def parse_participants(text: str) -> List[Dict]:
    """
    Accepts lines like:
      Alice: ML, Python, CV
    Returns list of dicts: {name, skills}
    """
    participants = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name, skills = line.split(":", 1)
        elif "-" in line:
            name, skills = line.split("-", 1)
        else:
            name, skills = line, ""
        skills_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
        participants.append({"name": name.strip(), "skills": skills_list})
    return participants

def simple_matchmake(participants: List[Dict], team_size: int = 3) -> List[List[str]]:
    """
    Simple greedy team formation maximizing skill diversity.
    """
    if not participants:
        return []
    pool = sorted(participants, key=lambda p: len(p["skills"]), reverse=True)
    teams = []
    used = set()
    i = 0
    while i < len(pool):
        if pool[i]["name"] in used:
            i += 1
            continue
        team = [pool[i]["name"]]
        used.add(pool[i]["name"])
        needed = team_size - 1
        if needed > 0:
            cur_skills = set(pool[i]["skills"])
            candidates = []
            for j, p in enumerate(pool):
                if p["name"] in used:
                    continue
                overlap = len(cur_skills.intersection(set(p["skills"])))
                candidates.append((overlap, -len(p["skills"]), p))
            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, _, p in candidates[:needed]:
                team.append(p["name"])
                used.add(p["name"])
        teams.append(team)
        i += 1
    leftovers = [p["name"] for p in pool if p["name"] not in used]
    for name in leftovers:
        if not teams:
            teams.append([name])
        else:
            teams[-1].append(name)
    return teams

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Hackathon Twin", layout="wide")
st.title("🚀 Hackathon Twin — Streamlit MVP")

st.sidebar.header("Settings")
st.sidebar.markdown(
    "This prototype uses Gemini Flash (embedded API key). For safety, don't share keys publicly."
)
st.sidebar.markdown("Model: **gemini-1.5-flash**")

# In-memory containers
if "ideas" not in st.session_state:
    st.session_state.ideas = []
if "announcements" not in st.session_state:
    st.session_state.announcements = []
if "teams" not in st.session_state:
    st.session_state.teams = []
if "judging_outputs" not in st.session_state:
    st.session_state.judging_outputs = []

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Idea Generator", "Rubric Builder", "Announcements", "Team Matchmaker", "Judging Assistant"]
)

# ---------------------------
# Tab 1: Idea Generator
# ---------------------------
with tab1:
    st.header("Idea Generator")
    theme = st.text_input("Theme or prompt (e.g., 'AI in agriculture')", value="AI in agriculture")
    n = st.slider("Number of ideas", 1, 10, 5)
    temp = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

    if st.button("Generate Ideas"):
        st.info("Requesting Gemini for ideas...")
        prompt = (
            f"You are an expert hackathon mentor. Suggest {n} concrete, doable and original hackathon project ideas for: {theme}. "
            "For each idea provide: (1) one-line summary, (2) core technical stack, (3) one-sentence success criteria and (4) expected difficulty (low/medium/high). "
            "Return a numbered list."
        )
        ok, out = call_gemini_with_text(prompt)
        if ok:
            st.session_state.ideas = out.splitlines()
            st.success("Ideas generated.")
            st.code(out)
        else:
            st.error(out)

    if st.session_state.ideas:
        st.subheader("Previous output")
        st.write("\n".join(st.session_state.ideas))

# ---------------------------
# Tab 2: Rubric Builder
# ---------------------------
with tab2:
    st.header("Rubric Builder")
    rubric_goal = st.text_input("What should judges evaluate?", value="Assess technical complexity, demo, and impact")
    categories_default = ["Originality", "Technical Complexity", "Impact", "Demo & Presentation"]
    categories = st.text_area("Optional: custom categories (one per line)", value="\n".join(categories_default), height=100)
    scale = st.selectbox("Score scale", ["0-5", "0-10", "0-100"])

    if st.button("Create Rubric"):
        cats = [c.strip() for c in categories.splitlines() if c.strip()]
        prompt = (
            f"As an experienced hackathon judge, produce a clear judging rubric for: {rubric_goal}. "
            f"Provide {len(cats)} categories: {cats}. For each category give a brief description, sub-criteria bullets, and recommended weighting totaling 100% across categories. Use the scoring scale {scale}."
        )
        st.info("Generating rubric...")
        ok, out = call_gemini_with_text(prompt)
        if ok:
            st.code(out)
        else:
            st.error(out)

# ---------------------------
# Tab 3: Announcements
# ---------------------------
with tab3:
    st.header("Announcement & Email Writer")
    event_name = st.text_input("Event / Hackathon name", value="Hack for Good")
    date_info = st.text_input("Dates / schedule", value="Oct 10-12, 2025")
    audience = st.text_input("Audience (participants/judges/mentors)", value="participants")
    tone = st.selectbox("Tone", ["Professional", "Casual", "Excited", "Concise"])

    if st.button("Generate Announcement"):
        prompt = (
            f"Write a short {tone.lower()} announcement for {audience} about the hackathon '{event_name}' happening {date_info}. "
            "Include 1) one-line headline, 2) 3 bullet points (what, why, how to join), 3) a short closing call-to-action. Keep it under 200 words."
        )
        st.info("Generating announcement...")
        ok, out = call_gemini_with_text(prompt)
        if ok:
            st.session_state.announcements.append(out)
            st.success("Announcement created.")
            st.markdown(out)
        else:
            st.error(out)

    if st.session_state.announcements:
        st.subheader("Saved Announcements (latest first)")
        for i, ann in enumerate(reversed(st.session_state.announcements[-5:]), 1):
            st.markdown(f"**Announcement #{len(st.session_state.announcements) - i + 1}:**")
            st.write(ann)
            st.markdown("---")

# ---------------------------
# Tab 4: Team Matchmaker
# ---------------------------
with tab4:
    st.header("Team Matchmaker (simple)")
    st.markdown("Paste participant lines like: `Alice: ML, Python, CV` (one per line).")
    participants_text = st.text_area("Participants list", height=200, key="participants_text")
    team_size = st.slider("Team size", 2, 6, 3)

    if st.button("Suggest Teams"):
        participants = parse_participants(participants_text)
        if not participants:
            st.error("No participants parsed. Use the format `Name: skill1, skill2` on multiple lines.")
        else:
            teams = simple_matchmake(participants, team_size=team_size)
            st.session_state.teams = teams
            st.success(f"Suggested {len(teams)} teams.")

    if st.session_state.teams:
        st.subheader("Teams")
        for idx, t in enumerate(st.session_state.teams, 1):
            st.write(f"Team {idx}: {', '.join(t)}")

# ---------------------------
# Tab 5: Judging Assistant
# ---------------------------
with tab5:
    st.header("Judging Assistant")
    st.markdown("Paste project description / README / submission text and get a summary + suggested rubric scores.")
    submission_text = st.text_area("Submission text", height=250)
    rubric_input = st.text_area("Rubric categories (one per line)", value="Originality\nTechnical Complexity\nImpact\nDemo")
    scale_choice = st.selectbox("Which numeric scale to suggest scores on?", ["0-5", "0-10", "0-100"])

    if st.button("Summarize & Score"):
        if not submission_text.strip():
            st.error("Paste submission content first.")
        else:
            cats = [c.strip() for c in rubric_input.splitlines() if c.strip()]
            prompt = (
                "You are a pragmatic hackathon judge assistant. Given the following submission content delimited by triple backticks, "
                f"1) produce a concise (<= 150 words) summary, 2) for each rubric category {cats} provide a short justification and a suggested numeric score on scale {scale_choice}. "
                "Return STRICT JSON with keys: summary (string), scores (object) where scores maps category -> {score:number, justification:string}. "
                "Do not include extra commentary or explanation outside the JSON.\n\n"
                "Submission:\n```" + submission_text + "```"
            )
            st.info("Requesting Gemini for summary & scores...")
            ok, out = call_gemini_with_text(prompt)
            if not ok:
                st.error(out)
            else:
                # try to parse JSON from the response
                parsed = None
                try:
                    # sometimes the model returns code blocks or text around JSON; extract first '{'...' }' pair
                    start = out.find("{")
                    if start != -1:
                        json_text = out[start:]
                        parsed = json.loads(json_text)
                    else:
                        parsed = json.loads(out)
                except Exception:
                    parsed = None

                if parsed:
                    st.subheader("Summary")
                    st.write(parsed.get("summary", "No summary found."))
                    st.subheader("Suggested Scores")
                    scores = parsed.get("scores", {})
                    for cat, detail in scores.items():
                        sc = detail.get("score", "N/A")
                        just = detail.get("justification", "")
                        st.markdown(f"**{cat}** — **{sc}**")
                        st.write(just)
                    st.session_state.judging_outputs.append(parsed)
                else:
                    st.warning("Could not parse strict JSON from model output — showing raw output below.")
                    st.code(out)

st.markdown("---")
st.caption("Hackathon Twin — Streamlit MVP using Gemini Flash. This is a prototype: consider adding storage, auth, rate-limiting, and removing the key from source code for production.")
