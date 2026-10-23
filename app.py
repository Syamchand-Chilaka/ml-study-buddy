"""ML Study Buddy — Streamlit chat UI and RAG step 5 (GENERATE).

Flow per message: retrieve relevant study-guide chunks (rag.py), build the
system prompt around them (prompts.py), ask the LLM, show answer + sources.
"""

import os

import openai
import streamlit as st

import prompts
import rag

# Generation runs on Fireworks AI's OpenAI-compatible inference API.
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/glm-5p3-flash"  # small, fast, cheap

# The key is never hardcoded. Lookup order: environment variable (a local
# .env file is read as a convenience), then Streamlit Cloud's secrets store.
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key, value)
API_KEY = os.environ.get("LLM_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["LLM_API_KEY"]
    except Exception:
        API_KEY = None


# Build the index once per server (RAG steps 1-3: ingest, chunk, embed).
@st.cache_resource(show_spinner="Reading the study guides...")
def load_index():
    return rag.build_index()


CHUNKS, VECTORS = load_index()
client = (openai.OpenAI(api_key=API_KEY, base_url=FIREWORKS_URL)
          if API_KEY else None)


def format_context(hits):
    """Turn retrieved chunks into the CONTEXT block of the system prompt."""
    return "\n\n---\n\n".join(
        f"[source: {chunk['source']}]\n{chunk['text']}" for chunk, _ in hits)


# STEP 5 — GENERATE: ask the LLM to answer from the retrieved context only.
def generate(system, messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=1024,
    )
    return response.choices[0].message.content


GREETINGS = ("hi", "hello", "hey", "hi there", "hello there", "hiya",
             "good morning", "good afternoon", "good evening")


def excerpt_answer(hits, sources):
    """Backup answer when the LLM is unavailable: quote the best excerpt."""
    return ("Here's what my study guide says about that:\n\n"
            f"{hits[0][0]['text']}\n\n*Sources: {sources}*")


def chat(message, history):
    """One chat turn. history is a list of {'role': ..., 'content': ...}."""
    lower = message.lower()

    # A plain greeting gets the welcome message, not a retrieval miss.
    if lower.strip(" !.?,") in GREETINGS:
        return prompts.WELCOME

    # Academic-integrity guard: a request to WRITE graded work is refused
    # outright (the system prompt is the backstop for paraphrased asks).
    if (any(verb in lower for verb in ("write", "draft", "do my", "make my"))
            and any(noun in lower for noun in ("assignment", "essay",
                    "reflection", "discussion post", "homework"))):
        return prompts.INTEGRITY_REFUSAL

    # Quiz mode: "quiz me" is a request ABOUT the quiz bank, not similar in
    # meaning to its text, so embedding search can't find it — route straight
    # to the quiz-bank chunks instead.
    if any(w in lower for w in ("quiz", "test me", "practice question")):
        hits = [(c, 1.0) for c in CHUNKS if "### Question" in c["text"]]
        hits = hits[:rag.TOP_K]
    else:
        # RAG STEP 4 — RETRIEVE chunks relevant to the student's message.
        hits = rag.retrieve(message, CHUNKS, VECTORS)
    if not hits and history:
        # Short follow-ups ("B?", "yes", a quiz answer) carry little meaning
        # alone, so retry once with the previous bot message as context.
        # The welcome greeting carries no topic, so it never counts as context.
        last = history[-1]["content"]
        if last != prompts.WELCOME:
            hits = rag.retrieve(last + "\n" + message, CHUNKS, VECTORS)
    if not hits:
        return prompts.OUT_OF_SCOPE  # nothing relevant -> honest refusal

    # Which study guides the answer comes from (proof of grounding).
    sources = ", ".join(sorted({chunk["source"] for chunk, _ in hits}))

    if client is None:  # no API key configured -> still answer, from excerpts
        return excerpt_answer(hits, sources)

    # RAG STEP 5 — GENERATE the answer from the retrieved chunks.
    system = prompts.SYSTEM_PROMPT.format(
        out_of_scope=prompts.OUT_OF_SCOPE, context=format_context(hits))
    messages = history + [{"role": "user", "content": message}]

    try:
        answer = generate(system, messages)
    except Exception:
        # Any LLM failure (timeout, rate limit, credits, bad key): never show
        # an error — fall back to quoting the retrieved study-guide excerpt.
        return excerpt_answer(hits, sources)

    if prompts.OUT_OF_SCOPE in answer:  # a decline needs no Sources line
        return prompts.OUT_OF_SCOPE
    return f"{answer}\n\n*Sources: {sources}*"


# --- The chat page (Streamlit runs this file with __name__ == "__main__") ---
if __name__ == "__main__":
    st.set_page_config(page_title="ML Study Buddy", page_icon="🎓")
    st.title("ML Study Buddy 🎓")
    st.caption("Your AIML-500 study partner — explanations, quizzes, and "
               "study tips, grounded in the course study guides.")

    if "messages" not in st.session_state:  # the bot speaks first
        st.session_state.messages = [
            {"role": "assistant", "content": prompts.WELCOME}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_msg := st.chat_input("Ask about any AIML-500 topic..."):
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        history = st.session_state.messages[:-1]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat(user_msg, history)
            st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
