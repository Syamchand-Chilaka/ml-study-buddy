# ML Study Buddy 🎓

A RAG (Retrieval-Augmented Generation) chatbot for AIML-500 — Machine
Learning Fundamentals. It answers **only** from the three course study guides
in `kb/`, shows its sources under every answer, can quiz you one question at
a time, and politely refuses to write graded work or discuss off-topic
subjects.

## How it works (the 5 RAG steps)

1. **Ingest** — `rag.py` loads the three markdown guides from `kb/`.
2. **Chunk** — each guide is split on headings, then packed into ~500-token
   chunks with ~50 tokens of overlap.
3. **Embed** — every chunk becomes a vector via the free
   `all-MiniLM-L6-v2` sentence-transformer (runs on CPU).
4. **Retrieve** — a question is embedded the same way; the top 4 chunks by
   cosine similarity are selected. Below a similarity threshold, the bot
   honestly says the question is outside its study guides.
5. **Generate** — the chunks (with their file names) are placed into the
   system prompt (`prompts.py`) and the LLM (GLM 5.3 Flash on Fireworks AI's
   OpenAI-compatible API) writes a grounded answer, shown with a
   "Sources:" line (`app.py`). If the LLM is ever unreachable, the bot
   falls back to quoting the best study-guide excerpt instead of erroring.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat UI + generation (RAG step 5) |
| `rag.py` | Ingest, chunk, embed, retrieve (RAG steps 1–4) |
| `prompts.py` | System prompt, welcome message, refusal texts |
| `kb/*.md` | The three course study guides (the knowledge base) |
| `test_rag.py` | Sanity tests for the retrieval pipeline |
| `requirements.txt` | Pinned dependencies |

## Deploy to Streamlit Community Cloud (free, public URL)

1. Push this folder to a GitHub repository (e.g. `ml-study-buddy`).
2. Go to <https://share.streamlit.io> and **sign in with GitHub**.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `YOUR_GITHUB_USERNAME/ml-study-buddy`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a name like `ml-study-buddy` (this becomes
     `https://ml-study-buddy.streamlit.app`)
5. Open **Advanced settings** before deploying and paste this into the
   **Secrets** box (TOML format — the quotes matter), with your real
   Fireworks AI key (create one at <https://app.fireworks.ai> under
   Settings → API Keys):

   ```toml
   LLM_API_KEY = "fw_your-key-here"
   ```

6. Click **Deploy**. The first build takes several minutes (it installs
   PyTorch and downloads the embedding model). Your public link is the
   `https://....streamlit.app` URL — that's what you submit.

Note: free apps sleep after ~12 hours without visitors. The first visitor
clicks the wake-up button and waits about a minute — mention this in your
submission so your professor isn't surprised.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'LLM_API_KEY=fw_your-key-here' > .env   # your Fireworks AI key
streamlit run app.py                          # opens http://localhost:8501
```

Run the pipeline tests (no API key needed):

```bash
python test_rag.py
```

## Switching providers

Generation runs on Fireworks AI (OpenAI-compatible). To use OpenAI itself,
drop the `base_url` and set `MODEL = "gpt-4o-mini"` in `app.py`. For
Anthropic, swap `generate()` for a `claude.messages.create(...)` call with
the `anthropic` package. To try a different Fireworks model, change `MODEL`
to any serverless model id from <https://fireworks.ai/models>.

## Updating the knowledge base

Replace or edit the markdown files in `kb/` and restart the app — the index
is rebuilt from whatever is in that folder at startup. No code changes needed.
