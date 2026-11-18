# Conversational Prompt Builder

Writing a good prompt is hard. You have to know what details matter, how to
structure them, and what to leave out. Conversational Prompt Builder does that
part for you.

Instead of asking you to write a prompt, the app has a conversation with you. You
describe what you want in plain language, the app asks a few focused follow-up
questions to fill in the gaps (things like who it's for, the tone, the format, and
how long it should be), and then it writes a polished, high-quality prompt based on
everything you shared.

No jargon, no templates to fill in, no prompt-engineering know-how required. You
talk, it listens, and it hands you a prompt you can use right away.

## What it does

- **Guided, multi-turn conversation.** The app interviews you one step at a time,
  asking only the questions it still needs answered.
- **Understands what you mean.** It reads your intent and keeps track of the details
  you've already given, so you never have to repeat yourself.
- **Synthesizes a high-quality prompt.** When it has enough to work with, it writes a
  complete prompt that pulls together everything you told it.
- **Review and refine.** You can read the result, ask for changes in plain language,
  and copy the finished prompt with one click.

It's a small, local app with a simple chat interface. There are no accounts and
nothing is saved between visits: your conversation lives only while you're using the
app and clears when you start a new one.

## Requirements

- Python 3.11 or newer
- An API key for a hosted, OpenAI-compatible language model (see
  [Configuration](#configuration) below)

## Setup

Create and activate a virtual environment, then install the dependencies.

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

The app talks to a hosted, OpenAI-compatible language model API. Supply your API key
through the `OPENAI_API_KEY` environment variable before running the app:

**macOS / Linux**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Windows (PowerShell)**

```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

The app reads the key from the environment first and then from Streamlit secrets,
so nothing sensitive needs to be stored in the code or committed to the repository.
For local TOML-based configuration, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml`; the real file is ignored by Git.

## Running the app

With the virtual environment active and `OPENAI_API_KEY` set, start the app:

```bash
streamlit run app.py
```

Streamlit opens the chat interface in your browser. Type what you want to create,
answer the follow-up questions, and copy the generated prompt when you're happy with
it.

## Deploying to Streamlit Community Cloud

1. In the Community Cloud workspace, choose **Create app** and select repository
   `lakshmiGanesh009/Conversational_prompt_builder`, branch `main`, and main file
   `app.py`.
2. Open **Advanced settings**, select Python **3.11**, and paste the required secret
   into the Secrets editor using the key shown in `.streamlit/secrets.toml.example`.
   Add `OPENAI_BASE_URL` only when using a compatible non-default endpoint.
3. Before adding the secret, restrict app access to the intended viewers and set a
   provider-side spend limit. The app uses the owner's server-side API key and does
   not include per-user authentication, rate limiting, or quotas. Do not expose it
   publicly unless that usage and cost risk is acceptable.
4. Deploy the app. Community Cloud installs the root `requirements.txt` and reads
   the checked-in `.streamlit/config.toml`. Never commit the real
   `.streamlit/secrets.toml` file.

## Observability and SLOs

Pick **Observability** in the sidebar to see how the agent is performing: every
model call, the service level objectives (SLOs) measured from them, and how much
error budget is left. Pick **Builder** to go back to building prompts.

**What the dashboard shows**

- Key numbers: calls, success rate, latency p50/p95/p99, tokens, and an estimated cost.
- A scorecard per SLO: status, current value, the objective, and the error budget
  (or headroom) remaining.
- Trends for latency, traffic, token usage, and errors; a breakdown by agent step
  (interpret, generate, revise); and a filterable table of recent calls.
- CSV and JSON export, and a choice of time window.

**The objectives**

| ID | Objective | Target |
| --- | --- | --- |
| SLO-01 | Agent call availability (including calls that fail before sending, such as a missing API key) | at least 99% |
| SLO-02 | Interpretation latency | p95 at most 6 s |
| SLO-03 | Prompt generation latency | p95 at most 30 s |
| SLO-04 | Revision latency | p95 at most 30 s |
| SLO-05 | Complete outputs: not cut off at the max-tokens limit | at least 99% |
| SLO-06 | Turn success rate: a message is processed without an error banner | at least 98% |
| SLO-07 | Turn latency, including every model call in the turn | p95 at most 40 s |
| SLO-08 | Retry recovery: a retry run succeeds within the 3-attempt cap | at least 90% |
| SLO-09 | Prompt completeness: no required parameter is omitted | at least 90% |
| SLO-10 | Token budget per call (input plus output) | p95 at most 6,000 tokens |

The objectives are defined in one place, `prompt_builder/slo.py`; edit the targets
there. An objective shows **Met**, **At risk** (still met, but under 25% of its error
budget or within 20% of its limit), **Breached**, or **Not enough data** (fewer than
5 measurements, so any verdict would be noise). *Burn rate* is how fast the error
budget is being spent: 1.0x means failing at exactly the allowed rate.

**Good to know**

- Metrics are kept **in memory** for the running server (the most recent 5,000 calls
  and 5,000 events). They are shared by every visitor and reset when the server
  restarts. Use the CSV/JSON export to keep a copy.
- Only timings, counts, token numbers, model names, and short error text (API keys
  are redacted) are recorded. Your messages and generated prompts are **never**
  stored in telemetry.
- The cost figure is a rough estimate from list prices in
  `prompt_builder/telemetry.py` (`PRICING_PER_M_TOKENS`). Check your provider's
  pricing and edit the table; models not listed show no cost.
- **Maintenance → Add sample data** fills the dashboard with clearly labelled
  synthetic traffic so you can explore it without an API key. **Clear all metrics**
  removes everything, for all visitors.
- On a public deployment the dashboard is visible to everyone who can open the app,
  and the Maintenance actions affect the shared store. Restrict access if that
  matters for your use.

## Running the tests

```bash
pytest
```
