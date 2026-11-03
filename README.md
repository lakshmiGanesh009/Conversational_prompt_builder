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

The app reads the key from this environment variable at startup, so nothing
sensitive is stored in the code or in the repository.

## Running the app

With the virtual environment active and `OPENAI_API_KEY` set, start the app:

```bash
streamlit run app.py
```

Streamlit opens the chat interface in your browser. Type what you want to create,
answer the follow-up questions, and copy the generated prompt when you're happy with
it.

## Running the tests

```bash
pytest
```
