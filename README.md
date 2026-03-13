# AI Agent Testing Lab: Context, Evals, and Observability 101

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/m3kessler1/evallab)

A 90-minute Live Lab for testing AI Agents with **LangChain**, **Langfuse**, and **DeepEval**.

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand why traditional assertions fail for AI agents
- Build a non-deterministic agent with LangChain
- Implement observability with Langfuse tracing
- Write automated evaluations using DeepEval's LLM-as-a-judge

## 🚀 Quick Start

### 1. Open in Codespaces
Click the badge above or go to **Code → Codespaces → Create codespace on main**

### 2. Configure Environment
Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your keys:
- `LLM_PROVIDER` — Set to `openai`, `moonshot`, or `openrouter`
- `OPENAI_API_KEY` — Get from [platform.openai.com](https://platform.openai.com) when using `openai`
- `MOONSHOT_API_KEY` — Get from [platform.moonshot.ai](https://platform.moonshot.ai) when using `moonshot`
- `OPENROUTER_API_KEY` — Get from [openrouter.ai/keys](https://openrouter.ai/keys) when using `openrouter`
- `LANGFUSE_SECRET_KEY` — Get from [cloud.langfuse.com](https://cloud.langfuse.com)
- `LANGFUSE_PUBLIC_KEY` — Get from [cloud.langfuse.com](https://cloud.langfuse.com)

Provider-specific model settings are available in `.env.example`:
- `OPENAI_MODEL`
- `MOONSHOT_MODEL`
- `OPENROUTER_MODEL`

### 3. Verify Setup

```bash
python -c "import langchain, langfuse, deepeval; print('✅ All packages installed')"
```

## 📁 Lab Structure

```
evallab/
├── .devcontainer/          # Codespace configuration
├── src/
│   ├── __init__.py
│   ├── tools.py           # Tool definitions (warranty checker, KB search)
│   └── agent.py           # Agent executor with LangChain
├── tests/
│   ├── __init__.py
│   └── test_agent.py      # DeepEval test suite
├── exercises/
│   ├── exercise_1_trace.py    # Add Langfuse observability
│   └── exercise_2_eval.py     # Write DeepEval assertions
├── .env.example           # Environment template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎓 Lab Segments

### Segment 1: The Agent Problem (00-20m)
**Concept:** Why traditional QA fails for AI

Key insight: Just as hardcoded XPaths break when the DOM changes, exact string assertions break when LLMs paraphrase.

### Segment 2: The Build (20-40m)
**Hands-on:** Build the Oak Brook IT Support Agent

```bash
# Run the baseline agent
python src/agent.py
```

Explore:
- `src/tools.py` — Pydantic-guarded tool definitions
- `src/agent.py` — Agent executor with tool-calling

### Segment 3: The Trace (40-65m)
**Hands-on:** Add Langfuse observability

Complete `exercises/exercise_1_trace.py`:
- Inject the CallbackHandler
- Generate traces in Langfuse Cloud
- Debug the "VIP Contradiction" infinite loop

### Segment 4: The Eval (65-90m)
**Hands-on:** Write automated evaluations

Complete `exercises/exercise_2_eval.py`:
- Implement Answer Relevancy metric
- Implement Faithfulness metric
- Run: `deepeval test run tests/test_agent.py`

## 🔑 Key Concepts

| Traditional QA | AI Agent QA |
|---------------|-------------|
| `assert "completed" in response` | Semantic evaluation (LLM-as-judge) |
| Static mocking | End-to-end trajectory tracing |
| Exception catching | Observability of reasoning paths |
| Playwright Trace Viewer | Langfuse cognitive timeline |

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com)
- [Langfuse Documentation](https://langfuse.com/docs)
- [DeepEval Documentation](https://docs.confident-ai.com)

## 🤝 Support

Stuck? Check the `solutions/` branch for reference implementations.

---

**Speaker:** Mike Kessler  
**Session:** Testing AI Agents: Context, Evals, and Observability 101