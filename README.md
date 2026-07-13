# 🐦 AI Tweet Generator — A Self-Improving Content Agent built with LangGraph

An autonomous multi-agent workflow that writes, critiques, and rewrites tweets until they're good enough to post — no human in the loop. Built to explore how far you can push **agentic orchestration** with open-source LLMs instead of relying on a single prompt-and-pray call to GPT.

Think of it as three "personalities" working together:
- ✍️ **The Writer** — drafts a tweet on a given topic
- 🧐 **The Editor** — grades it against a strict rubric and returns structured feedback
- 🔁 **The Rewriter** — takes that feedback and improves the draft

...and it loops between the Editor and the Rewriter until the tweet is approved or it hits a max number of iterations. This is what "agentic" actually means in practice — not just calling an LLM, but giving it a feedback loop and letting it grade its own work.

---

## Why I built this

I wanted a hands-on project that went beyond a basic chatbot wrapper and actually exercised the parts of LLM engineering that matter in production: state machines, structured output validation, conditional branching, and stitching together multiple models with different jobs. **LangGraph** was the natural fit — it lets you model an LLM pipeline as an actual graph with nodes, edges, and routing logic, instead of a linear chain.

## How it works

```
        ┌─────────────┐
        │  generate   │  ← drafts a tweet from the topic
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  evaluate   │  ← scores the tweet against a strict rubric
        └──────┬──────┘
               │
      approved?│  needs_improvement?
       │       └────────────┐
       ▼                    ▼
      END              ┌──────────┐
                        │ optimize │  ← rewrites using the feedback
                        └────┬─────┘
                             │
                             └────► back to evaluate
```

**The nodes:**

| Node | Job | Model |
|---|---|---|
| `generate_tweet` | Writes the first draft using a hook-driven, "sounds human" system prompt | Qwen2.5-7B-Instruct |
| `evaluate_tweet` | Grades the tweet on hook strength, clarity, technical accuracy, engagement potential, and human tone — returns a structured `approved` / `needs_improvement` verdict with feedback | DeepSeek-R1-Distill-Qwen-7B |
| `optimize_tweet` | Rewrites the tweet to address the reviewer's feedback without changing the underlying facts | DeepSeek-R1-Distill-Qwen-7B |

The **routing logic** (`route_evaluation`) is what makes this a real agent rather than a fixed pipeline: it checks the evaluator's verdict and either ends the graph or sends the tweet back for another pass, up to a configurable `max_iterations`.

Evaluation output is enforced with a **Pydantic schema** (`TweetEvaluation`) parsed via `PydanticOutputParser`, so the graph never has to deal with the model rambling instead of returning clean JSON.

## Tech stack

- **LangGraph** — state machine / orchestration layer
- **LangChain** (`langchain_huggingface`) — model wrappers and message types
- **Hugging Face Inference Endpoints** — serving Qwen2.5-7B-Instruct and DeepSeek-R1-Distill-Qwen-7B
- **Pydantic** — structured output validation for the evaluator
- **python-dotenv** — environment/config management

**Built by [Yakub Naik]** — feel free to reach out if you'd like to chat about agentic workflows, LangGraph, or LLM engineering in general.
