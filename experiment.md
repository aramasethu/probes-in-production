# Probing RAG Groundedness vs. Parametric Hallucination

## Objective

Obeso et al. (2026) show that a linear probe on late-layer hidden states can detect when an LLM is fabricating content with no grounding in its parametric memory. Their setup assumes no external context — the model is answering from what it "knows."

We want to investigate what happens to this signal when retrieval context is added to the prompt. In a RAG setting, "groundedness" has two possible sources: the model's parametric memory, and the provided context. A generated token can be supported by one, both, or neither — a distinction the original probe was never trained to make.

The central research question is whether the internal representation of "I lack grounding" is a single unified direction in activation space, or two separable signals (parametric vs. contextual grounding). Concretely, we plan to:

1. Replicate the paper's probe on Qwen in the original no-context setting.
2. Train a second probe on RAG groundedness labels (answer-vs-context).
3. Measure cross-transfer between the two probes, compare the directions they learn, and characterize which layers carry each signal.

A clean result in either direction is informative: shared direction → evidence for a general "ungrounded generation" representation; separate directions → evidence that the model explicitly represents the *source* of its information internally.

## Model

Llama-3.1-8B-Instruct. This is the paper's primary model, which gives us a direct replication baseline for Probe A and lets us reuse the paper's released annotated completions. Runs on M4 in 4-bit.

## Datasets

**Probe A (no-context hallucination)**: the paper's released dataset, filtered to Llama-3.1-8B completions. Already includes prompts, completions, and entity-level supported/unsupported labels.

**Probe B (RAG groundedness)**: RAGTruth QA subset, filtered to Llama-2-7B-chat completions (closest in distribution to Llama-3.1). Completions are teacher-forced through Llama-3.1-8B to extract hidden states — i.e., we feed `(context + question + existing completion)` and read activations at completion tokens. The paper validates this cross-model approach (their key finding that probes transfer between models means the supervision signal still works when the generator differs from the probed model).

Limitation: hidden states reflect what Llama-3.1 would think of this text, not text it actually generated. If cross-transfer results are noisy, the contingent next step is regenerating completions with Llama-3.1-8B directly.

## Out of scope (for now)

**Parametric-knowledge annotation.** To cleanly separate "model ignored context and used parametric memory" from "model fabricated freely," we would label each ungrounded span with whether the model produces the same content when re-prompted *without* the context. This is the most informative slice for the central research question (shared vs. separate directions), since it splits Probe B's ungrounded class into the two mechanistically distinct cases Probe A was originally trained on.

Deferring because: ~half a day to a full day of glue work (re-prompting + output comparison + validation), and it may not be needed. We'll run v1 with span-level grounded/ungrounded labels only and revisit if the cross-transfer results are ambiguous and need disambiguation.
