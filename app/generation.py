import os

PROMPT = (
    "Answer the question using ONLY the context below. "
    "If the context is insufficient, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def generate(question: str, contexts: list[str]) -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:  # extractive fallback so the pipeline runs without an LLM
        return contexts[0] if contexts else "I don't know."
    import anthropic

    resp = anthropic.Anthropic(api_key=key).messages.create(
        model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=500,
        messages=[{"role": "user", "content": PROMPT.format(
            context="\n\n".join(contexts), question=question)}],
    )
    return resp.content[0].text
