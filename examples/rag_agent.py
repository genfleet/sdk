"""
RAG-powered agent using LightRAG for knowledge retrieval.

Native RAG support via Agent(data=...) is planned but not yet implemented.
This example shows the current recommended approach: implement AgentProtocol
directly and inject retrieved context into the system prompt per-query.

Requires:
    pip install 'withfleet-sdk[openai,serve]' lightrag-hku[openai]

Index documents first:
    python -c "
    import asyncio
    from rag_agent import index
    asyncio.run(index('Your company knowledge goes here...'))
    "

Run:
    export OPENAI_API_KEY=sk-...
    python examples/rag_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H 'Content-Type: application/json' \
      -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tasks/send\",\"params\":{\"id\":\"t1\",\"message\":{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"text\":\"What are your business hours?\"}]}}}' \
      | python -m json.tool
"""

import os
from functools import partial
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from fleet.sdk import AgentInput, AgentOutput
from fleet.sdk.serve import serve


RAG_STORAGE = str(Path(__file__).parent / "rag_storage")

SYSTEM_PROMPT = """You are a helpful assistant.
Base your answers on the retrieved context provided below.
If the context doesn't cover the question, say so honestly.
"""


def _build_rag() -> LightRAG:
    api_key = os.environ["OPENAI_API_KEY"]

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            "gpt-4o-mini", prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=partial(openai_embed, model="text-embedding-3-small", api_key=api_key),
    )

    Path(RAG_STORAGE).mkdir(parents=True, exist_ok=True)
    return LightRAG(working_dir=RAG_STORAGE, llm_model_func=llm_func, embedding_func=embedding_func)


_rag: LightRAG | None = None


async def get_rag() -> LightRAG:
    global _rag
    if _rag is None:
        _rag = _build_rag()
        await _rag.initialize_storages()
    return _rag


async def index(text: str) -> None:
    """Index a text document into the RAG storage."""
    await (await get_rag()).ainsert(text)
    print("Indexed.")


class RAGAgent:
    async def run(self, input: AgentInput):
        from openai import AsyncOpenAI

        # Retrieve relevant context for this query
        rag = await get_rag()
        context = await rag.aquery(input.message, param=QueryParam(mode="naive"))

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n--- RETRIEVED CONTEXT ---\n{context}\n--- END CONTEXT ---",
            }
        ]
        for msg in input.history:
            messages.append({"role": msg.role, "content": msg.content or ""})
        messages.append({"role": "user", "content": input.message})

        stream = await client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield AgentOutput(content=delta.content, done=False)

        yield AgentOutput(content="", done=True)


if __name__ == "__main__":
    serve(RAGAgent(), name="rag-agent", description="RAG-powered agent with LightRAG knowledge base", port=8000)
