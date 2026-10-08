"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    human_messages = [
        message for message in state["messages"]
        if isinstance(message, HumanMessage)
    ]

    if not human_messages:
        return {
            "original_query": "",
            "rewritten_query": "",
        }

    original_query = human_messages[-1].content

    rewrite_prompt = (
        "Rewrite the following user question into a concise search query "
        "optimized for vector similarity retrieval from deep learning study "
        "materials. Preserve the technical meaning and important keywords. "
        "Return only the rewritten query.\n\n"
        f"User question: {original_query}"
    )

    try:
        llm = LLMFactory.create()
        response = llm.invoke(rewrite_prompt)
        rewritten_query = response.content.strip()

        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query or original_query,
        }

    except Exception:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }
# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    # TODO: implement
    # 1. Instantiate VectorStoreManager (consider caching this)
    # 2. manager.query(
    #        query_text=state.rewritten_query,
    #        topic_filter=state.topic_filter,
    #        difficulty_filter=state.difficulty_filter
    #    )
    # 3. If result is empty: return {"retrieved_chunks": [], "no_context_found": True}
    # 4. Otherwise: return {"retrieved_chunks": chunks, "no_context_found": False}
    manager = VectorStoreManager()

    chunks = manager.query(
        query_text=state["rewritten_query"],
        topic_filter=state.get("topic_filter"),
        difficulty_filter=state.get("difficulty_filter"),
    )

    if not chunks:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    return {
        "retrieved_chunks": chunks,
        "no_context_found": False,
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if state.get("no_context_found", False):
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.get("rewritten_query", ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
        # ---- Build Context from Retrieved Chunks -------------------------------
    context_parts = []
    sources = []

    for chunk in state.get("retrieved_chunks", []):
        citation = chunk.to_citation()
        sources.append(citation)

        context_parts.append(
            f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]\n"
            f"{chunk.chunk_text}\n"
        )

    context = "\n".join(context_parts)

    # Calculate average confidence score
    confidence = (
        sum(chunk.score for chunk in state.get("retrieved_chunks", []))
        / len(state.get("retrieved_chunks", []))
    )

    # Build messages for the LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(
            content=(
                "Use the following retrieved study material as context "
                "for answering the user's question:\n\n"
                f"{context}"
            )
        ),
    ]

    # Trim conversation history
    try:
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=settings.max_context_tokens,
            strategy="last",
            token_counter="approximate",
            include_system=False,
            start_on="human",
        )
        messages.extend(trimmed_history)
    except Exception:
        # Safe fallback if trimming is unavailable for the current model/setup
        messages.extend(state.get("messages", [])[-6:])

    # Add original user query if it is not already the final human message
    if not messages or not (
        isinstance(messages[-1], HumanMessage)
        and messages[-1].content == state.get("original_query", "")
    ):
        messages.append(HumanMessage(content=state.get("original_query", "")))

    # Generate answer
    llm_result = llm.invoke(messages)
    answer = (
        llm_result.content
        if hasattr(llm_result, "content")
        else str(llm_result)
    )

    # Build structured response
    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query", ""),
    )

    new_ai_message = AIMessage(content=answer)

    return {
        "final_response": response,
        "messages": [new_ai_message],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    if state.get("no_context_found", False):
        return "end"

    return "generate"