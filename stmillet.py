"""Streamlit Millet Knowledge chat interface backed by OpenAI Responses API."""

from __future__ import annotations

import html
import math
import os
import textwrap
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from openai.types.responses import Response
from pypdf import PdfReader


load_dotenv()


PAGE_TITLE = "Millet Knowledge Companion"
PAGE_ICON = "🌾"
PDF_NAME = "MilletsKnowledgeDataset.pdf"
DEFAULT_COMPLETION_MODEL = "gpt-5-chat"
EMBED_MODEL_FALLBACK = "text-embedding-3-large"


def set_page() -> None:
	st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
	st.markdown(
		"""
		<style>
			body {background-color: #f5f7fa;}
			.main-header {margin-bottom: 1.5rem;}
			.chat-history {
				max-height: 520px;
				overflow-y: auto;
				padding: 1.25rem;
				border-radius: 16px;
				border: 1px solid #dce1eb;
				background: linear-gradient(180deg,#ffffff 0%,#f6f9fc 100%);
				box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
			}
			.chat-bubble {margin-bottom: 1.25rem; line-height: 1.6;}
			.chat-bubble:last-child {margin-bottom: 0;}
			.chat-bubble .chat-role {font-weight: 600; margin-bottom: 0.4rem;}
			.chat-bubble.user {
				background: #124c3a;
				color: #f1fff7;
				border-radius: 14px;
				padding: 1rem;
			}
			.chat-bubble.assistant {
				background: #ffffff;
				color: #1f2933;
				border-radius: 14px;
				padding: 1rem;
				border: 1px solid #e3e9ed;
			}
			.chat-bubble .sources {margin-top: 0.75rem; font-size: 0.88rem; opacity: 0.9;}
			.chat-bubble .sources ul {padding-left: 1.2rem; margin: 0.35rem 0 0;}
			.chat-bubble .sources li {margin-bottom: 0.35rem;}
			.chat-bubble .sources li:last-child {margin-bottom: 0;}
			.placeholder {opacity: 0.7; text-align: center; margin: 2rem 0;}
			.stChatInputContainer textarea {border-radius: 999px; padding: 0.9rem 1.2rem;}
		</style>
		""",
		unsafe_allow_html=True,
	)


def build_pdf_path() -> Path:
	return Path(__file__).resolve().parent / "pdfs" / PDF_NAME


@st.cache_data(show_spinner=False)
def load_pdf_chunks(pdf_path: Path, chunk_size: int = 220, overlap: int = 40) -> List[Dict[str, str]]:
	reader = PdfReader(pdf_path)
	chunks: List[Dict[str, str]] = []
	for page_index, page in enumerate(reader.pages, start=1):
		page_text = page.extract_text() or ""
		words = page_text.split()
		start = 0
		while start < len(words):
			end = min(len(words), start + chunk_size)
			chunk_words = words[start:end]
			if not chunk_words:
				break
			chunk_text = " ".join(chunk_words).strip()
			if chunk_text:
				chunks.append({
					"page": page_index,
					"content": chunk_text,
				})
			start += max(1, chunk_size - overlap)
	return chunks


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
	dot = sum(a * b for a, b in zip(vec_a, vec_b))
	norm_a = math.sqrt(sum(a * a for a in vec_a))
	norm_b = math.sqrt(sum(b * b for b in vec_b))
	if not norm_a or not norm_b:
		return 0.0
	return dot / (norm_a * norm_b)


def create_openai_client() -> AzureOpenAI | OpenAI:
	api_key = (
		os.getenv("OPENAI_API_KEY")
		or os.getenv("AZURE_OPENAI_KEY")
		or os.getenv("AZURE_OPENAI_API_KEY")
	)
	if not api_key:
		raise RuntimeError("Missing OpenAI API key in environment variables.")

	endpoint = os.getenv("OPENAI_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
	api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")

	if endpoint:
		if not api_version:
			raise RuntimeError("Azure OpenAI endpoint detected but no API version provided.")
		return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

	return OpenAI(api_key=api_key)


def get_completion_model_name() -> str:
	return (
		os.getenv("OPENAI_DEPLOYMENT_NAME")
		or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
		or os.getenv("OPENAI_MODEL_NAME")
		or os.getenv("OPENAI_MODEL")
		or DEFAULT_COMPLETION_MODEL
	)


def get_embedding_model_name() -> str:
	return (
		os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
		or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
		or os.getenv("OPENAI_EMBEDDING_MODEL")
		or EMBED_MODEL_FALLBACK
	)


def ensure_embeddings(client: AzureOpenAI | OpenAI, embedding_model: str) -> None:
	if st.session_state.get("kb_embedded"):
		return

	chunks = st.session_state["knowledge_chunks"]
	if not chunks:
		raise RuntimeError("No chunks available from the PDF knowledge base.")

	embeddings: List[List[float]] = []
	batch_size = 16
	for start in range(0, len(chunks), batch_size):
		batch = [chunk["content"] for chunk in chunks[start:start + batch_size]]
		response = client.embeddings.create(model=embedding_model, input=batch)
		embeddings.extend([item.embedding for item in response.data])

	for chunk, vector in zip(chunks, embeddings):
		chunk["embedding"] = vector

	st.session_state["kb_embedded"] = True


def retrieve_context(
	client: AzureOpenAI | OpenAI,
	embedding_model: str,
	question: str,
	top_k: int = 4,
) -> List[Dict[str, str]]:
	ensure_embeddings(client, embedding_model)
	response = client.embeddings.create(model=embedding_model, input=[question])
	query_vector = response.data[0].embedding

	scored: List[Dict[str, float]] = []
	for chunk in st.session_state["knowledge_chunks"]:
		similarity = cosine_similarity(query_vector, chunk["embedding"])
		scored.append({"score": similarity, "chunk": chunk})

	scored.sort(key=lambda item: item["score"], reverse=True)
	top_chunks = [entry["chunk"] for entry in scored[:top_k]]
	return top_chunks


def format_prompt(question: str, context_chunks: List[Dict[str, str]]) -> str:
	context_blocks = []
	for chunk in context_chunks:
		block = f"Page {chunk['page']}: {chunk['content']}"
		context_blocks.append(block)
	context_text = "\n\n".join(context_blocks)
	return (
		"You are an expert assistant on millet crops."
		" Answer strictly using the supplied knowledge excerpts."
		" If the context does not cover the answer, say you do not know."
		"\n\nKnowledge excerpts:\n"
		f"{context_text}\n\nUser question: {question}"
	)


def call_responses_api(
	client: AzureOpenAI | OpenAI,
	model_name: str,
	prompt: str,
) -> Response:
	system_message = (
		"You are MilletGPT, a concise, trustworthy agronomy assistant focused on millets."
		" Highlight key facts, keep answers structured, and cite pages when helpful."
	)
	return client.responses.create(
		model=model_name,
		input=[
			{
				"role": "system",
				"content": [{"type": "input_text", "text": system_message}],
			},
			{
				"role": "user",
				"content": [{"type": "input_text", "text": prompt}],
			},
		],
		temperature=0.2,
		max_output_tokens=800,
	)


def message_to_html(message: Dict[str, str]) -> str:
	role = message["role"]
	label = "You" if role == "user" else "MilletGPT"
	html_lines = "<br>".join(html.escape(line) for line in message["content"].splitlines())
	bubble_class = "user" if role == "user" else "assistant"
	sources = message.get("sources", [])
	sources_html = ""
	if sources:
		source_items = []
		for source in sources:
			snippet = textwrap.shorten(
				" ".join(source["snippet"].split()),
				width=160,
				placeholder="…",
			)
			source_items.append(
				f"<li><strong>Page {source['page']}</strong> – {html.escape(snippet)}</li>"
			)
		sources_html = (
			"<div class='sources'><span>Context</span><ul>"
			+ "".join(source_items)
			+ "</ul></div>"
		)
	return (
		f"<div class='chat-bubble {bubble_class}'>"
		f"<div class='chat-role'>{label}</div>"
		f"<div class='chat-content'>{html_lines}</div>"
		f"{sources_html}</div>"
	)


def render_history(messages: List[Dict[str, str]]) -> None:
	if not messages:
		st.markdown(
			"<div class='chat-history'><p class='placeholder'>Ask anything about millets to get started.</p></div>",
			unsafe_allow_html=True,
		)
		return

	history_html = "".join(message_to_html(message) for message in messages)
	st.markdown(
		f"<div class='chat-history'>{history_html}</div>",
		unsafe_allow_html=True,
	)


def sidebar_controls():
	with st.sidebar:
		st.image(
			"https://images.unsplash.com/photo-1601000934645-d4aa1f0f02e7?auto=format&fit=crop&w=400&q=60",
			use_column_width=True,
			caption="Pearl millet field"
		)
		st.header("Session")
		if st.button("Start fresh", type="primary"):
			st.session_state["messages"] = []
			st.session_state.pop("kb_embedded", None)
			st.rerun()
		st.divider()
		st.markdown(
			"""
			**How it works**
			- Queries the Millets knowledge PDF
			- Finds the closest reference passages
			- Crafts a grounded answer with GPT-5.1
			"""
		)


def main() -> None:
	set_page()

	pdf_path = build_pdf_path()
	if not pdf_path.exists():
		st.error(f"Knowledge source not found at {pdf_path}")
		return

	sidebar_controls()

	st.markdown(
		"""
		<div class="main-header">
			<h1 style="margin-bottom:0.4rem;">🌾 Millet Knowledge Companion</h1>
			<p style="color:#475569; font-size:1.05rem;">Ask grounded questions about millets and get expertly curated answers sourced from the Millets Knowledge Dataset.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	if "messages" not in st.session_state:
		st.session_state["messages"] = []
	if "knowledge_chunks" not in st.session_state:
		st.session_state["knowledge_chunks"] = load_pdf_chunks(pdf_path)

	render_history(st.session_state["messages"])

	user_prompt = st.chat_input("Ask a question about millet cultivation, nutrition, or usage…")
	if not user_prompt:
		return

	st.session_state["messages"].append({"role": "user", "content": user_prompt})

	try:
		client = st.session_state.get("openai_client")
		if client is None:
			client = create_openai_client()
			st.session_state["openai_client"] = client
		completion_model = get_completion_model_name()
		embedding_model = get_embedding_model_name()

		with st.spinner("Consulting the millet knowledge base…"):
			context_chunks = retrieve_context(client, embedding_model, user_prompt)
			formatted_prompt = format_prompt(user_prompt, context_chunks)
			response = call_responses_api(client, completion_model, formatted_prompt)
			assistant_reply = response.output_text
			sources = [
				{
					"page": chunk["page"],
					"snippet": chunk["content"],
				}
				for chunk in context_chunks
			]

	except Exception as exc:  # noqa: BLE001
		st.session_state["messages"].append(
			{
				"role": "assistant",
				"content": "I ran into a configuration issue while contacting the model. "
				"Please verify your .env settings and try again.\n\n"
				f"Details: {exc}",
			}
		)
		st.rerun()
		return

	st.session_state["messages"].append(
		{
			"role": "assistant",
			"content": assistant_reply.strip(),
			"sources": sources,
		}
	)
	st.rerun()


if __name__ == "__main__":
	main()
