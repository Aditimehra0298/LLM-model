import os
import io
import csv
import json
import torch
import streamlit as st
from typing import Optional
from transformers import AutoTokenizer
from PyPDF2 import PdfReader
try:
	from openai import OpenAI
except Exception:
	OpenAI = None

from inference import load_model, generate_text


def load_cached_model(model_dir: str):
	# Cache model + tokenizer across reruns
	@st.cache_resource
	def _load(dir_path: str):
		model, tokenizer = load_model(dir_path)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model.to(device)
		return model, tokenizer, device
	return _load(model_dir)


def build_prompt(
    facts: str,
    mode: str,
    num_items: int,
    style: str,
    concise: bool,
    max_words: int,
    standard: str = "None",
    clause: str = "",
    objective: str = "",
) -> str:
	constraints = []
	if concise:
		constraints.append("Be concise and avoid redundancy.")
	if max_words > 0:
		constraints.append(f"Each item must be no more than {max_words} words.")
	constraints_text = (" " + " ".join(constraints)).strip()

	# Parameter headers (ISO-style) to guide extraction
	param_headers = []
	if standard and standard != "None":
		param_headers.append(f"Standard: {standard}")
	if clause:
		param_headers.append(f"Clause: {clause}")
	if objective:
		param_headers.append(f"Objective: {objective}")
	params_block = ("\n" + " | ".join(param_headers) + "\n") if param_headers else "\n"

	if mode == "Questions":
		instruction = (
			f"You are an auditor. Based on the provided facts, generate {num_items} clear, specific audit questions. "
			f"Write questions in {style} tone. Number the questions 1..{num_items}. "
			+ constraints_text
		)
	else:
		instruction = (
			f"You are a technical writer. Based on the provided facts, generate {num_items} content paragraphs. "
			f"Write in a {style} tone. Number the paragraphs 1..{num_items}. "
			+ constraints_text
		)

	prompt = (
		"Facts:\n" + facts.strip() + params_block + "\n" +
		instruction + "\n\nOutput:\n"
	)
	return prompt


st.set_page_config(page_title="LLM Question/Content Generator", page_icon="🧠", layout="centered")
st.title("🧠 LLM Question/Content Generator")
st.caption("Provide your own facts; the model will generate questions or content.")

with st.sidebar:
	st.header("Backend")
	backend = st.radio("Select backend", options=["Local model", "OpenAI"], index=0)

	st.header("Model Settings")
	default_model_dir = "./output"
	model_dir = st.text_input("Model directory", value=default_model_dir)
	max_length = st.slider("Max tokens to generate", min_value=32, max_value=512, value=200, step=8)
	temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.9, step=0.1)
	colk, colp, colr = st.columns(3)
	with colk:
		top_k = st.number_input("top_k", min_value=0, max_value=200, value=50)
	with colp:
		top_p = st.slider("top_p", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
	with colr:
		rep_penalty = st.slider("repetition_pen.", min_value=1.0, max_value=2.0, value=1.0, step=0.05)

	if backend == "OpenAI":
		st.divider()
		st.subheader("OpenAI Settings")
		openai_model = st.text_input("Model", value="gpt-4o-mini")
		openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))

	load_button = st.button("Load / Reload Model", type="primary")

if backend == "Local model":
	if model_dir:
		try:
			model, tokenizer, device = load_cached_model(model_dir)
			if load_button:
				# Force reload by clearing cache and re-calling
				load_cached_model.clear()
				model, tokenizer, device = load_cached_model(model_dir)
				st.success("Model reloaded.")
		except Exception as e:
			st.error(f"Failed to load model from {model_dir}: {e}")
			st.stop()
	else:
		st.warning("Please provide a model directory.")
		st.stop()
else:
	if OpenAI is None:
		st.error("OpenAI SDK not installed. Please install 'openai' package.")
		st.stop()

st.subheader("Upload a file with facts (TXT, CSV, PDF)")
uploaded = st.file_uploader("Choose a file", type=["txt", "csv", "pdf"], accept_multiple_files=False)

def extract_text_from_upload(uploaded_file) -> str:
	if uploaded_file is None:
		return ""
	name = uploaded_file.name.lower()
	# TXT
	if name.endswith(".txt"):
		return uploaded_file.getvalue().decode("utf-8", errors="ignore")
	# CSV: concatenate cells row-wise
	if name.endswith(".csv"):
		content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
		reader = csv.reader(io.StringIO(content))
		rows = [", ".join([c.strip() for c in row if c is not None and c.strip() != ""]) for row in reader]
		return "\n".join([r for r in rows if r])
	# PDF
	if name.endswith(".pdf"):
		pdf_bytes = io.BytesIO(uploaded_file.getvalue())
		try:
			reader = PdfReader(pdf_bytes)
			if reader.is_encrypted:
				try:
					reader.decrypt("")  # try empty password
				except Exception:
					return ""  # cannot decrypt; return empty so UI shows need for plaintext
			texts = []
			for page in reader.pages:
				try:
					texts.append(page.extract_text() or "")
				except Exception:
					continue
			return "\n".join(texts)
		except Exception as e:
			return ""  # fall back to empty on PDF parsing errors
	# Fallback
	return uploaded_file.getvalue().decode("utf-8", errors="ignore")

facts = extract_text_from_upload(uploaded)
if uploaded is not None:
	st.caption(f"Loaded: {uploaded.name}")
	with st.expander("Preview extracted text", expanded=False):
		st.code((facts[:2000] + ("\n..." if len(facts) > 2000 else "")).strip())
else:
	st.info("Upload a TXT, CSV, or PDF file to proceed.")

col1, col2, col3 = st.columns(3)
with col1:
	mode = st.selectbox("Generate", options=["Questions", "Content"], index=0)
with col2:
	num_items = st.number_input("How many items?", min_value=1, max_value=20, value=5)
with col3:
	style = st.selectbox("Tone/Style", options=["formal", "neutral", "conversational"], index=0)

# Conciseness controls
cc1, cc2 = st.columns(2)
with cc1:
	concise = st.checkbox("Concise mode", value=True)
with cc2:
	max_words = st.number_input("Max words per item", min_value=0, max_value=200, value=20, help="0 disables the limit")

# ISO-style parameter headers
st.subheader("Compliance context (optional)")
ph1, ph2, ph3 = st.columns(3)
with ph1:
	standard = st.selectbox(
		"Standard",
		options=["None", "ISO 9001", "ISO 22000", "ISO 27001", "FSSC 22000"],
		index=0,
	)
with ph2:
	clause = st.text_input("Clause/Requirement", value="")
with ph3:
	objective = st.text_input("Audit objective / Content objective", value="")

run = st.button("Generate", type="primary", disabled=(facts.strip() == ""))

if run:
	with st.spinner("Generating..."):
		prompt = build_prompt(
			facts=facts,
			mode=mode,
			num_items=int(num_items),
			style=style,
			concise=concise,
			max_words=int(max_words),
			standard=standard,
			clause=clause,
			objective=objective,
		)
		try:
			if backend == "Local model":
				output = generate_text(
					model=model,
					tokenizer=tokenizer,
					prompt=prompt,
					max_length=max_length,
					temperature=temperature,
					top_k=int(top_k),
					top_p=float(top_p),
					repetition_penalty=float(rep_penalty),
					device=device,
				)
			else:
				if not openai_api_key:
					st.error("Please provide an OpenAI API key.")
					st.stop()
				client = OpenAI(api_key=openai_api_key)
				resp = client.chat.completions.create(
					model=openai_model,
					messages=[{"role":"user","content": prompt}],
					temperature=float(temperature),
					max_tokens=int(max_length),
				)
				output = resp.choices[0].message.content or ""
			st.divider()
			st.markdown("### Output")
			st.code(output.strip())
			st.download_button(
				label="Download Output",
				data=output,
				file_name=f"generation_{mode.lower()}.txt",
				mime="text/plain",
			)
		except Exception as e:
			st.error(f"Generation failed: {e}")
else:
	st.info("Enter facts and click Generate.")
