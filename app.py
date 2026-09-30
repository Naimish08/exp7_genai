"""
EXP 7 - Domain-Specific Generative AI Chatbot
Domain : Generative AI (same domain as the LoRA / fine-tuning experiment)
Method : Pre-trained LLM + system prompt + domain information in the prompt
NOT RAG: no retriever, no embeddings, no vector database.
"""

import os
import time

import streamlit as st
from huggingface_hub import InferenceClient

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
DOMAIN_NAME = "Generative AI"

st.set_page_config(page_title="GenAI Helpdesk", page_icon="🤖", layout="centered")

# ----------------------------------------------------------------------------
# Domain information (Step 4)  - supplied directly inside the prompt
# ----------------------------------------------------------------------------


domain_information = """
GENERATIVE AI HELP DESK INFORMATION

1. Generative AI refers to models that create new content such as text, images, audio, video and code by learning patterns from training data.
2. A Large Language Model (LLM) is a neural network, usually based on the Transformer architecture, trained on large text corpora to predict the next token.
3. The Transformer architecture was introduced in the 2017 paper "Attention Is All You Need" and relies on self-attention instead of recurrence.
4. A token is a small piece of text (a word or part of a word). Models read and produce tokens, and the context window is the maximum number of tokens a model can handle at once.
5. LLM development typically has stages: pre-training on large unlabelled data, supervised fine-tuning on instruction data, and alignment (for example RLHF) to follow human preferences.
6. Fine-tuning updates the model's parameters using a training dataset, which is computationally more expensive than prompting.
7. LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method. It freezes the base model weights and trains small low-rank adapter matrices.
8. Prompt engineering means designing instructions, context and examples so a model produces better outputs, without changing the model's parameters.
9. Zero-shot prompting gives no examples, few-shot prompting gives a few examples in the prompt, and chain-of-thought prompting asks the model to reason step by step.
10. Hallucination is when a model produces fluent text that is false or not supported by the provided information.
11. Temperature controls randomness in generation: low values give more deterministic output, high values give more varied output.
12. Retrieval-Augmented Generation (RAG) retrieves relevant documents using embeddings and adds them to the prompt before the model generates an answer.
13. Embeddings are numerical vectors that represent the meaning of text, images or other data so that similar items are close together.
14. Diffusion models generate images by starting from noise and gradually denoising it. GANs use a generator and a discriminator that compete with each other.
15. Common text-generation metrics include perplexity, BLEU and ROUGE. Human evaluation is also widely used for chatbot quality.
16. Responsible use of generative AI involves handling bias, privacy, copyright, misinformation and safety risks.
17. Agentic AI refers to AI systems that can act independently on behalf of users, make decisions, and take actions to achieve specified goals. Its purpose is to let systems autonomously perform tasks, solve problems, and adapt to changing environments with minimal human intervention.
18. The concept of AI agents dates back to the 1950s with early cybernetics and goal-seeking systems. Milestones include the Logic Theorist (1956), the STRIPS planning system (1971), the BDI architecture (1987), and modern LLM-based agents (2020s).
19. Key components of AI agents include perception components (text inputs, visual systems, data feeds), action mechanisms (generating text, controlling robotic components, making API calls), and knowledge (the information and facts the agent has access to).
20. Reactive agents respond directly to current environmental stimuli without maintaining internal state or considering past experiences. They offer simple, fast responses, minimal computational requirements and predictable behavior, and are typically implemented with condition-action rules or stimulus-response mappings.
21. LLM agent architectures use large language models as the core reasoning engine for autonomous agents. They typically include the LLM itself, prompt management, memory systems, tool integration and planning modules.
22. Tool use lets AI agents use external tools, APIs or systems to extend their capabilities. It involves tool discovery, selection, parameter setting, execution and result interpretation, for example web searches, code execution, database queries and API calls.
23. Tool-augmented LLM agents require defining tool interfaces, teaching the LLM how to use tools, and handling tool outputs.
24. Multi-agent systems are composed of multiple interacting autonomous agents that collectively solve problems beyond individual capabilities. They feature distributed control, decentralized data, asynchronous computation and emergent behavior, and require mechanisms for communication, negotiation, cooperation and conflict resolution.
25. Explainability in agent actions is the ability of agents to provide understandable justifications for their decisions and actions. It is critical for building trust, enabling oversight, facilitating debugging and meeting regulatory requirements, and includes transparent reasoning, post-hoc explanations and visualization of decision factors.
26. Human-agent collaboration is the cooperative relationship between humans and AI agents working toward shared goals. It includes delegation (human directs agent), partnership (shared responsibility) and oversight (human supervises agent).
27. Agency allows humans to delegate complex or repetitive tasks to AI systems that can operate independently, and agents can handle complex multi-step processes that would be difficult to specify completely in advance.
"""

# Modification 2: five additional facts the student can switch on in the sidebar
extra_facts = """
28. QLoRA combines LoRA with a quantized (for example 4-bit) base model so larger models can be fine-tuned on limited GPU memory.
29. Quantization reduces the numerical precision of model weights (for example from 16-bit to 4-bit) to cut memory use and speed up inference.
30. Top-p (nucleus) sampling picks the next token from the smallest set of tokens whose cumulative probability reaches p.
31. An AI agent is an LLM-based system that can plan steps and call external tools such as search or code execution.
32. Mixture of Experts (MoE) models route each token to a few specialised sub-networks, so only part of the model is active per token.
"""

# ----------------------------------------------------------------------------
# Hugging Face client (Steps 5 and 6) - token comes from secrets, never hard-coded
# ----------------------------------------------------------------------------
def load_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN")


HF_TOKEN = load_token()

# Use an instructor-approved chat model that is currently available to you.
# Change it here or in the sidebar. Availability on the free tier changes often.
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_OPTIONS = [
    DEFAULT_MODEL,
    "Qwen/Qwen3-4B-Instruct-2507",
    "Custom model ID",
]

# ----------------------------------------------------------------------------
# Prompts (Step 7 + Modification 1)
# ----------------------------------------------------------------------------
def system_prompt_b(info: str) -> str:
    """Prompt B - domain-specific system prompt."""
    return f"""You are a helpful domain-specific AI assistant for the domain: {DOMAIN_NAME}.

Use the following domain information when answering the question.

DOMAIN INFORMATION:
{info}

INSTRUCTIONS:
1. Answer clearly and concisely.
2. Use the domain information when relevant.
3. Do not intentionally invent domain-specific facts.
4. If the information is not available, clearly state that the information is not available in the provided information.
5. Do not answer questions outside the domain unless general information is explicitly requested.
"""


SYSTEM_PROMPT_A = "Answer the following question."  # Prompt A - basic, no domain info


def build_messages(question, history, prompt_mode, info):
    system = SYSTEM_PROMPT_A if prompt_mode.startswith("A") else system_prompt_b(info)
    messages = [{"role": "system", "content": system}]
    for turn in history[-6:]:  # keep the last 6 turns so the prompt stays small
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


# ----------------------------------------------------------------------------
# Generation (Step 8)
# ----------------------------------------------------------------------------
def generate_answer(question, history, model_name, prompt_mode, info, temperature):
    client = InferenceClient(token=HF_TOKEN, provider="auto")
    messages = build_messages(question, history, prompt_mode, info)
    start = time.time()
    response = client.chat_completion(
        messages=messages,
        model=model_name,
        max_tokens=300,
        temperature=temperature,
    )
    latency = time.time() - start
    return response.choices[0].message.content.strip(), latency


# ----------------------------------------------------------------------------
# Session state (Step 15)
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ----------------------------------------------------------------------------
# Sidebar (Step 14)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.write(f"**Domain:** {DOMAIN_NAME}")

    selected_model = st.selectbox("Model", MODEL_OPTIONS)
    model_name = (
        st.text_input("Custom Hugging Face chat model ID", value=DEFAULT_MODEL)
        if selected_model == "Custom model ID"
        else selected_model
    )
    prompt_mode = st.radio(
        "Prompt type",
        ["B - Domain-specific", "A - Basic"],
        help="Compare both for Modification 1.",
    )
    use_extra = st.checkbox("Add 5 extra facts (Modification 2)", value=False)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    st.subheader("Example questions")
    examples = [
        "What is agentic AI and what is its purpose?",
        "What are reactive agents?",
        "Why is explainability in agent actions important?",
        "What is a multi-agent system?",
        "What is the capital of France?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

active_info = domain_information + (extra_facts if use_extra else "")

# ----------------------------------------------------------------------------
# Main interface (Step 12)
# ----------------------------------------------------------------------------
st.title("🤖 Generative AI Helpdesk")
st.caption(
    f"Domain: {DOMAIN_NAME}. Answers come from a pre-trained LLM guided by a prompt; "
    "the model itself is not trained or modified."
)

if not HF_TOKEN:
    st.error(
        "No Hugging Face token found. Add HF_TOKEN to .streamlit/secrets.toml "
        "(or the Streamlit Cloud secrets box), or set the HF_TOKEN environment variable."
    )
    st.stop()

# Show conversation history
for turn in st.session_state.messages:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(f"{turn['mode']} | {turn['latency']:.1f}s")

typed = st.chat_input("Ask a question about Generative AI and agentic AI")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, latency = generate_answer(
                    question,
                    st.session_state.messages,
                    model_name,
                    prompt_mode,
                    active_info,
                    temperature,
                )
            st.write(answer)
            st.caption(f"{prompt_mode} | {latency:.1f}s")
            st.session_state.messages.append(
                {
                    "question": question,
                    "answer": answer,
                    "mode": prompt_mode,
                    "latency": latency,
                }
            )
        except Exception as e:
            st.error(
                "The model request failed. The model may be unavailable on your "
                "provider, or the token may lack inference permission. "
                f"Details: {e}"
            )