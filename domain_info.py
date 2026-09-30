"""Shared domain facts used by the app and batch evaluator."""

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

extra_facts = """
28. QLoRA combines LoRA with a quantized (for example 4-bit) base model so larger models can be fine-tuned on limited GPU memory.
29. Quantization reduces the numerical precision of model weights (for example from 16-bit to 4-bit) to cut memory use and speed up inference.
30. Top-p (nucleus) sampling picks the next token from the smallest set of tokens whose cumulative probability reaches p.
31. An AI agent is an LLM-based system that can plan steps and call external tools such as search or code execution.
32. Mixture of Experts (MoE) models route each token to a few specialised sub-networks, so only part of the model is active per token.
"""
