# Generative AI Helpdesk (EXP 7)

Prompt-based, domain-specific chatbot. Domain: **Generative AI**.
Pre-trained LLM + system prompt + domain facts in the prompt + Streamlit UI.
No fine-tuning, no retriever, no embeddings (that is Experiment 8).

## Run locally
    pip install -r requirements.txt
    mkdir -p .streamlit
    echo 'HF_TOKEN = "hf_your_token_here"' > .streamlit/secrets.toml
    streamlit run app.py

Add `.streamlit/secrets.toml` to `.gitignore`. Never commit the token.

## Run in Google Colab
    !pip install -q streamlit huggingface_hub
    # upload app.py, then:
    import os; from getpass import getpass
    os.environ["HF_TOKEN"] = getpass("Hugging Face token: ")
    !npm install -q localtunnel
    !streamlit run app.py &>/content/logs.txt &
    !npx localtunnel --port 8501
(Open the printed URL. The tunnel password is the Colab VM's public IP: `!curl ipv4.icanhazip.com`.)

## Deploy on Streamlit Community Cloud
Push app.py, requirements.txt, README.md to GitHub, create the app, and paste
`HF_TOKEN = "hf_..."` into Settings > Secrets.
