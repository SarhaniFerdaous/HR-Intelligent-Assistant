# 🤖 HR Intelligence Assistant

An autonomous AI-powered HR assistant that answers employee questions from verified sources only — never from general knowledge.

Built with **LangChain**, **ChromaDB**, **Groq/Llama 3.3**, and **Streamlit**.

---

## What It Does

- **Answers HR questions** from your uploaded company documents (PDF, DOCX, TXT, CSV)
- **Fetches public holidays** by country in real time
- **Converts currencies** for international salary questions
- **Checks compliance** against internal company policies
- **Finds the right HR contact** for any topic
- **Responds in the same language** as the employee (EN / FR / AR)
- **Shows its reasoning steps** transparently

---

## Architecture

The agent uses **3 controlled data sources** — it never hallucinates:

| Source | Technology | Content |
|--------|-----------|---------|
| RAG | ChromaDB + all-MiniLM-L6-v2 | Uploaded HR documents |
| APIs | date.nager.at / open.er-api.com | Real-time data |
| Memory | LangChain + Streamlit session | Conversation history |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| Interface | Streamlit |
| LLM | Groq / Llama 3.3 70B Versatile |
| Agent Framework | LangChain + langchain-classic |
| Vector Store | ChromaDB (local) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, free, local) |
| Holiday API | date.nager.at |
| Currency API | open.er-api.com |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SarhaniFerdaous/HR-Intelligent-Assistant.git
cd HR-Intelligent-Assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file
Create a file called `.env` in the project root and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free API key at https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
hr-assistant/
│
├── app.py              # Streamlit UI — main entry point
├── agent.py            # AI agent logic, system prompt, memory
├── tools.py            # 5 LangChain tools (RAG, APIs, compliance)
├── ingest.py           # Document ingestion pipeline (RAG)
├── requirements.txt    # Python dependencies
├── .env                # API keys (not pushed to GitHub)
├── docs/               # Sample HR documents
└── chroma_db/          # Vector store (auto-generated, not pushed)
```

---

## Example Questions You Can Ask

- *"How many leave days am I entitled to?"*
- *"What are the public holidays in Tunisia this year?"*
- *"Convert 3000 EUR to TND"*
- *"What is the remote work policy?"*
- *"Who should I contact for payroll questions?"*
- *"Am I compliant if I take 3 sick days without a doctor's note?"*

---

## Notes

- The `venv/` folder and `.env` file are excluded from GitHub for security
- The `chroma_db/` folder is auto-generated when you upload your first document
- Groq free tier has a limit of **100,000 tokens/day** — upgrade at https://console.groq.com/settings/billing

---

## 👩‍💻 Author

**Ferdaous Sarhani**  
HR Intelligence Assistant — Dimploy  
Module Intelligence Artificielle Générative
