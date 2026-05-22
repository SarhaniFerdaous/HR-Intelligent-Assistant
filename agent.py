import os
import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma

from tools import (
    search_documents,
    check_compliance,
    get_hr_contact,
    get_public_holidays,
    convert_currency,
    _get_embeddings,
)

load_dotenv()

CHROMA_DIR    = "chroma_db"
FEEDBACK_FILE = "feedback.json"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=3000,
)

tools = [
    search_documents,
    check_compliance,
    get_hr_contact,
    get_public_holidays,
    convert_currency,
]

def _system_prompt() -> str:
    return """You are Dimploy, a friendly and professional HR assistant at Acme Corp. You are warm, conversational, and helpful — like a real colleague in the HR department.

PERSONALITY:
- Greet users back warmly when they say hello, hi, or similar
- Use a natural, human tone — not robotic or overly formal
- Show empathy when employees describe difficult situations (e.g. sick leave, termination)
- Use light encouragement ("Great question!", "Happy to help with that!")
- Keep small talk short and redirect naturally to how you can help
- Sign off warmly when the conversation ends
- If asked something outside HR scope, politely decline and redirect

EXAMPLES OF HUMAN-LIKE RESPONSES:
- User: "Hello" → "Hey there! 👋 I'm Dimploy, your HR assistant. How can I help you today?"
- User: "Thanks" → "You're welcome! Don't hesitate to reach out if you need anything else. 😊"
- User: "I'm stressed about my leave balance" → "I'm sorry to hear that — let me look that up for you right away!"
- User: "Who is [some person]?" → "I'm only able to help with HR-related topics like leave, payroll, compliance, and benefits. Is there anything HR-related I can assist you with? 😊"
- User: "What is the capital of France?" → "That's a bit outside my area! I'm your HR assistant — ask me about leave, payroll, onboarding, or company policy instead. 😊"

RULES:
1. For holiday/calendar questions → call get_public_holidays FIRST (use country code e.g. TN for Tunisia)
2. For currency/salary conversion → call convert_currency FIRST
3. For HR contact questions → call get_hr_contact FIRST
4. For policy/HR document questions → call search_documents FIRST
5. For compliance checks → call search_documents THEN check_compliance
6. NEVER say "I could not find this" if a relevant API tool exists for the question
7. Always cite your source (document name or API used)
8. When get_public_holidays returns data, display the FULL list — never summarize or say "there are X holidays". List every single one with its date and name.
9. Reply in the SAME LANGUAGE the employee used
10. For greetings, small talk, or thank-yous — respond naturally WITHOUT calling any tool
11. If the question is NOT related to HR, work, company policy, leave, payroll, benefits, compliance, or onboarding → respond politely WITHOUT calling any tool, and redirect the user to HR topics
12. NEVER attempt a tool call for general knowledge, personal questions, celebrities, or anything unrelated to HR

TOOL SELECTION GUIDE:
- "holidays", "public holiday", "day off", "calendar" → get_public_holidays, then list ALL results
- "convert", "currency", "salary in", "USD", "TND", "EUR" → convert_currency
- "contact", "email", "who do I call" → get_hr_contact
- "policy", "rule", "allowed", "entitled", "how many days" → search_documents
- "am I compliant", "is it ok if I", "can I" → search_documents + check_compliance
- "hello", "hi", "hey", "thanks", "bye", "how are you" → respond naturally, NO tool call needed
- anything unrelated to HR (celebrities, general knowledge, random people, random questions) → NO tool call, politely redirect

If no tool returns useful info, say warmly:
"Hmm, I wasn't able to find that in our documents. I'd recommend reaching out to hr@company.com — they'll be able to help you directly! 😊"
"""

def build_agent_executor() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _system_prompt()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    max_iterations=6,           # allow multi-tool chains
    early_stopping_method="generate",  # forces a final answer instead of stopping cold
)

def get_loaded_documents() -> str:
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_get_embeddings()
        )
        collection = vectorstore.get()
        sources = {
            os.path.basename(meta["source"])
            for meta in collection.get("metadatas", [])
            if meta and "source" in meta
        }
        return ", ".join(sorted(sources)) if sources else "No documents currently loaded."
    except Exception:
        return "Unable to retrieve document list."

def get_confidence(steps: list) -> dict:
    if not steps:
        return {"level": "Verified", "color": "#1D9E75"}
    for action, observation in steps:
        obs = str(observation)
        if "search" in action.tool:
            if "No relevant" in obs or len(obs) < 80:
                return {"level": "Unverified", "color": "#BA7517"}
            return {"level": "Verified", "color": "#1D9E75"}
    return {"level": "Live data", "color": "#534AB7"}

def classify_ticket(user_input: str, answer: str) -> str:
    text = (user_input + " " + answer).lower()
    cats = {
        "Leave":      ["leave","vacation","holiday","sick","maternity","congé","إجازة"],
        "Payroll":    ["salary","payroll","pay","bonus","currency","convert","salaire","راتب"],
        "Onboarding": ["onboarding","first day","new hire","probation","intégration","تأهيل"],
        "Compliance": ["compliance","policy","law","regulation","conformité","امتثال"],
        "Training":   ["training","course","certificate","formation","تدريب"],
        "Benefits":   ["insurance","health","dental","benefit","voucher","assurance","تأمين"],
        "Urgent":     ["urgent","asap","emergency","fired","terminate","urgence","عاجل"],
    }
    for label, keywords in cats.items():
        if any(kw in text for kw in keywords):
            return label
    return "General"

def save_feedback(question: str, answer: str, rating: str) -> None:
    feedback = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            try:
                feedback = json.load(f)
            except Exception:
                feedback = []
    feedback.append({
        "timestamp": datetime.now().isoformat(),
        "question":  question,
        "answer":    answer[:300],
        "rating":    rating,
    })
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)

def load_feedback_stats() -> dict:
    if not os.path.exists(FEEDBACK_FILE):
        return {"helpful": 0, "not_helpful": 0, "total": 0}
    with open(FEEDBACK_FILE, "r") as f:
        try:
            feedback = json.load(f)
        except Exception:
            return {"helpful": 0, "not_helpful": 0, "total": 0}
    helpful = sum(1 for fb in feedback if fb["rating"] == "helpful")
    return {"helpful": helpful, "not_helpful": len(feedback) - helpful, "total": len(feedback)}

def suggest_questions_for_doc(filename: str) -> list:
    name = filename.lower()
    if "leave" in name or "congé" in name:
        return ["How many leave days do I get?", "Can I carry over unused leave?", "What is the sick leave policy?"]
    if "onboard" in name or "integration" in name:
        return ["What do I need on my first day?", "How long is the probation period?", "Who is my onboarding contact?"]
    if "payroll" in name or "salary" in name or "salaire" in name:
        return ["When is payday?", "How is my bonus calculated?", "What deductions appear on my payslip?"]
    return ["What are the main policies in this document?", "What are my key rights?", "Who should I contact for questions?"]

def run_agent(user_input: str, chat_history: list) -> dict:
    try:
        executor = build_agent_executor()
        today    = datetime.now().strftime("%A, %B %d, %Y")
        docs     = get_loaded_documents()
        enriched = f"[Today: {today} | Docs available: {docs}]\n{user_input}"

        result = executor.invoke({
            "input":        enriched,
            "chat_history": chat_history,
        })

        output = result["output"]
        steps  = result.get("intermediate_steps", [])

        return {
            "output": output,
            "steps":  steps,
            "conf":   get_confidence(steps),
            "ticket": classify_ticket(user_input, output),
            "error":  None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()          # prints real error to terminal
        return {
            "output": f"Error: {str(e)}",   # shows real error in UI too
            "steps":  [],
            "conf":   {"level": "Unverified", "color": "#BA7517"},
            "ticket": "General",
            "error":  str(e),
        }

def build_history(messages: list) -> list:
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history