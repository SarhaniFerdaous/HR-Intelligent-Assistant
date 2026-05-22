import os
import requests
import streamlit as st
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime

CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# CACHED EMBEDDINGS — loaded once per process (#6 fix)
# ─────────────────────────────────────────────

@st.cache_resource
def _get_embeddings():
    """Load the embedding model once and reuse across all tool calls."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_retriever():
    """Return a retriever using the cached embedding model."""
    embeddings  = _get_embeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 6})


# ─────────────────────────────────────────────
# SOURCE 1 — RAG
# ─────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """Search all uploaded HR documents — PDFs, DOCX, TXT, CSV —
    to answer questions about policies, onboarding, compensation,
    benefits, leave, and company rules. Always try this tool first."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in uploaded documents."
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        results.append(f"[Document {i} — {source}]\n{doc.page_content}")
    return "\n\n".join(results)


# ─────────────────────────────────────────────
# SOURCE 2 — EXTERNAL APIS
# ─────────────────────────────────────────────

@tool
def get_hr_contact(topic: str) -> str:
    """Get the HR contact or department email for a specific topic.
    Use when the employee needs to be directed to the right person."""
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/users", timeout=5)
        users    = response.json()
        contacts = {
            "leave":      ("hr.leave@acme.com",     users[0]["phone"]),
            "payroll":    ("hr.payroll@acme.com",    users[1]["phone"]),
            "onboarding": ("hr.onboarding@acme.com", users[2]["phone"]),
            "training":   ("hr.training@acme.com",   users[3]["phone"]),
            "benefits":   ("hr.benefits@acme.com",   users[4]["phone"]),
        }
        for key, (email, phone) in contacts.items():
            if key in topic.lower():
                return f"Contact: {email} | Phone: {phone}"
        return f"General HR: hr@acme.com | Phone: {users[0]['phone']}"
    except Exception as e:
        return f"HR directory unavailable. Email hr@acme.com directly. Error: {e}"


@tool
def get_public_holidays(country_code: str) -> str:
    """Get public holidays for a given country to calculate accurate
    leave entitlements. Use ISO country code e.g. TN, FR, US, GB."""
    try:
        year     = datetime.now().year
        url      = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code.upper()}"
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return f"Could not fetch holidays for country code: {country_code}."
        holidays = response.json()
        if not holidays:
            return f"No holidays found for {country_code.upper()}."
        lines = [f"  - {h['date']}: {h['localName']}" for h in holidays[:12]]
        return (
            f"Public holidays in {country_code.upper()} ({year}) "
            f"— {len(holidays)} total:\n" + "\n".join(lines)
        )
    except Exception as e:
        return f"Holiday API error: {e}"


@tool
def convert_currency(amount_and_currencies: str) -> str:
    """Convert a salary or allowance amount between currencies.
    Input format: '5000 USD to TND' or '3000 EUR to USD'."""
    try:
        parts    = amount_and_currencies.lower().replace(" to ", " ").split()
        amount   = float(parts[0])
        from_cur = parts[1].upper()
        to_cur   = parts[2].upper()
        url      = f"https://open.er-api.com/v6/latest/{from_cur}"
        response = requests.get(url, timeout=5)
        data     = response.json()
        if "rates" not in data:
            return f"Could not retrieve exchange rates for {from_cur}."
        rate = data["rates"].get(to_cur)
        if not rate:
            return f"Currency {to_cur} not found in exchange rates."
        converted = round(amount * rate, 2)
        return (
            f"{amount:,.2f} {from_cur} = {converted:,.2f} {to_cur}\n"
            f"Live rate: 1 {from_cur} = {rate} {to_cur}"
        )
    except Exception as e:
        return f"Currency conversion error: {e}"


# ─────────────────────────────────────────────
# SOURCE 3 — COMPLIANCE CHECK
# ─────────────────────────────────────────────

@tool
def check_compliance(situation: str) -> str:
    """Evaluate whether a described employee situation complies with
    company policy. Use when an employee describes a specific scenario."""
    rules = {
        "sick":      "Sick leave: 10 days paid/year. Doctor note required after 3 consecutive days.",
        "overtime":  "Overtime: 1.5x rate weekdays, 2x on weekends.",
        "remote":    "Remote work: up to 2 days/week after probation.",
        "leave":     "Annual leave: 25 days/year. Max 5 carry-over days.",
        "bonus":     "Bonus: 10-15% of salary for Exceeds Expectations rating.",
        "probation": "Probation: 3 months. Review at month 2.",
        "dress":     "Dress code: business casual Mon-Thu, casual Friday, formal for clients.",
        "maternity": "Maternity: 16 weeks fully paid. Notify HR 8 weeks in advance.",
        "paternity": "Paternity: 2 weeks fully paid within 3 months of birth.",
        "training":  "Training budget: 1200 TND/year per employee.",
        "transport": "Transport allowance: 150 TND/month for commutes over 20km.",
        "meal":      "Meal vouchers: 8 TND/day per working day.",
        "insurance": "Health insurance: full medical + dental from day 1 for employee and family.",
    }
    matches = [msg for kw, msg in rules.items() if kw in situation.lower()]
    rag_hint = search_documents.invoke(situation)
    if matches:
        return (
            "Policy compliance check:\n"
            + "\n".join(matches)
            + "\n\nFrom your documents:\n"
            + rag_hint
        )
    return "No direct rule matched. Searching your uploaded documents...\n\n" + rag_hint