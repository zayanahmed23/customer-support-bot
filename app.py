from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from chatbot.nlp import predict_intent
from chatbot.handlers import handle_intent
from chatbot.config import INTENT_RESPONSES

# -----------------------
# Paths & logging helpers
# -----------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "interactions.csv"


def log_interaction(user_text: str, intent: str, reply: str) -> None:
    """
    Append a single interaction (user message + intent + bot reply)
    to logs/interactions.csv
    """
    now = datetime.now().isoformat(timespec="seconds")
    df = pd.DataFrame(
        [
            {
                "timestamp": now,
                "intent": intent,
                "user_text": user_text,
                "bot_reply": reply,
            }
        ]
    )
    file_exists = LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", index=False, header=not file_exists)


# -----------------------
# Streamlit page settings
# -----------------------

st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="wide",
)

# -------- Global custom CSS --------
st.markdown(
    """
    <style>

    /* Center main content and control width */
    .block-container {
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* Card wrapper for the chat area */
    .chat-card {
        background-color: #FFFFFF;
        padding: 1.5rem 1.75rem;
        border-radius: 1rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Chat message text improvement */
    .stChatMessage p {
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
    }

    /* Pill-style chat input */
    div[data-baseweb="textarea"] textarea {
        border-radius: 999px !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
    }

    /* Sidebar font size */
    section[data-testid="stSidebar"] {
        font-size: 0.9rem !important;
    }

    hr {
        margin: 1.5rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------

st.sidebar.title("📦 Customer Support Bot")
mode = st.sidebar.radio("Mode", ["Chat", "Analytics"])

st.sidebar.markdown(
    """
**About this app**

- 🤖 NLP-powered customer support assistant  
- 🧠 Intent classification (Logistic Regression + TF-IDF)  
- 🧮 Semantic FAQ fallback (cosine similarity)  
- 🔄 Multi-turn order / cancellation flows  
- 📊 Built-in analytics dashboard  
"""
)

# ------------- Session state -------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I’m your customer support assistant 🤖\n\n"
                "You can ask me about:\n"
                "- Order status\n"
                "- Refunds\n"
                "- Shipping\n"
                "- Order cancellation\n"
                "- Talking to a human"
            ),
        }
    ]

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None


# ------------------- CHAT MODE -------------------

if mode == "Chat":
    st.title("Customer Support Bot 🤖")
    st.button(
    "🧹 Clear conversation",
    on_click=lambda: (
        st.session_state.pop("messages", None),
        st.session_state.pop("last_intent", None)
    )
    )
    st.caption(
        "Ask me anything about your orders, refunds, shipping, or cancellations."
    )

    # ----- Chat card wrapper -----
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)

    # Quick suggestion buttons
    st.markdown("**Quick questions (click to try):**")
    col1, col2, col3, col4 = st.columns(4)
    preset_message = None

    with col1:
        if st.button("📦 Track my order"):
            preset_message = "where is my order"
    with col2:
        if st.button("💰 Refund policy"):
            preset_message = "what is your refund policy"
    with col3:
        if st.button("🚚 Shipping info"):
            preset_message = "do you ship internationally"
    with col4:
        if st.button("🧑 Talk to human"):
            preset_message = "i want to talk to a human"

    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)


    # Show chat history
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):

            timestamp = msg.get("time", "")  # fallback for older messages

            # timestamp display
            st.markdown(
                f"<div style='text-align: right; margin-bottom: -8px;'>"
                f"<span style='font-size: 11px; color: #9CA3AF;'>{timestamp}</span>"
                "</div>",
                unsafe_allow_html=True
            )

            # message text
            st.markdown(msg["content"])


    # Chat input
    user_input = st.chat_input("Type your message here...")

    # Decide which input to process: preset button or manual typing
    message_to_process = None
    if preset_message is not None:
        message_to_process = preset_message
    elif user_input:
        message_to_process = user_input

    if message_to_process:
        from datetime import datetime
        st.session_state.messages.append(
            {
                "role": "user",
                "content": message_to_process,
                "time": datetime.now().strftime("%I:%M %p")
            }
        )
        with st.chat_message("user", avatar="🧑"):
            st.markdown(message_to_process)

        only_digits = message_to_process.strip().isdigit()
        last_intent = st.session_state.get("last_intent")

        # 2) Decide which intent to use (support multi-turn order ID flow)
        if only_digits and last_intent in {"order_status", "cancel_order"}:
            intent = last_intent
        else:
            intent = predict_intent(message_to_process)

        # 3) Get bot reply
        reply = handle_intent(intent, message_to_process)

        # 4) Update simple memory for order/cancel flows
        if intent in {"order_status", "cancel_order"}:
            st.session_state.last_intent = intent
        else:
            st.session_state.last_intent = None

        # 5) Show bot reply
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "time": datetime.now().strftime("%I:%M %p")
            }
        )

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(reply)

        # 6) Log interaction for analytics
        log_interaction(message_to_process, intent, reply)

    # Close chat card wrapper
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------- ANALYTICS MODE -------------------

else:
    st.title("📊 Analytics Dashboard")
    st.caption("High-level overview of how users interact with the chatbot.")

    if not LOG_FILE.exists():
        st.info("No interaction data yet. Use the Chat mode first.")
    else:
        df = pd.read_csv(LOG_FILE)

        # ---- Metrics row ----
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", len(df))
        with col2:
            st.metric("Unique Intents", df["intent"].nunique())
        with col3:
            st.metric("Last Activity", df["timestamp"].max())

        st.markdown("---")

        # ---- Messages per intent ----
        st.subheader("Messages by Intent")
        intent_counts = df["intent"].value_counts().sort_values(ascending=False)
        st.bar_chart(intent_counts)

        # ---- Recent interactions ----
        st.subheader("Recent Interactions")
        df_sorted = df.sort_values("timestamp", ascending=False)
        st.dataframe(df_sorted.head(20), use_container_width=True)

        # ---- Raw data expander ----
        with st.expander("View full interaction log"):
            st.dataframe(df, use_container_width=True)
