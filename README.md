# Customer Support Chatbot (Python + NLP + Streamlit):
A fully functional AI-powered customer support assistant built using Python, Machine Learning, NLP, and a modern chat UI.
It can handle:
    Order status queries
    Refund policy questions
    Shipping information
    Order cancellation
    Talking to a human
    Greetings, small talk
    FAQ answering using semantic search (TF-IDF + cosine similarity)
    Multi-turn conversations (context memory)
This project is built from scratch without using third-party chatbot APIs — ideal for learning NLP fundamentals, ML pipelines, and UI integration.


# Features
Intent Classification (ML Model)
    The bot uses a logistic regression model trained on a custom dataset of ~120 labeled examples across 8 intents:
        greeting
        goodbye
        order_status
        refund_policy
        shipping_info
        cancel_order
        human_agent
        small_talk


# Semantic FAQ Search (TF-IDF + Cosine Similarity)
If the intent classifier is unsure, the bot performs semantic search:
    Vectorizes all FAQ questions
    Computes cosine similarity with user message
    Returns best-matching answer if similarity > threshold

This allows the bot to answer:
    “Do you accept PayPal?”
    “How do I contact support?”
    “Is COD available?”
even if these questions weren't part of the training data.


# Fake Order Database (data/orders.csv)
    The chatbot uses a synthetic order database to mimic real e-commerce behavior.
        - Stored in `data/orders.csv`
        - 200–700 randomly generated orders
        - Fields include:
        - `order_id`
        - `status` (Processing, Shipped, Delivered, Cancelled)
        - `eta` (estimated delivery date)
        - `total` (amount)
        - `shipping_provider`
    This allows the bot to provide realistic responses like:
        - “Your order 123456 is Shipped via DHL and will arrive on 2025-12-09.”
        - “Order 555555 is Delivered and cannot be cancelled.”
        - “Order ID not found” if it doesn’t exist.
Order logic is handled in `chatbot/handlers.py`.


# Multi-turn Conversation Memory
The bot remembers the user’s last intent to allow natural conversations:
Example:
    You: where is my order?
    Bot: Please provide your order ID.
    You: 12345
    Bot: Order 12345 is currently being processed.
This is implemented using Streamlit session state.


# Modern Streamlit Chat UI
The chatbot interface includes:
    Chat bubbles with avatars (user + bot)
    Typing simulation (optional upgrade)
    Sidebar for branding and project information
    Clean, responsive layout
    Message timestamps for both user and bot messages
    “Clear conversation” button to reset chat history and memory


# Built-In Analytics Dashboard
    The project includes an analytics dashboard accessible via the sidebar.
    It provides:
        - Total number of messages
        - Number of unique intents used
        - Last activity timestamp
        - Bar chart: Messages per intent
        - Table of recent interactions
        - Expandable full log viewer

    All interactions are logged to `logs/interactions.csv` in the format:
        - timestamp
        - intent
        - user_text
        - bot_reply


# Modular Architecture — Easy to Extend
customer-support-bot/
├── app.py                 # Streamlit UI (chat + analytics)
├── api.py                 # FastAPI backend
├── train_intent_model.py  # ML training pipeline
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── chatbot/
│   ├── nlp.py             # Intent model loading + prediction
│   ├── handlers.py        # Order logic, cancellation logic
│   ├── config.py          # Predefined responses
│   ├── faq.py             # Semantic FAQ engine
│   └── __init__.py
├── data/
│   ├── intents.csv        # Intent training data
│   ├── faq.csv            # FAQ dataset
│   └── orders.csv         # Fake order "database"
├── models/
│   ├── intent_classifier.pkl
│   └── vectorizer.pkl
└── logs/
    └── interactions.csv   # Auto-generated conversation logs


# FastAPI Backend (Optional API Layer)
    The chatbot logic is also exposed via a FastAPI backend (`api.py`), allowing this model to be used by:
        - Web apps
        - Mobile apps
        - Other microservices

    Endpoints:
        - `GET /health` → health check  
        - `POST /chat` → process a user message and return:
            - detected intent
            - chatbot reply
            - next expected intent (for multi-turn flows)

The Streamlit UI can be connected directly to this API for a true frontend–backend separation.


# AI & NLP Under the Hood
1. Intent Classification (Supervised ML)
The core brain of the bot uses:
    Logistic Regression (scikit-learn)
    TF-IDF vectorization (unigrams + bigrams)
    Custom training dataset (intents.csv)
    Saved model + vectorizer in models/

Training pipeline includes:
    Load and preprocess data
    Vectorize text using TF-IDF
    Train the classifier
    Save model and vectorizer for inference

2. Semantic FAQ Fallback (NLP + IR Hybrid)
When the model predicts fallback, the bot runs:
    TF-IDF on FAQ questions
    Cosine similarity with user query
    If score ≥ 0.25 → return closest FAQ answer
    Otherwise → return a guided fallback response
This allows human-like flexibility.

3. Conversation Memory
Using Streamlit’s session_state, we maintain:
    last_intent
    Pending actions (e.g., expecting an order ID)
    This enables multi-turn flows.

4. Deterministic + ML Hybrid Design
The system combines:
    Component	            Role
    ML intent classifier	Detects user intent
    FAQ semantic engine	    Handles open-ended questions
    Rule-based handlers	    Ensures reliable order/cancel logic
    Streamlit UI	        User-friendly chat interface
This hybrid architecture offers accuracy + flexibility + stability.


# Installation & Setup
    Clone the repo
    git clone https://github.com/zayanahmed23/customer-support-chatbot
    cd customer-support-bot

    Create a virtual environment
        python -m venv .venv
        Activate it:
        Windows:
        .venv\Scripts\activate

    Install dependencies
        pip install -r requirements.txt

    Train the ML model
        python train_intent_model.py
        
    This will create:
        models/
        ├── intent_classifier.pkl
        └── vectorizer.pkl

    Run the chatbot UI
        streamlit run app.py

    Visit:
    http://localhost:8501


# Contact / Author
Author: Zayan Ahmed
Email: zayanahmed222@gmail.com
GitHub: https://github.com/zayanahmed23
LinkedIn: https://www.linkedin.com/in/zayan-ahmed-1623b71b4/


If you like this project
Please give it a ⭐ on GitHub — it helps others discover it!