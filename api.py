from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from chatbot.nlp import predict_intent
from chatbot.handlers import handle_intent

app = FastAPI(
    title="Customer Support Chatbot API",
    description="FastAPI backend for intent classification and chatbot replies.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str
    last_intent: Optional[str] = None  # for multi-turn flows (optional)


class ChatResponse(BaseModel):
    intent: str
    reply: str
    next_intent: Optional[str] = None  # frontend can store this for context


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Main chatbot endpoint.

    - Takes user message (and optional last_intent)
    - Predicts intent
    - Returns reply + next_intent for the client to store
    """
    user_text = payload.message.strip()
    last_intent = payload.last_intent

    # Simple multi-turn handling: if user sends only digits and last_intent was order-related
    if user_text.isdigit() and last_intent in {"order_status", "cancel_order"}:
        intent = last_intent
    else:
        intent = predict_intent(user_text)

    reply = handle_intent(intent, user_text)

    # Decide what next_intent the client should remember
    if intent in {"order_status", "cancel_order"}:
        next_intent = intent
    else:
        next_intent = None

    return ChatResponse(intent=intent, reply=reply, next_intent=next_intent)
