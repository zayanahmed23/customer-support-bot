import re
from pathlib import Path

import pandas as pd

from .config import INTENT_RESPONSES
from .faq import semantic_faq_search

# -------------------------
# Order ID detection
# -------------------------

# Simple pattern: order IDs are 5+ digit numbers
ORDER_ID_PATTERN = r"\b\d{5,}\b"


def extract_order_id(text: str):
    """
    Extract a numeric order ID from the user's text, if present.
    """
    match = re.search(ORDER_ID_PATTERN, text)
    if match:
        return match.group(0)
    return None


# -------------------------
# Fake order "database"
# -------------------------

ORDERS_PATH = Path("data/orders.csv")

if ORDERS_PATH.exists():
    _orders_df = pd.read_csv(ORDERS_PATH)

    # Normalize to strings to avoid type issues
    _orders_df["order_id"] = _orders_df["order_id"].astype(str)
    _orders_df["status"] = _orders_df["status"].astype(str)
    _orders_df["eta"] = _orders_df["eta"].astype(str)
    _orders_df["total"] = _orders_df["total"].astype(str)
    _orders_df["shipping_provider"] = _orders_df["shipping_provider"].astype(str)

    # Index by order_id for fast lookup
    _orders_df = _orders_df.set_index("order_id")
else:
    # If the file is missing, use an empty DataFrame to avoid crashes
    _orders_df = pd.DataFrame(
        columns=["status", "eta", "total", "shipping_provider"]
    )


def get_order_info(order_id: str):
    """
    Look up order info in the fake 'database'.

    Returns a dict like:
    {
        "status": "...",
        "eta": "...",
        "total": "...",
        "shipping_provider": "..."
    }
    or None if not found.
    """
    if order_id in _orders_df.index:
        row = _orders_df.loc[order_id]
        return {
            "status": row["status"],
            "eta": row["eta"],
            "total": row["total"],
            "shipping_provider": row["shipping_provider"],
        }
    return None


# -------------------------
# Intent handling
# -------------------------


def handle_intent(intent: str, user_text: str) -> str:
    """
    Given an intent and the original user text, decide what to reply.
    """

    # -------- Order status flow --------
    if intent == "order_status":
        order_id = extract_order_id(user_text)

        if order_id:
            info = get_order_info(order_id)
            if info:
                return (
                    f"Here’s what I found for order **{order_id}**:\n\n"
                    f"- **Status:** {info['status']}\n"
                    f"- **Estimated delivery:** {info['eta']}\n"
                    f"- **Total amount:** {info['total']}\n"
                    f"- **Shipping provider:** {info['shipping_provider']}\n\n"
                    "If you have more questions about this order, you can ask me!"
                )
            else:
                return (
                    f"I couldn’t find any details for order **{order_id}**. 🧐\n"
                    "Please double-check the order ID or contact support if you think this is a mistake."
                )
        else:
            return "Please provide your order ID (e.g., a 6–8 digit number)."

    # -------- Cancellation flow --------
    if intent == "cancel_order":
        order_id = extract_order_id(user_text)

        if order_id:
            info = get_order_info(order_id)
            if not info:
                return (
                    f"I couldn’t find any details for order **{order_id}**. 🧐\n"
                    "Please double-check the order ID or contact support for help with cancellation."
                )

            status = info["status"]

            if status.lower() == "processing":
                return (
                    f"Order **{order_id}** is currently in *Processing* and has been marked for cancellation. ✅\n"
                    "You’ll receive a confirmation email shortly, and any payment will be refunded according to our refund policy."
                )
            elif status.lower() == "shipped":
                return (
                    f"Order **{order_id}** has already been *Shipped* 📦\n"
                    "We can no longer cancel it at this stage. You may refuse delivery or request a return once it arrives."
                )
            elif status.lower() == "delivered":
                return (
                    f"Order **{order_id}** has already been *Delivered* 📬\n"
                    "We cannot cancel a delivered order, but you may be able to request a return or refund depending on our return policy."
                )
            elif status.lower() == "cancelled":
                return (
                    f"Order **{order_id}** is already marked as *Cancelled*. ✅\n"
                    "If you didn’t request this, please contact support."
                )
            else:
                # Unknown status label
                return (
                    f"Order **{order_id}** has status **{status}**.\n"
                    "Please contact support if you’d like to change or cancel this order."
                )

        else:
            return "To cancel an order, please tell me your order ID (e.g., a 6–8 digit number)."

    # -------- Fallback: semantic FAQ search --------
    if intent == "fallback":
        faq_answer = semantic_faq_search(user_text)
        if faq_answer:
            return faq_answer
        return INTENT_RESPONSES["fallback"]

    # -------- All other intents use pre-defined responses --------
    return INTENT_RESPONSES.get(intent, INTENT_RESPONSES["fallback"])
