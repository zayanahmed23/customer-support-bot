from .nlp import predict_intent
from .handlers import handle_intent
from .config import INTENT_RESPONSES


def run_chatbot():
    print("Customer Support Bot 🤖")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Bot:", INTENT_RESPONSES["goodbye"])
            break

        intent = predict_intent(user_input)
        response = handle_intent(intent, user_input)
        print("Bot:", response)


if __name__ == "__main__":
    run_chatbot()
