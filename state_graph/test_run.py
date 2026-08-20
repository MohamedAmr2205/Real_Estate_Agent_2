from graph_1_offer_negotiation import graph_1
from checkpoint import get_thread_config

def main():
    config = get_thread_config("thread_test_001")
    
    # Initial negotiation state with > 15% discount to trigger HITL
    initial_input = {
        "property_id": "PROP-999",
        "offered_price": 700000.0,
        "original_price": 1000000.0
    }

    print("--- Starting Negotiation Graph ---")
    for event in graph_1.stream(initial_input, config):
        print("Event:", event)

if __name__ == "__main__":
    main()