# main.py
# Entry point for the swap-moe pipeline.

from swap_manager import SwapManager


def main():
    manager = SwapManager()
    print("\nStatus:", manager.get_status())

    test_inputs = [
        "Wie viel Vitamin C brauche ich täglich?",
        "Ich forsche über Mangelkrankheiten, was ist der Zusammenhang?",
        "Mein Python Script gibt einen TypeError aus",
        "Was kann ich mit Hähnchenbrust kochen?",
    ]

    for user_input in test_inputs:
        answer = manager.process(user_input)
        print(f"\nInput:  {user_input}")
        print(f"Answer: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
