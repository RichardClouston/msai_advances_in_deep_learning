import math

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def main():
    model = CoTModel()
    dataset = Dataset("valid")

    questions = [dataset[i][0] for i in range(100)]
    expected_answers = [dataset[i][1] for i in range(100)]
    prompts = [model.format_prompt(question) for question in questions]
    generations = model.batched_generate(prompts)

    shown = 0
    for question, expected, generation in zip(
        questions, expected_answers, generations
    ):
        parsed = model.parse_answer(generation)
        correct = is_answer_valid(parsed, expected)

        if not correct:
            reason = "missing/unparseable answer tag" if math.isnan(parsed) else "wrong value"

            print("\n" + "=" * 80)
            print("Reason:  ", reason)
            print("Question:", question)
            print("Expected:", expected)
            print("Generated:", repr(generation))
            print("Parsed:  ", parsed)

            shown += 1
            if shown == 20:
                break


if __name__ == "__main__":
    main()