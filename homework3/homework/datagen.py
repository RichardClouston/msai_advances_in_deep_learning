import json
import gc
import torch

from .cot import CoTModel
from .data import Dataset, is_answer_valid
from tqdm import tqdm


def generate_dataset(
    output_json: str,
    oversample: int = 25,
    rollouts_per_call: int = 5,
    temperature: float = 0.6,
):
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    model.model.eval()

    training_set = Dataset("train")
    questions = [item[0] for item in training_set]
    correct_answers = [item[1] for item in training_set]
    prompts = [model.format_prompt(q) for q in questions]

    chunk_size = 1
    dataset = []

    if oversample % rollouts_per_call != 0:
        raise ValueError("oversample must be divisible by rollouts_per_call")

    rollout_groups = oversample // rollouts_per_call
    
    for start in tqdm(range(0, len(prompts), chunk_size)):
        chunk_prompts = prompts[start : start + chunk_size]
        chunk_questions = questions[start : start + chunk_size]
        chunk_answers = correct_answers[start : start + chunk_size]

        for prompt, question, correct_answer in zip(
            chunk_prompts, chunk_questions, chunk_answers
        ):
            found_reasoning = None

            for _ in range(rollout_groups):
                completions = model.batched_generate(
                    [prompt],
                    num_return_sequences=rollouts_per_call,
                    temperature=temperature,
                )[0]

                for completion in completions:
                    answer = model.parse_answer(completion)
                    if is_answer_valid(answer, correct_answer):
                        found_reasoning = completion.split("</answer>")[0] + "</answer>"
                        break

                if found_reasoning is not None:
                    break

            if found_reasoning is not None:
                dataset.append([question, correct_answer, found_reasoning])

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
    with open(output_json, "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
