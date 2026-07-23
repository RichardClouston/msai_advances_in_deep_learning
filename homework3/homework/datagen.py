import json
import gc
import torch

from .cot import CoTModel
from .data import Dataset, is_answer_valid
from tqdm import tqdm


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    training_set = Dataset("train")
    questions = [item[0] for item in training_set]
    correct_answers = [item[1] for item in training_set]
    prompts = [model.format_prompt(q) for q in questions]
    chunk_size = 1
    dataset = []
    
    for start in tqdm(range(0, len(prompts), chunk_size)):
        chunk_prompts = prompts[start : start + chunk_size]
        chunk_questions = questions[start : start + chunk_size]
        chunk_answers = correct_answers[start : start + chunk_size]

        completions_batch = model.batched_generate(
            chunk_prompts, num_return_sequences=oversample, temperature=temperature
        )
        for question, correct_answer, completions in zip(chunk_questions, chunk_answers, completions_batch):
            for completion in completions:
                answer = model.parse_answer(completion)
                if is_answer_valid(answer, correct_answer):
                    reasoning = completion.split("</answer>")[0] + "</answer>"
                    dataset.append([question, correct_answer, reasoning])
                    break
        torch.mps.empty_cache()
        gc.collect()
    with open(output_json, "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
