from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that converts units. "
                    "State the equivalence between the units, then apply it. "
                    "Be concise: one short sentence of reasoning, then the final "
                    "numeric result wrapped in <answer></answer> tags. "
                    "Do not add anything after the closing tag."
                    ),
            },
            {
                "role": "user",
                "content": "How many minutes are in 3 hours?",
            },
            {
                "role": "assistant",
                "content": "There are 60 minutes in 1 hour. 3 hours * 60 = <answer>180</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
