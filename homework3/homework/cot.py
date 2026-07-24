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
                    "You are a unit-conversion assistant. Use the correct conversion "
                    "factor, calculate carefully, and be concise. Give one short "
                    "equation followed by exactly one final numeric answer in "
                    "<answer></answer> tags. Do not omit the tags and do not write "
                    "anything after </answer>."
                ),
            },
            {
                "role": "user",
                "content": "How many gram are there per 3 kg?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 kg = 1000 grams. 3 * 1000 = <answer>3000</answer>"
                ),
            },
            {
                "role": "user",
                "content": "How much is 3 mi/h when converted to m/s?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 mi/h = 0.44704 m/s. 3 * 0.44704 = "
                    "<answer>1.34112</answer>"
                ),
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

    test_set = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, test_set, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
