from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    return {
        "question": question,
        "answer": reasoning,
    }
    
def train_model(
    output_dir: str,
    **kwargs,
):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from .data import Dataset
    from .sft import TokenizedDataset

    llm = BaseLLM()

    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
    )
    model = get_peft_model(llm.model, lora_config)
    model.enable_input_require_grads()
    train_dataset = TokenizedDataset(llm.tokenizer, Dataset("rft"), format_example)
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        learning_rate=2e-4,
        save_strategy="no",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
