import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto"  # let deepspeed decides the device
    )
    return tokenizer, model


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--train_file", type=str, required=True)
    args.add_argument("--val_file", type=str, required=True)
    args.add_argument("--save_dir", type=str, required=True)
    args.add_argument("--deepspeed", type=str, help="Path to deepseed config.json")

    args.add_argument("--task", type=str, choices=['drugrec', 'diagnosis'], required=True)

    args.add_argument("-model", "--model_name_or_path", type=str, help="Large Language Model", required=True)

    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--steps", type=int, default=200, help="Global step for evaluation and checkpointing.")

    args.add_argument("-bs", "--batch_size", type=int, default=1, help="Per GPU batch size")
    args.add_argument("-grad_accu", "--gradient_accumulation_steps", type=int, default=4)
    args.add_argument("-lr", "--learning_rate", type=float, default=5e-06)
    config = args.parse_args()

    model_name = config.model_name_or_path.rstrip("/").split("/")[-1]


    # Use `Prompt-completion` template of trl, which will automatically mask loss of `prompt` tokens
    # chat template will be automatically added
    def process_sample(example):
        return {"prompt": [{"role": "user", "content": example['input']}],
                "completion": [{"role": "assistant", "content": example['output']}]}


    tokenizer, model = load_model()
    datasets = load_dataset("json", data_files={"train": config.train_file, "val": config.val_file})
    datasets = datasets.map(process_sample)
    train_dataset, val_dataset = datasets['train'], datasets['val']

    training_args = SFTConfig(
        output_dir=config.save_dir,
        overwrite_output_dir=True,
        report_to="none",

        do_train=True,
        do_eval=True,

        max_length=8192,
        # packing=True,
        assistant_only_loss=True,

        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.1,

        logging_strategy="steps",
        logging_steps=5,
        logging_dir=config.save_dir,

        eval_strategy="steps",
        eval_steps=config.steps,

        save_strategy="steps",
        save_steps=config.steps,
        save_total_limit=10,
        save_only_model=True,

        bf16=True,
        remove_unused_columns=True,

        gradient_checkpointing=True,
        deepspeed=config.deepspeed
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
