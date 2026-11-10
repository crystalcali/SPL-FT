"""Shared Trainer arguments for production and compatibility smoke tests."""


def training_argument_kwargs(c, folder):
    # Transformers 5.20 accepts fractional warmup_steps; warmup_ratio was removed.
    return dict(output_dir=str(folder / "checkpoints"), num_train_epochs=c.epochs,
        max_steps=c.max_steps, per_device_train_batch_size=c.micro_batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=c.accumulation, learning_rate=c.learning_rate, lr_scheduler_type="cosine",
        warmup_steps=0.05, weight_decay=0.01, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1,
        eval_strategy="steps", eval_steps=c.save_steps, save_strategy="steps", save_steps=c.save_steps,
        save_total_limit=2, seed=c.training_seed, data_seed=c.training_seed, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, prediction_loss_only=True,
        include_num_input_tokens_seen=True, optim="adamw_torch")


def build_training_arguments(c, folder, **runtime_overrides):
    from transformers import TrainingArguments
    return TrainingArguments(**{**training_argument_kwargs(c, folder), **runtime_overrides})
