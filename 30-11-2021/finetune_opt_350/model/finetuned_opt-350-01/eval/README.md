---
tags:
- generated_from_trainer
model-index:
- name: eval
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# eval

This model is a fine-tuned version of [model/pre_trained_model_opt-350M/](https://huggingface.co/model/pre_trained_model_opt-350M/) on an unknown dataset.
It achieves the following results on the evaluation set:
- eval_loss: 2.6740
- eval_accuracy: 0.4972
- eval_runtime: 8.3918
- eval_samples_per_second: 0.357
- eval_steps_per_second: 0.119
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.21.0.dev0
- Pytorch 1.12.0
- Datasets 2.3.2
- Tokenizers 0.11.0
