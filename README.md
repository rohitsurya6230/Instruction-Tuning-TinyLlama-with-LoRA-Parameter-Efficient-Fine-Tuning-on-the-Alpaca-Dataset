# Instruction-Tuning TinyLlama with LoRA: Parameter-Efficient Fine-Tuning on the Alpaca Dataset

## 📌 Project Overview

This project demonstrates **parameter-efficient instruction tuning of TinyLlama using LoRA (Low-Rank Adaptation)** on the **Alpaca instruction-following dataset**.

Instead of updating all parameters of the TinyLlama model, LoRA introduces a small number of trainable parameters into selected transformer layers. This significantly reduces memory usage and training cost while adapting the model to instruction-following tasks.

The complete workflow is implemented in **Google Colab** using the Hugging Face ecosystem.

---

## 🎯 Objectives

- Load and evaluate the pretrained TinyLlama model.
- Prepare the Alpaca instruction-following dataset.
- Convert Alpaca examples into a conversational format.
- Apply **LoRA-based Parameter-Efficient Fine-Tuning (PEFT)**.
- Train TinyLlama using supervised fine-tuning.
- Compare baseline and fine-tuned model responses.
- Evaluate the model on held-out examples.
- Save the trained LoRA adapter.
- Merge the LoRA adapter with the base model.
- Save the final merged model for inference.

---

## 🤖 Model

**Base Model:**

`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`

TinyLlama is a compact language model based on the Llama architecture.

The project uses LoRA to adapt the pretrained model without updating all of its original parameters.

---

## 📚 Dataset

**Dataset:** `tatsu-lab/alpaca`

The Alpaca dataset contains instruction-following examples with fields such as:

- `instruction`
- `input`
- `output`
- `text`

The dataset contains approximately **52K instruction examples**.

The project creates training and evaluation splits before fine-tuning.

---

## 🔄 Project Workflow

```text
                    ┌─────────────────────┐
                    │   TinyLlama Model   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Baseline Evaluation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Alpaca Dataset   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Preparation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Chat Formatting     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     LoRA / PEFT     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Supervised Fine-    │
                    │ Tuning with TRL     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fine-Tuned Model    │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ Evaluation + Adapter Saving     │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Merged Model        │
                    └─────────────────────┘
