Parameter-Efficient Fine-Tuning of TinyLlama using LoRA for Instruction Following

A complete instruction-tuning project using TinyLlama 1.1B, the Alpaca dataset, and LoRA (Low-Rank Adaptation).

The project is designed as a clean Google Colab workflow that loads a pretrained language model, evaluates its baseline behavior, fine-tunes it with LoRA on Alpaca-style instructions, evaluates the adapted model, saves the LoRA adapter, and optionally merges the adapter with the base model.

1. Project Overview

Project Title

Parameter-Efficient Fine-Tuning of TinyLlama using LoRA for Instruction Following

Short Project Name

TinyLlama-LoRA-Alpaca Instruction Tuning

Objective

The objective of this project is to adapt a pretrained TinyLlama 1.1B language model so that it follows natural-language instructions more effectively.

Instead of updating every parameter of the language model, the project uses LoRA, a parameter-efficient fine-tuning method. LoRA adds a small number of trainable parameters while keeping the original pretrained model weights frozen.

The training data comes from the tatsu-lab/alpaca dataset, which contains instruction/input/output examples.

Main Technologies

Python

PyTorch

Hugging Face Transformers

Hugging Face Datasets

PEFT

TRL

LoRA

TinyLlama 1.1B

Alpaca instruction dataset

Google Colab

2. Project Workflow

The complete workflow is:

Pretrained TinyLlama
        |
        v
Load Tokenizer
        |
        v
Baseline Evaluation
        |
        v
Load Alpaca Dataset
        |
        v
Train / Validation Split
        |
        v
Convert Examples to Chat Messages
        |
        v
Apply Chat Template
        |
        v
Configure LoRA
        |
        v
Supervised Fine-Tuning
        |
        v
Fine-Tuned TinyLlama + LoRA
        |
        +----------------------+
        |                      |
        v                      v
Qualitative Evaluation     Held-Out Evaluation
        |
        v
Save LoRA Adapter
        |
        v
Optional Adapter Merge
        |
        v
Final Standalone Model

3. Model Used

The project uses:

TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

TinyLlama is a relatively small language model, making it suitable for experimentation with instruction tuning and parameter-efficient fine-tuning.

The project does not train the language model from scratch.

Instead:

Pretrained Model
       +
LoRA Adaptation
       =
Instruction-Tuned Model

4. Dataset

The project uses:

tatsu-lab/alpaca

The dataset contains approximately 52,002 examples and includes fields such as:

instruction
input
output
text

An example can conceptually be represented as:

Instruction:
Explain machine learning.

Input:
<optional additional context>

Output:
Machine learning is ...

The notebook converts these examples into conversational messages before training.

5. Why LoRA?

Full fine-tuning updates all parameters of the pretrained model.

For a language model, this can require substantial GPU memory and computation.

LoRA uses a different approach:

Original pretrained weights
        |
        | frozen
        v
     Base Model
        |
        +----> Small trainable LoRA matrices

The base model remains unchanged while the LoRA parameters learn the task-specific adaptation.

Advantages of LoRA

Fewer trainable parameters

Lower memory requirements than full fine-tuning

Faster experimentation

Smaller adapter files

The original base model can be reused

Multiple task-specific adapters can be maintained separately

6. LoRA Configuration

The project uses the following LoRA configuration:

r = 16
lora_alpha = 32
lora_dropout = 0.05

Target modules:

q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj

Meaning of the parameters

r = 16

The LoRA rank controls the size/capacity of the low-rank update.

A larger rank can provide more adaptation capacity but increases the number of trainable parameters.

lora_alpha = 32

This controls the scaling of the LoRA update.

lora_dropout = 0.05

Dropout is applied to the LoRA path during training.

Target modules

The project applies LoRA to important projection layers in the Transformer architecture:

Attention:
q_proj
k_proj
v_proj
o_proj

MLP:
gate_proj
up_proj
down_proj

7. Environment Setup

The notebook installs the main libraries using:

pip install -U transformers datasets accelerate peft trl

The project intentionally avoids depending on torchao because the original notebook produced a compatibility warning related to the installed PyTorch version.

The original environment reported a warning involving:

torch 2.10.0+cu128

and a torchao C++ extension.

The cleaned notebook therefore does not require torchao.

8. Step 1 — Install Dependencies

Run:

!pip install -q -U transformers datasets accelerate peft trl

These packages provide:

Package

Purpose

transformers

Model and tokenizer

datasets

Dataset loading and processing

accelerate

Training/device management

peft

LoRA implementation

trl

Supervised fine-tuning workflow

After installation, restart the runtime only if Colab requests it.

9. Step 2 — Check the Environment

The notebook checks:

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

A GPU is strongly recommended for this project because model loading and fine-tuning are computationally intensive.

10. Step 3 — Define Project Configuration

Important configuration values include:

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATASET_ID = "tatsu-lab/alpaca"

SEED = 42
EVAL_SPLIT = 0.02

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

MAX_SEQ_LENGTH = 512

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9

The configuration is kept in one place so that experiments can be reproduced and modified easily.

11. Step 4 — Load the Tokenizer

The tokenizer converts text into token IDs that the model can process.

The notebook loads:

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

If the tokenizer does not have a padding token, the notebook uses:

tokenizer.pad_token = tokenizer.eos_token

This avoids padding-related problems during batching.

12. Step 5 — Load the Base Model

The pretrained model is loaded before fine-tuning.

The cleaned workflow uses:

torch_dtype=torch.float16
device_map="auto"

This allows the model to use available hardware more efficiently.

The base model is first used for a baseline evaluation.

13. Step 6 — Baseline Generation

Before training, it is useful to know how the original model responds.

The notebook defines a generation function.

A key correction from the original notebook is that tokenized inputs are explicitly moved to the same device as the model.

Conceptually:

inputs = tokenizer(..., return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

This prevents the previous warning:

input_ids is on cpu, whereas the model is on cuda

Generation parameters

The cleaned notebook uses:

max_new_tokens=150
temperature=0.7
top_p=0.9

Only max_new_tokens is used for output length.

The original notebook specified both:

max_new_tokens = 150
max_length = 2048

which caused the Transformers warning that max_new_tokens takes precedence.

The cleaned notebook removes that conflict.

14. Step 7 — Baseline Evaluation

A small collection of test prompts is used to inspect the original model.

For example:

Explain machine learning in simple terms.

The purpose of this stage is not to produce a formal accuracy score.

It establishes a qualitative baseline before instruction tuning.

15. Step 8 — Load the Alpaca Dataset

The dataset is loaded with:

from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")

The dataset contains approximately:

52,002 examples

The main fields are:

instruction
input
output
text

16. Step 9 — Train / Evaluation Split

The notebook creates a train/evaluation split.

The configured evaluation fraction is:

2%

The original dataset was approximately:

Training examples: 50,961
Evaluation examples: 1,041

The split is created with a fixed random seed:

SEED = 42

Using a fixed seed makes the experiment more reproducible.

17. Step 10 — Convert Alpaca Examples to Messages

The raw Alpaca records are converted into conversational messages.

The structure is approximately:

[
    {
        "role": "user",
        "content": instruction
    },
    {
        "role": "assistant",
        "content": output
    }
]

If an additional input field exists, it is incorporated into the user content.

This produces a format compatible with the model's chat template.

18. Step 11 — Chat Template

The project uses a custom chat format based on:

### User:
...

### Assistant:
...

The tokenizer's chat template is configured so that the same format is used consistently during preprocessing and generation.

Consistency is important:

Training format
       =
Inference format

If the model is trained using one conversation structure and tested using a completely different structure, the resulting behavior may be less consistent.

19. Step 12 — Sanity Check the Chat Template

Before training, the notebook formats one example and prints the resulting text.

This step verifies that:

The user message appears correctly.

The assistant response appears correctly.

The special formatting is applied correctly.

The tokenizer can process the resulting conversation.

This is a simple but important debugging step.

20. Step 13 — Configure LoRA

The LoRA configuration is created using PEFT.

Conceptually:

from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

The exact training code is contained in the Colab notebook.

21. Step 14 — Calculate Warmup Steps

The original notebook used:

warmup_ratio

The newer training configuration uses an explicit:

warmup_steps

This avoids the warning about the deprecated warmup_ratio argument in the training configuration used by the original notebook.

The notebook calculates warmup steps from the number of training steps.

22. Step 15 — Configure Supervised Fine-Tuning

The project uses the TRL supervised fine-tuning workflow.

Important training settings include:

Epochs: 1
Batch size: 2
Gradient accumulation: 8
Learning rate: 1e-4
Weight decay: 0.01
Maximum sequence length: 512

Gradient accumulation effectively allows the optimizer to see a larger effective batch than the per-device batch size alone.

23. Step 16 — Create the SFT Trainer

The trainer combines:

Base Model
+
Tokenizer / Processing
+
Training Dataset
+
Evaluation Dataset
+
LoRA Configuration
+
Training Configuration

The trainer is responsible for the supervised fine-tuning loop.

24. Step 17 — Check Trainable Parameters

One important benefit of LoRA is that only a small portion of the model parameters are trainable.

The notebook prints trainable parameter information.

Conceptually:

Total parameters
Trainable parameters
Percentage trainable

This confirms that the experiment is actually using parameter-efficient fine-tuning rather than full-model training.

25. Step 18 — Train the Model

Training is started using:

trainer.train()

The original training run completed approximately:

3,186 steps
Epoch: 1.0
Training loss: approximately 1.2437

These values describe the original run represented in the source notebook.

Training loss alone should not be interpreted as a complete measure of instruction-following quality.

26. Step 19 — Fine-Tuned Generation

After training, the adapted model is used for generation.

The same important device correction is applied:

inputs = {k: v.to(model.device) for k, v in inputs.items()}

This prevents CPU/GPU input mismatch.

Generation again uses:

max_new_tokens=150
temperature=0.7
top_p=0.9

There is no simultaneous max_length argument.

27. Step 20 — Test a Single Prompt

A simple prompt can be used:

Explain artificial intelligence in simple terms.

The response can be compared with the baseline model.

This provides an easy way to see whether the model learned the expected instruction-following style.

28. Step 21 — Qualitative Evaluation

The notebook evaluates multiple examples.

The comparison can be organized as:

Prompt
   |
   +--> Base TinyLlama response
   |
   +--> LoRA fine-tuned response

The purpose is to inspect changes in:

Instruction following

Response structure

Relevance

Completeness

Clarity

This is a qualitative evaluation.

It should not be presented as a formal benchmark score unless a standardized benchmark is actually run.

29. Step 22 — Held-Out Evaluation

The evaluation split is not used for gradient updates.

This allows the project to test the adapted model on examples that were kept separate from the training data.

The notebook evaluates a small set of held-out Alpaca examples.

Some outputs may improve while others may remain imperfect.

This is expected for a small instruction-tuning experiment and should be reported honestly.

30. Step 23 — Save the LoRA Adapter

The adapter is saved separately.

The output includes files such as:

adapter_config.json
adapter_model.safetensors
README.md
tokenizer.json
tokenizer_config.json
chat_template.jinja

The most important LoRA files are:

adapter_config.json
adapter_model.safetensors

The adapter contains the learned LoRA parameters rather than a complete copy of the original model.

31. Step 24 — Merge LoRA with the Base Model

The notebook also provides an optional merge step.

Conceptually:

Base TinyLlama
      +
LoRA Adapter
      |
      v
Merged Model

The merged model can then be saved as a standalone model directory.

The original workflow successfully produced files including:

config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja

32. Adapter vs Merged Model

There are two useful ways to store the result.

Option 1 — LoRA Adapter

Base Model
+
Adapter

Advantages:

Smaller task-specific files

Easy to keep multiple adapters

Original base model remains reusable

Option 2 — Merged Model

Single model directory

Advantages:

Easier deployment in environments where a standalone model is preferred

No separate LoRA loading step

The project saves both forms.

33. Project Directory Structure

A recommended project structure is:

TinyLlama-LoRA-Alpaca/
│
├── TinyLlama_Alpaca_LoRA_Clean_Colab.ipynb
│
├── README.md
│
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
│
└── merged_model/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── chat_template.jinja

The actual generated directories may differ depending on the runtime and save paths.

34. Important Corrections Made from the Original Notebook

The cleaned notebook fixes several issues found in the original workflow.

Correction 1 — CPU/GPU Input Mismatch

Original warning:

input_ids is on cpu, whereas the model is on cuda

Fix

Move tokenized inputs to the model's device:

inputs = {k: v.to(model.device) for k, v in inputs.items()}

Correction 2 — Conflicting Generation Length Arguments

Original warning:

Both max_new_tokens (=150) and max_length (=2048) have been set.
max_new_tokens will take precedence.

Fix

Use:

max_new_tokens=150

and remove the conflicting max_length setting from generation calls.

Correction 3 — Warmup Configuration

The original training workflow generated a warning related to:

warmup_ratio

The cleaned notebook calculates an explicit:

warmup_steps

Correction 4 — TorchAO Compatibility Warning

The original environment reported:

Skipping import of cpp extensions due to incompatible torch version.
Please upgrade to torch >= 2.11.0
(found 2.10.0+cu128)

The cleaned notebook removes the unnecessary torchao dependency instead of making it a required component.

35. Common Errors and Solutions

Error 1 — CUDA Not Available

Check:

import torch
print(torch.cuda.is_available())

If it returns:

False

enable a GPU runtime in Colab.

Error 2 — Out of Memory

Possible adjustments:

Reduce batch size
Reduce max sequence length
Reduce LoRA rank
Increase gradient accumulation

For example:

BATCH_SIZE = 1

and keep gradient accumulation higher.

Error 3 — CPU/GPU Mismatch

Make sure generation inputs are moved to the model device:

inputs = {k: v.to(model.device) for k, v in inputs.items()}

Error 4 — Padding Token Error

Use:

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

Error 5 — Generation Warning About max_length

Do not use both:

max_length=...
max_new_tokens=...

Use:

max_new_tokens=150

for this project.

Error 6 — Training Runtime Restart

After package installation, Colab may request a runtime restart.

If that happens:

Restart the runtime.

Run the notebook cells from the beginning.

Do not skip model/tokenizer initialization cells.

36. Reproducibility

The project uses:

SEED = 42

A fixed seed helps make the dataset split and experiment more reproducible.

However, exact training results can still vary because of hardware, library versions, CUDA behavior, and other runtime factors.

37. What I Learned from This Project

This project demonstrates practical experience with:

LLMs

Pretrained language models

TinyLlama

Causal language modeling

Instruction tuning

Fine-Tuning

LoRA

PEFT

Supervised fine-tuning

Adapter training

Model merging

Hugging Face Ecosystem

Transformers

Datasets

PEFT

TRL

Tokenizers

Training

Train/evaluation split

Gradient accumulation

Learning rate

Weight decay

Warmup steps

Sequence length

Inference

Tokenization

Device management

Temperature

Top-p sampling

Maximum generated tokens

Evaluation

Baseline comparison

Qualitative evaluation

Held-out examples

Project Title

Parameter-Efficient Fine-Tuning of TinyLlama using LoRA for Instruction Following

Resume Bullet 1

Fine-tuned the TinyLlama 1.1B language model on the Alpaca instruction dataset using parameter-efficient LoRA adaptation.

Resume Bullet 2

Implemented an end-to-end Hugging Face workflow covering dataset preprocessing, chat-template formatting, supervised fine-tuning, inference, qualitative evaluation, and model saving.

Resume Bullet 3

Applied LoRA to Transformer attention and MLP projection layers and used gradient accumulation to support efficient training.

Resume Bullet 4

Built separate baseline and fine-tuned inference pipelines with device-aware tensor placement and controlled text generation.

Resume Bullet 5

Saved both the LoRA adapter and an optional merged model for downstream inference and deployment.

“My project is parameter-efficient fine-tuning of TinyLlama using LoRA for instruction following. I used the TinyLlama 1.1B pretrained model and fine-tuned it on the Alpaca instruction dataset. Instead of updating all model parameters, I used LoRA on the attention and MLP projection layers. I created a train/evaluation split, converted the Alpaca records into chat messages, applied a consistent chat template, trained the model using supervised fine-tuning, and compared the fine-tuned model with the baseline using qualitative and held-out examples. Finally, I saved the LoRA adapter and also created an optional merged model.”

40. Interview Questions

Q1. What is LoRA?

LoRA stands for Low-Rank Adaptation. It adapts a pretrained model by training low-rank matrices instead of updating all original model parameters.

Q2. Why use LoRA?

It reduces the number of trainable parameters and makes fine-tuning more memory-efficient than full fine-tuning.

Q3. What model did you use?

TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

Q4. What dataset did you use?

tatsu-lab/alpaca

Q5. What is instruction tuning?

Instruction tuning trains a language model on examples where an instruction is associated with an expected response, improving its ability to respond to natural-language tasks.

Q6. What is the difference between the adapter and merged model?

The adapter stores the LoRA-specific learned parameters separately, while the merged model combines the LoRA update with the base model weights.

Q7. Why do you need a chat template?

It provides a consistent format for representing user and assistant messages during training and inference.

Q8. Why was the CPU/GPU warning generated?

The input tensors were on the CPU while the model was on CUDA. Moving the tokenized inputs to the model's device fixes the mismatch.

Q9. Why was max_new_tokens preferred?

The original notebook specified both max_length and max_new_tokens, causing a warning. The cleaned workflow uses max_new_tokens alone to control the generated continuation.

Q10. Is the training loss an accuracy score?

No. Training loss measures the model's optimization objective during training. It should not be presented as an instruction-following accuracy metric.

41. Limitations

This project is an instruction-tuning experiment rather than a complete production LLM evaluation.

Important limitations include:

The model is relatively small.

Only one training epoch was used in the original workflow.

Qualitative evaluation was used for the final response comparison.

Some generated responses can still contain factual or reasoning errors.

A small held-out evaluation is not equivalent to a standardized benchmark.

Fine-tuning does not guarantee factual correctness.

Exact results can depend on the software and hardware environment.

42. Future Improvements

Possible extensions include:

Evaluate on a larger held-out test set.

Add standardized instruction-following benchmarks.

Compare different LoRA ranks such as 8, 16, and 32.

Compare different learning rates.

Experiment with sequence lengths.

Train for additional epochs with appropriate validation.

Compare LoRA with QLoRA.

Add experiment tracking.

Add automated evaluation metrics.

Deploy the final model through an inference application.

43. How to Run the Project

Step 1

Open Google Colab.

Step 2

Upload:

TinyLlama_Alpaca_LoRA_Clean_Colab.ipynb

Step 3

Select a GPU runtime.

Step 4

Run the installation cell.

Step 5

Run the environment check.

Step 6

Run the tokenizer and model-loading cells.

Step 7

Run the baseline evaluation.

Step 8

Load and split the Alpaca dataset.

Step 9

Prepare the chat-format training data.

Step 10

Configure LoRA.

Step 11

Create the SFT trainer.

Step 12

Run training.

Step 13

Evaluate the fine-tuned model.

Step 14

Save the LoRA adapter.

Step 15

Merge and save the model if required.

44. Final Result

The final project produces two useful artifacts:

1. LoRA Adapter
2. Merged Fine-Tuned Model

The complete pipeline demonstrates how a pretrained small language model can be adapted for instruction-following tasks using parameter-efficient fine-tuning.

45. Repository

Recommended repository name:

TinyLlama-LoRA-Alpaca-Instruction-Tuning

Recommended files:

TinyLlama-LoRA-Alpaca-Instruction-Tuning/
│
├── README.md
├── TinyLlama_Alpaca_LoRA_Clean_Colab.ipynb
├── adapter/
└── merged_model/

License and Model/Data Attribution

Before publishing the model weights, adapter, or dataset-derived artifacts publicly, check the applicable licenses and usage terms for the TinyLlama model and the Alpaca dataset.

The project should clearly identify the original model and dataset sources in the repository.

Summary

This project implements a complete TinyLlama instruction-tuning pipeline using LoRA and the Alpaca dataset.

The main stages are:

Load pretrained TinyLlama
        ↓
Evaluate baseline
        ↓
Load Alpaca
        ↓
Prepare instruction/chat data
        ↓
Configure LoRA
        ↓
Supervised fine-tuning
        ↓
Evaluate fine-tuned model
        ↓
Save LoRA adapter
        ↓
Merge adapter with base model
        ↓
Final model

Core technologies:

Python
PyTorch
Transformers
Datasets
PEFT
TRL
LoRA
TinyLlama
Alpaca
Google Colab
