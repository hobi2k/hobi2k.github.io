---
layout: post
title:  "허깅페이스 스몰 코스 - Direct Preference Optimization with SmolLM3"
date:   2025-12-09 00:10:22 +0900
categories: Huggingface_smol
---

# Direct Preference Optimization with SmolLM3

이 글에서는 SmolLM3 모델을 DPO 방식으로 학습하여 **선호도 기반의 정렬 모델(aligned model)**을 만들고,
Hugging Face Jobs를 활용해 클라우드 환경에서 대규모 학습을 수행한 뒤,
리더보드 제출까지 완료하는 전체 과정을 다룬다.

## 실습 목표

- DPO 학습 흐름 이해
- Preference dataset을 이용해 SmolLM3 정렬
- 로컬 테스트 -> HF Jobs로 확장
- 학습된 모델을 Hugging Face Hub에 업로드
- 결과를 리더보드로 제출

## 환경 

```python
pip install "transformers>=4.56.1" "trl>=0.23.0" "datasets>=4.1.0"
pip install "accelerate>=1.10.1" "peft>=0.17.0" "trackio"
```

4. 기본 코드 템플릿: 장비 확인 + 로그인

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

# Device 확인
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU 없음 → HF Jobs 사용 필요")

# HF 로그인
login()
```

## DPO 데이터 구조 이해하기

DPO는 다음 3개 필드를 가진 “선호 쌍 데이터”를 사용한다:

- prompt
- chosen(선호된 답변)
- rejected(덜 선호된 답변)

예시 출력

```python
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
sample = dataset[0]

print(sample["prompt"])
print(sample["chosen"][:200])
print(sample["rejected"][:200])
```

데이터가 이렇게 구성된 이유는

- chosen 확률을 높이고
- rejected 확률을 낮추는 방향으로

모델을 직접 업데이트하기 위해서다.

## 로컬 테스트용 DPO 학습 설정(옵션)

로컬 GPU에서 간단히 “훈련이 되는지”만 확인하는 코드이다.

```python
small = load_dataset("Anthropic/hh-rlhf", split="train").select(range(1000))

model_name = "HuggingFaceTB/SmolLM3-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=50,
    output_dir="./local_dpo_test",
)

trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=small,
    processing_class=tokenizer,
)

print("DPO trainer 생성 완료")
```

## Hugging Face Jobs에서 DPO 학습 수행

HF Jobs에서는 uv 스크립트 기반으로 쉽게 학습을 제출할 수 있다.
다음은 DPO 학습 전체가 포함된 예시 스크립트이다.

```python
dpo_training.py
# dpo_training.py
# /// script
# dependencies = [
#   "trl[dpo]>=0.7.0",
#   "transformers>=4.36.0",
#   "datasets>=2.14.0",
#   "accelerate>=0.24.0",
#   "torch>=2.0.0"
# ]
# ///

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    # 1. Preference dataset 로드
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    train_dataset = dataset.select(range(10000))

    # 2. Reference + Policy 모델 준비 (SFT된 SmolLM3)
    model_name = "HuggingFaceTB/SmolLM3-3B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. DPO 설정
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-7,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=1000,
        max_prompt_length=512,
        max_length=1024,
        logging_steps=50,
        save_steps=250,
        output_dir="./smollm3-dpo-aligned",
        push_to_hub=True,
        hub_model_id="your-username/smollm3-dpo-aligned",
        report_to="trackio",
        bf16=True,
        gradient_checkpointing=True,
    )

    # 4. Trainer 생성
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 5. 학습 시작
    trainer.train()

if __name__ == "__main__":
    main()
```

8. HF Jobs로 학습 제출

```python
hf jobs uv run \
  --flavor a100-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  dpo_training.py
```

9. TRL 공식 스크립트로 학습

```python
hf jobs uv run \
  --flavor a100-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/dpo.py" \
  --model_name_or_path HuggingFaceTB/SmolLM3-3B \
  --dataset_name Anthropic/hh-rlhf \
  --learning_rate 5e-7 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 1000 \
  --beta 0.1 \
  --max_prompt_length 512 \
  --max_length 1024 \
  --output_dir smollm3-dpo-aligned \
  --push_to_hub \
  --hub_model_id your-username/smollm3-dpo-aligned \
  --report_to trackio
```

## Job 모니터링

```python
hf jobs ps -a
hf jobs logs <job_id> --follow
hf jobs inspect <job_id>
```

## 학습된 모델 로컬 평가

```python
from transformers import pipeline

model_name = "your-username/smollm3-dpo-aligned"
pipe = pipeline("text-generation", model=model_name, tokenizer=model_name)

prompts = [
    "How should I handle a disagreement with a friend?",
    "How do I become more productive?",
]

for p in prompts:
    out = pipe(p, max_length=200, temperature=0.7)[0]["generated_text"]
    print("\nPrompt:", p)
    print("Response:", out[len(p):].strip())
```

참고자료
Huggingface, a smol course, https://huggingface.co/learn