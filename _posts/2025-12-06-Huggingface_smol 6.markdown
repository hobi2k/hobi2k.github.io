---
layout: post
title:  "허깅페이스 스몰 코스 - Hugging Face Jobs을 이용한 SmolLM3 훈련"
date:   2025-12-06 00:10:22 +0900
categories: Huggingface_smol
---

# Hugging Face Jobs을 이용한 SmolLM3 훈련

Hugging Face Jobs는 GPU 설정, CUDA 버전 충돌, 도커 이미지 관리, 의존성 설치 문제 없이
바로 LLM 학습을 돌릴 수 있는 완전 관리형 환경이다.

특히 SmolLM3 SFT처럼 GPU를 지속적으로 사용하는 작업에서
Jobs는 안정적이고 재현성 있는 환경을 제공한다.

## Hugging Face Jobs를 사용하는 이유

SmolLM3 SFT는 로컬 환경에서 다음 문제가 자주 발생한다.

- CUDA/PyTorch 버전 충돌
- VRAM 부족
- GPU 스케줄링 및 백그라운드 프로세스 간섭
- 장시간 실행 중 노트북/로컬환경의 불안정

Jobs를 사용하면 다음과 같은 장점이 생긴다.

- 고성능 GPU 제공 (A10G, A100, L4 등)
- 의존성 자동 관리 (uv 기반)
- 학습이 끝나면 자동 종료 -> 비용 절약
- Hub과 완전 통합 -> 결과 자동 업로드
- CLI 기반 재현성 높은 실행 가능
- 로그, 메트릭, 실패 원인을 Hub에서 추적 가능

결론: SFT를 안정적으로 돌리려면 가장 추천되는 비용 효율적 환경이다.

## 사전 요구사항

Jobs를 사용하려면 다음이 필요하다.

- Hugging Face Pro, Team, Enterprise 플랜
(GPU 있는 Job 실행은 무료 계정에서 불가)

- CLI 로그인

```python
hf auth login
```

로그인하면 Jobs, Spaces, Hub 업로드 등이 가능해진다.

## Hugging Face Jobs로 SFT 실행하는 두 방법

Jobs에서 SmolLM3을 학습하는 방법은 크게 2가지이다.

## 방법 A. TRL의 공식 SFT 스크립트를 직접 실행

Jobs는 URL로 Python 스크립트를 받아 바로 실행할 수 있다.

```python
hf jobs uv run \
    --flavor a10g-large \
    --timeout 2h \
    --secrets HF_TOKEN \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path HuggingFaceTB/SmolLM3-3B-Base \
    --dataset_name HuggingFaceTB/smoltalk2_everyday_convs_think \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --output_dir smollm3-sft-jobs \
    --push_to_hub \
    --hub_model_id your-username/smollm3-sft \
    --report_to trackio
```

이 스크립트는 TRL 팀이 유지보수하는 표준 SFT 스크립트이며,
대부분의 학습 목적에 그대로 사용 가능하다.

### 방법 B. 사용자 정의 스크립트 실행

직접 Python 스크립트를 만들고 inline dependency로 uv로 실행할 수도 있다.

예시: sft_training.py

```python
# /// script
# dependencies = [
#     "trl[sft]>=0.7.0",
#     "transformers>=4.36.0", 
#     "datasets>=2.14.0",
#     "accelerate>=0.24.0",
#     "peft>=0.7.0"
# ]
# ///

from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
dataset = load_dataset("HuggingFaceTB/smoltalk2", "SFT")

config = SFTConfig(
    output_dir="./smollm3-jobs-sft",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    max_steps=1000,
    push_to_hub=True,
    hub_model_id="your-username/smollm3-jobs-sft"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["smoltalk_everyday_convs_reasoning_Qwen3_32B_think"],
    args=config,
)
trainer.train()
```

실행

```python
hf jobs uv run \
    --flavor a10g-large \
    --timeout 2h \
    --secrets HF_TOKEN \
    sft_training.py
```

## 어떤 GPU를 선택해야 하나?

SmolLM3-3B SFT 기준으로 정리한 권장 사항이다.

### SmolLM3-3B 기준 추천


| 용도                  | 추천 Flavor             |
| ------------------- | --------------------- |
| 가장 비용 효율 + 충분한 VRAM | **a10g-large (24GB)** |
| 가장 빠르고 안정적          | **a100-large (40GB)** |
| 분산 학습               | **l4x4**, **l4x1**    |


비용 효율성

- t4-small (16GB): 속도는 느리지만 작은 실험 가능.

## LoRA/PEFT 학습을 Jobs에서 실행하기

TRL 스크립트는 기본적으로 PEFT 옵션을 지원한다.

```python
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
  --model_name_or_path HuggingFaceTB/SmolLM3-3B-Base \
  --dataset_name HuggingFaceTB/smoltalk2_everyday_convs_think \
  --output_dir smollm3-lora-sft-jobs \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --max_steps 1000 \
  --use_peft \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear
```

LoRA를 사용하면 VRAM 사용량은 크게 줄고 학습 속도도 빨라진다.

## Job 모니터링

CLI 또는 웹페이지에서 상태 확인 가능.

CLI

```python
hf jobs ps -a
hf jobs inspect <job_id>
hf jobs logs <job_id> --follow
hf jobs cancel <job_id>
```

웹 UI
https://huggingface.co/settings/jobs

- 실시간 로그
- 시작/종료 시간
- GPU 사용량
- 에러 메시지

## Trackio로 학습 모니터링

Jobs 환경에서도 Trackio를 그대로 사용할 수 있다.

메트릭은 자동으로 Hugging Face Hub와 연결되며,
Loss, LR, Token 수 등을 시각적으로 확인할 수 있다.

## 비용 가이드

SmolLM3 1000-step SFT 기준 예상 비용


| GPU        | 시간당 비용(대략) | 예상 학습 시간 | 총 비용 |
| ---------- | ---------- | -------- | ---- |
| l4x1       | $3–4       | 30–90분   | $2–6 |
| a10g-large | $4–6       | 30–60분   | $3–6 |
| a100-large | $8–12      | 20–40분   | $4–8 |


비용 절약 팁

- batch size 줄이고 gradient accumulation 사용
- 먼저 200~300 step로 설정해 파이프라인 검증
- l4x1 같은 저렴한 GPU로 테스트 후 A100으로 넘어가기
- timeout을 반드시 설정해서 비정상 종료 방지

## 문제 해결(Troubleshooting)

1. Out of Memory

- per_device_train_batch_size 줄이기
- max_length 줄이기
- LoRA 사용
- gradient checkpointing 켜기

2. Timeout

- timeout 연장
- GPU를 상위 스펙으로 바꾸기
- 학습 step 수 줄이기

3. Authentication 실패

- HF_TOKEN의 permission 확인 (write 권한 필수)
- hf auth login 재시도
- Pro 플랜 이상인지 확인

## 핵심 흐름 정리

SFT를 Jobs에서 돌리는 전체 과정은 아래와 같다.

1. Hugging Face 로그인
2. GPU flavor 선택
3. TRL 공식 스크립트 또는 커스텀 스크립트 준비
4. hf jobs uv run 명령으로 실행
5. Hub에서 로그 모니터링
6. 학습 완료 모델 자동 업로드
7. 필요한 경우 LoRA merge 또는 추가 tuning
8. Inference 테스트 및 배포

Jobs는 로컬의 복잡함을 제거하고
LLM 학습에만 집중할 수 있는 최적의 환경이다.

참고자료
Huggingface, a smol course, https://huggingface.co/learn