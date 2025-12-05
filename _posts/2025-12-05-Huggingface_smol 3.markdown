---
layout: post
title:  "허깅페이스 스몰 코스 - 지도 파인튜닝"
date:   2025-12-05 00:10:22 +0900
categories: Huggingface_smol
---

# 지도 파인튜닝

Supervised Fine-Tuning(SFT)은 사전학습된 언어 모델을 특정 작업과 응답 스타일에 맞게 조정하는 과정이다.
SmolLM3의 경우, 강력한 pre-training 위에 SFT를 적용하여 “지시를 따르는 모델”로 완성된다.

## Supervised Fine-Tuning이란

SFT는 라벨이 있는 입력-출력 데이터로 모델을 추가 학습시키는 단계다.
여기서 중요한 점은 새로운 지식을 학습시키는 것이 아니라, 이미 보유한 지식을 적절하게 사용하는 방법을 학습시키는 것이다.

**Pre-training과의 차이**


| 단계           | 목적          | 입력 데이터  | 비용     | 결과           |
| ------------ | ----------- | ------- | ------ | ------------ |
| Pre-training | 언어 이해 형성    | 대규모 텍스트 | 매우 높음  | 언어, 사실·문법 인코딩 |
| SFT          | 특정 행동 패턴 학습 | 지시-응답 쌍 | 비교적 낮음 | 모델 행동 변화     |


즉, SFT는 기존 능력을 재배치하고 특정 목적에 맞게 정렬(alignment)시키는 과정이다.

**SFT가 효과적인 이유**

- Pre-training에서 이미 풍부한 표현을 학습
- SFT는 특정 스타일, 형식, 응답 패턴을 학습시키는 데 초점을 둔다
- 적은 예시로도 동작 변화를 유도할 수 있다

학술 연구에서도 일관되게 밝혀진 핵심은 다음과 같다.

- 행동 적응(Behavioral adaptation): 지시를 감지하고 그에 맞는 형식으로 답변
- 작업 특화(Task specialization): 기존 지식을 특정 도메인 방식으로 표현
- 안전성 강화(Safety alignment): 원치 않는 행동 억제

## SmolLM3의 SFT 과정

SmolLM3는 다음 단계를 통해 instruct 모델로 완성되었다.

1. Base 모델 학습: 11T 토큰으로 언어 이해 형성
2. SFT: SmolTalk2 등 고품질 데이터로 지시 수행 능력 학습
3. Preference Alignment(APO 등): 사용자 선호에 맞는 응답 강화

이 3단계를 통해 모델은 "지식이 있고 도움이 되며 안전한" 어시스턴트로 변화한다.

## 언제 Supervised Fine-Tuning을 해야 하는가

다음 질문에 "예"라고 대답할수록 SFT의 필요성이 높다.

- 기존 instruct 모델에 프롬프트 엔지니어링만으로 문제 해결이 어려운가
- 출력 형식이 매우 엄격하게 요구되는가
- 전문 도메인(의료, 법률, 기술 문서)처럼 일반 모델이 다루기 힘든가
- 최소 1,000개 이상의 고품질 데이터셋을 보유했는가
- 학습 리소스를 확보할 수 있는가

SFT는 비용과 시간이 들어가므로, 충분히 필요할 때만 수행하는 것이 가장 좋다.

## SFT 전체 과정 개요

SFT는 다음 네 단계로 구성된다.

1. 데이터 준비
2. 환경 설정 (GPU, Jobs, Colab, Cloud 등)
3. 훈련 구성 (하이퍼파라미터 설정)
4. 모니터링 및 평가

아래에서 하나씩 자세히 설명한다.

## 데이터셋 준비와 선택

데이터 품질은 SFT 성공의 핵심 요소다.
모델은 제공한 예시를 그대로 모방하므로 잘 구성된 데이터는 필수적이다.

**데이터 구성 기본 단위**

각 예시는 다음 요소로 구성된다.

1. Prompt (User)
2. Completion (Assistant)
3. Optional Context

**데이터셋 예시**

- SmolTalk2: SmolLM3 훈련에 사용된 고품질 대화 데이터
- 도메인 특화 데이터: 의료, 법률, 고객지원 등
- 자체 수집 데이터: 특정 형식 요구, 스타일 기반 응답 등

**데이터 크기 가이드**

- 최소: 1,000개
- 권장: 10,000개 이상
- 핵심: 양보다 질

## SFT 환경 구성

훈련을 위한 GPU 옵션은 다음과 같다.

1. 로컬 GPU (최소 16GB VRAM)
2. Hugging Face Jobs (가장 안정적이며 설정 부담 없음)
3. Google Colab/Tensor Notebooks
4. AWS/GCP/Azure Cloud GPU

학습을 위한 SmolLM3 최소 요구사항은 대략 다음과 같다.

- GPU VRAM: 16GB 이상
- 예시: RTX 4080, A10G, L4, T4(적절한 설정 필요)

## SFT 하이퍼파라미터 구성

효과적인 SFT를 위해 다음 요소를 조절해야 한다.

**주요 하이퍼파라미터**

Learning Rate (5e-5 ~ 1e-4)

- 너무 높으면 발산
- 너무 낮으면 학습 속도 저하
- 권장: 1e-4

**Batch Size (4~16)**

- VRAM 상황에 따라 gradient accumulation을 활용

예시

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

**Max Sequence Length (512 ~ 4096)**

- 대화 길이와 응답 길이에 따라 설정
- 길어질수록 VRAM 소비 증가

**Training Steps (1000 ~ 5000)**

- 데이터 크기와 학습 목표에 따라 설정
- validation loss 기준 조기 종료 고려

## SFT 모니터링과 평가 방법

SFT는 단순히 loss 감소만 보는 것이 아니다.
**질적 평가(모델 응답 출력 검사)**가 매우 중요하다.

**모니터링 요소**

1. Training Loss

- 서서히 감소하는 것이 이상적
- 급격한 변동은 학습 불안정 신호

2. Validation Loss

- Training Loss와 격차가 크면 과적합
- 조기 종료 기준으로 사용

3. Sample Outputs

- 형식 준수 여부
- 문맥 유지 능력
- 반복, 환각(Hallucination) 여부

4. Resource Usage

- GPU 메모리
- 학습 속도

## Loss 패턴 이해

SFT 과정에서 loss 그래프는 세 단계로 구성된다.

1. 초기 급격한 하락
2. 점진적 안정화
3. 수렴

**문제 패턴**

- 과적합: validation loss 증가
- 학습 부족: loss 감소 폭이 작음
- 암기 현상: loss가 비정상적으로 낮으나 일반화 성능 저하

## Trackio를 통한 실험 관리

Trackio는 Hugging Face 기반의 경량 실험 추적 도구다.

**장점**

- wandb와 API 호환
- 완전 무료
- 로컬 또는 Spaces에서 대시보드 실행

예시

```python
import trackio
trackio.init(project="sft-exp")
trackio.log({"train_loss": 0.5})
trackio.finish()
```

TRL + Trackio 조합은 매우 효율적이다.

## 기대되는 데이터 형식

SFTTrainer는 다음 네 가지 형식을 모두 지원한다.

1. 텍스트

```python
{"text": "Hello world."}
```

2. Conversational

```python
{"messages": [{"role":"user","content":"Hi"}, {"role":"assistant","content":"Hello"}]}
```

3. Prompt-Completion

```python
{"prompt": "Question:", "completion": "Answer."}
```

4. Conversational Prompt-Completion

```python
{"prompt":[...], "completion":[...]}
```

데이터가 다른 구조라면 전처리가 필요하다.

## Chat Template의 역할 (훈련 시)

SFTTrainer는 conversational 데이터가 입력되면 자동으로 Chat Template을 적용한다.

이때 중요한 설정

- 훈련 시에는 add_generation_prompt=False
- 템플릿에 포함된 특수토큰을 중복 삽입하지 말 것

훈련 목표는 Cross-Entropy Loss이며, padding 토큰은 mask된다.

## TRL 기반 SFT

TRL은 SFT와 alignment 기법을 위한 대표 라이브러리다.

**장점**

- SFT, DPO, ORPO, PPO 모두 지원
- Hugging Face 모델 및 생태계 긴밀하게 통합

CLI 및 Python API 지원

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")

dataset = load_dataset("HuggingFaceTB/smoltalk2_everyday_convs_think")

config = SFTConfig(
    output_dir="./finetuned",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    max_steps=1000,
    report_to="trackio",
)

trainer = SFTTrainer(model=model, train_dataset=dataset["train"], args=config)
trainer.train()
```

## CLI 기반 예제

```python
trl sft \
    --model_name_or_path HuggingFaceTB/SmolLM3-3B-Base \
    --dataset_name HuggingFaceTB/smoltalk2_everyday_convs_think \
    --output_dir ./sft-model \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --max_steps 1000 \
    --logging_steps 50 \
    --save_steps 200 \
    --report_to trackio \
    --push_to_hub \
    --hub_model_id your-username/smollm3-custom
```

## Serverless Training: Hugging Face Jobs

GPU 환경을 직접 설치하고 관리할 필요 없이, Jobs 플랫폼을 활용해 손쉽게 훈련할 수 있다.
대규모 학습, 안정적인 실행, 모델 버전 관리 등이 매우 편리하다.

## 마무리

- SFT는 모델의 행동을 바꾸는 핵심 과정이다.
- 데이터 품질이 가장 중요한 요소이다.
- 하이퍼파라미터는 보수적으로 설정하며 모니터링을 철저히 해야 한다.
- TRL은 SFT에 가장 적합한 통합 라이브러리다.
- SmolLM3는 학습하기 쉽고 효과적인 compact 모델이다.
- Trackio를 활용해 실험을 체계적으로 관리할 수 있다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn