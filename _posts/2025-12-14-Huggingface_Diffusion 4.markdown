---
layout: post
title:  "허깅페이스 디퓨전 코스 - DreamBooth"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_diffusion
---

# Diffusion Course DreamBooth

## 전체 학습 개념 -> 실전 파이프라인 설계 -> 파인튜닝 가이드


이 글은 **Stable Diffusion 파인튜닝을 “실제로 수행하는 엔지니어”의 관점**에서  
DreamBooth를 하나의 **완결된 파인튜닝 레시피**로 정리한 문서다.

목표는 단순하다.

- DreamBooth가 **무엇을 하는 방법인지 정확히 이해**
- Stable Diffusion **파인튜닝 전체 흐름을 구조적으로 파악**
- 실제 프로젝트에서 **재현 가능한 코드 구조**로 옮길 수 있게 만들기

## DreamBooth는 무엇인가

**DreamBooth는 사전학습된 diffusion 모델에 “새로운 개체/캐릭터/개념”을 few-shot 데이터로 주입하는 파인튜닝 방법이다.**

핵심은 다음이다.

- 모델 구조를 바꾸지 않는다
- 데이터는 매우 적다 (보통 3~10장)
- 대신 **프롬프트 + 손실 설계**로 개념을 “고정”시킨다

## DreamBooth가 필요한 문제 설정

Stable Diffusion을 그대로 쓰면 다음 문제가 생긴다.

- “이 캐릭터”를 정확히 다시 만들 수 없다
- 얼굴, 의상, 정체성이 매번 흔들린다
- 프롬프트로만 일관성을 강제하기 어렵다

DreamBooth는 이 문제를 이렇게 해결한다.

> **“모델에게 새로운 고유 토큰(identifier)을 가르쳐 그 토큰이 특정 대상(캐릭터/사람/물체)을 의미하도록 만든다.”**

## DreamBooth의 핵심 아이디어 (설계 관점)

### 고유 토큰 + 클래스 프롬프트

DreamBooth는 항상 이 형태의 프롬프트를 쓴다.

```python
"a photo of <saya_token> girl"
```

- `<saya_token>` : 새로 학습할 **고유 식별자**
- `girl` : 이미 모델이 알고 있는 **클래스 단어**

이 조합이 중요한 이유:

- 모델은 `girl`이라는 개념을 이미 잘 안다
- `<saya_token>`은 그 클래스 안에서의 **특정 인스턴스**로 학습된다

### Prior Preservation (망각 방지)

DreamBooth는 파인튜닝 중 다음 위험이 있다.

- 모델이 `<saya_token>`만 과도하게 학습
- `girl` 전체 개념을 망각 (catastrophic forgetting)

이를 막기 위해 **prior preservation loss**를 추가한다.

개념적으로는:

- “이 토큰을 학습하되”
- “기존 `girl`의 분포는 유지하라”

그래서 데이터가 두 종류로 나뉜다.

| 데이터 종류 | 목적 |
|-----------|------|
| Instance images | 새 캐릭터 학습 |
| Class images | 기존 분포 유지 |


## Stable Diffusion 파인튜닝에서 실제로 바뀌는 것

### 어떤 파라미터를 학습하는가?

Stable Diffusion의 구성요소:

- Text Encoder (CLIP)
- UNet (denoising model)
- VAE (보통 고정)

DreamBooth에서 일반적인 선택:

- **UNet만 학습**
- Text Encoder는 고정 또는 선택적 학습

이유:
- UNet이 실제 시각적 표현을 담당
- CLIP까지 학습하면 과적합/불안정 위험 증가

### Loss 구조 (핵심)

DreamBooth의 손실은 단일 MSE가 아니다.
개념적으로는 다음 합이다.

total_loss =
instance_loss

prior_preservation_weight * class_loss


- instance_loss: `<saya_token>` 이미지에 대한 noise prediction loss
- class_loss: `girl` 일반 이미지에 대한 noise prediction loss

이 구조 덕분에:
- 캐릭터는 고정되고
- 일반 생성 능력은 유지된다

## 전체 학습 파이프라인

```text
[ Dataset ]
  ├─ instance_images/   (내 캐릭터 이미지)
  └─ class_images/  (girl 일반 이미지)

[ Prompt Template ]
  ├─ instance_prompt = "a photo of <saya_token> girl"
  └─ class_prompt = "a photo of a girl"

[ Training ]
  ├─ pretrained SD load
  ├─ tokenizer: <saya_token> 추가
  ├─ UNet fine-tuning
  ├─ prior preservation loss
  └─ checkpoint 저장

[ Inference ]
  └─ "<saya_token>" 포함 프롬프트로 생성
```

## 실전 코드 구조 설계

DreamBooth는 노트북이 아니라 스크립트 기반으로 관리하는 게 좋다.

### 프로젝트 구조 예시

```text
dreambooth_saya/
├─ data/
│  ├─ instance/
│  │   ├─ saya_01.png
│  │   ├─ saya_02.png
│  │   └─ ...
│  └─ class/
│      ├─ girl_001.png
│      └─ ...
├─ prompts/
│  ├─ instance.txt   # "a photo of <saya_token> girl"
│  └─ class.txt      # "a photo of a girl"
├─ train.py
├─ inference.py
├─ config.yaml
└─ README.md
```

### config.yaml

```python
model:
  pretrained_model: stabilityai/stable-diffusion-2-1-base
  train_unet: true
  train_text_encoder: false

token:
  identifier: "<saya_token>"
  initializer: "girl"

training:
  resolution: 512
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-6
  max_train_steps: 800
  prior_loss_weight: 1.0

dataset:
  instance_dir: data/instance
  class_dir: data/class
```

6. 학습 스크립트 핵심 흐름

```python
# 1. 모델/토크나이저 로드
pipe = StableDiffusionPipeline.from_pretrained(...)
tokenizer.add_tokens("<saya_token>")
text_encoder.resize_token_embeddings(len(tokenizer))

# 2. 데이터 로더
instance_dataset = DreamBoothDataset(
    image_dir="data/instance",
    prompt="a photo of <saya_token> girl"
)
class_dataset = DreamBoothDataset(
    image_dir="data/class",
    prompt="a photo of a girl"
)

# 3. 학습 루프
for step in range(max_steps):
    # instance
    loss_instance = compute_diffusion_loss(instance_batch)
    # class
    loss_class = compute_diffusion_loss(class_batch)

    loss = loss_instance + prior_weight * loss_class
    loss.backward()
    optimizer.step()
```
   
핵심 포인트:

- UNet forward는 일반 diffusion과 동일
- 차이는 “어떤 프롬프트 + 어떤 데이터”로 학습하느냐

7. Inference 파이프라인 (DreamBooth의 결과 확인)

```python
pipe = StableDiffusionPipeline.from_pretrained("dreambooth_saya")
image = pipe(
    prompt="a portrait of <saya_token> girl, anime style",
    guidance_scale=7.5,
    num_inference_steps=30,
).images[0]
```

이 시점에서:

- <saya_token>이 캐릭터 정체성을 담당
- 나머지 프롬프트는 스타일/연출 담당

8. DreamBooth vs LoRA


| 항목     | DreamBooth | LoRA |
| ------ | ---------- | ---- |
| VRAM   | 큼          | 작음   |
| 학습 시간  | 김          | 짧음   |
| 정체성 고정 | 매우 강함      | 중간~강 |
| 다중 캐릭터 | 어려움        | 쉬움   |
| 배포     | 무거움        | 가벼움  |


실전 권장:

- 연구/완전 고정 캐릭터 -> DreamBooth
- 여러 캐릭터/빠른 실험 -> LoRA

9. 캐릭터 AI 파이프라인에의 위치

DreamBooth는 이 위치에 들어간다.

```text
LLM (감정/상태)
   ↓
Prompt Compiler
   ↓
[ DreamBooth SD ]
   ↓
캐릭터 이미지
```

- DreamBooth는 “이미지 생성 엔진의 정체성 레이어”
- 감정/연출은 프롬프트/CFG/시드로 제어

참고자료
Huggingface, Diffusion Course, https://huggingface.co/learn