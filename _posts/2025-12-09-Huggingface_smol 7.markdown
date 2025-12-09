---
layout: post
title:  "허깅페이스 스몰 코스 - Direct Preference Optimization (DPO)"
date:   2025-12-09 00:10:22 +0900
categories: Huggingface_smol
---

# Direct Preference Optimization (DPO)

DPO(Direct Preference Optimization)는 RLHF를 훨씬 단순화한 최신 선호도 학습 기법이다.
기존 RLHF처럼 보상 모델(reward model)과 PPO 정책 최적화 단계를 거치지 않고,
선호도 데이터 자체를 이용해 모델을 직접 최적화한다.

LLM alignment(정렬)를 구현하는 가장 안정적이고 효율적인 방법 중 하나로 평가되며,
Meta LLaMA 등 최신 대형 모델에서도 채택되는 방식이다.

## 왜 DPO인가? - RLHF와 비교

기존 RLHF(RLHF 1.0) 파이프라인

1. SFT 모델 학습
2. Reward Model 학습
3. RL(PPO 등)을 이용해 Policy 모델 업데이트

문제점

- 구성요소가 3개나 필요
- PPO는 불안정하며 튜닝 난이도가 매우 높음
- 훈련 시간/비용이 큼

### DPO의 핵심 아이디어

보상 모델을 만들 필요 없이,
선호도 데이터(chosen vs rejected) 만으로
언어 모델을 직접 업데이트할 수 있다.

즉, 선호된 답변의 확률을 높이고, 덜 선호된 답변의 확률을 낮추는 방향으로 모델을 훈련한다.

이것을 분류(classification) 문제로 재해석한 점이 DPO의 혁신이다.

## DPO가 실제로 어떻게 동작하는가?

DPO는 다음 두 단계로 이루어진다.

### 단계 1 - SFT 모델 만들기

먼저 기존 SFT 방식으로 모델을 instruction-following 모델로 만든다.
이 모델은 reference model로 사용되며, DPO는 이 모델을 기준으로 학습된다.

### 단계 2 — 선호도 학습 (Preference Learning)

데이터 구조는 다음과 같다.


| prompt                      | chosen(선호됨) | rejected(비선호됨) |
| --------------------------- | ----------- | -------------- |
| "Explain quantum computing" | 좋은 답변       | 나쁜 답변          |


학습 목표

- chosen 확률 ↑
- rejected 확률 ↓

이를 위해 DPO는 reference model 대비 현재 모델의 로그확률 비율을 비교하는 방식으로 업데이트한다.

## DPO Loss - 직관적 설명(LaTeX 없이)

원래 수식은 복잡하지만, 핵심 개념은 매우 단순하다.

DPO는 다음 항을 크게 만들도록 모델을 학습한다:

```python
log( πθ(y_w | x) / πref(y_w | x) )  
    -  
log( πθ(y_l | x) / πref(y_l | x) )
```

- winning(선호된) 응답은 reference보다 더 높은 확률을 갖도록 만들고
- losing(비선호된) 응답은 reference보다 더 낮은 확률을 갖도록 만든다.

이 차이를 sigmoid 함수에 넣어 binary classification loss를 계산한 것이 DPO Loss다.

여기서 β(beta) 파라미터는 다음 역할을 한다:

- beta 작음: reference 모델과 더 비슷하게 유지
- beta 큼: 선호도 방향으로 더 강하게 당김 (과적합 위험)

## DPO Dataset Format

최소한 다음 3개 필드가 필요하다:

```python
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

권장 사항

- 명확한 선호도 기준
- 높은 annotator 일치도
- 다양한 프롬프트(특정 문제에만 편향되지 않도록)


## DPO 구현 — TRL 예제

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

training_args = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    max_prompt_length=512,
    max_length=1024,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)

trainer.train()
```

## Hyperparameter 가이드


| 파라미터             | 의미               | 권장값         |
| ---------------- | ---------------- | ----------- |
| **β (beta)**     | preference 신호 강도 | 0.1~0.5     |
| **LR**           | 매우 낮아야 함         | 5e-7 ~ 5e-6 |
| **Batch Size**   | 작아도 됨            | 4~8         |
| **Dataset Size** | 최소 수천 개          | 10k+ 추천     |


DPO는 미세한 업데이트만 필요하므로 LR이 너무 크면 모델이 망가진다.
(= catastrophic forgetting)

## Best Practices

1. 데이터 품질이 가장 중요

- label 명확성
- chosen과 rejected의 차이가 확실해야 함
- 다채로운 prompt 분포

2. beta 조절

- 모델이 reference와 너무 달라졌다면 beta를 줄이기
- alignment가 충분하지 않다면 beta 늘리기

3. 평가 방법

- 동일 prompt에 대해 reference 모델과 비교
- edge-case prompt 포함
- alignment degradation(유틸리티 감소) 여부 반드시 체크

## 흔한 문제와 해결

1. 문제: DPO 학습 뒤 모델이 망가짐

원인: learning rate 너무 큼
해결: 5e-7 이하로 낮추기

2. 문제: 선호도가 전혀 반영되지 않음

원인: beta가 너무 작음
해결: 0.3~0.5로 증가

3. 문제: 특정 스타일에 과하게 치우침

원인: 데이터 다양성 부족
해결: dataset mixing 또는 rejected 다양화

## 전체 흐름 정리

DPO 학습 파이프라인은 다음과 같은 순서를 따른다:

1. Base Model 준비
2. SFT (reference model) 생성
3. Preference Dataset 준비 (prompt + chosen + rejected)
4. DPOTrainer로 직접 preference optimization 수행
5. beta, LR 조절하며 alignment 강화
6. reference 모델과 비교하여 품질 평가
7. 필요 시 LoRA merge 또는 full save

DPO는 RLHF의 복잡도를 제거하면서
더 안정적인 학습 + 더 적은 비용 + 높은 품질의 alignment를 제공하는 최신 표준 방식이다.

참고자료
Huggingface, a smol course, https://huggingface.co/learn