---
layout: post
title:  "허깅페이스 에이전트 코스 - Function-Calling을 위한 모델 파인튜닝"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# Function-Calling을 위한 모델 파인튜닝 정리

이 글에서는 **Function-Calling이 가능한 모델을 어떻게 학습시키는지**를 정리한다.  
핵심은 “프롬프트 요령”이 아니라, **모델이 도구 호출 형식을 직접 학습하도록 만드는 것**이다.

## 1. Function-Calling 모델은 어떻게 만들어지는가?

한 문장으로 요약하면:

> **데이터로 학습시킨다.**

LLM이 Function-Calling을 할 수 있으려면,  
“도구를 호출하는 출력 형식” 자체가 **훈련 데이터에 포함**되어 있어야 한다.

## 2. 모델 학습의 3단계 구조

대부분의 LLM은 아래 3단계를 거쳐 만들어진다.

### Pre-training (사전학습)

- 대규모 텍스트로 **다음 토큰 예측**만 학습
- 명령 이해, 대화 능력 없음
- 예시:
  - `google/gemma-2-2b`

특징:
- “문장을 이어 쓰는 능력”만 있음
- Agent, Tool, Function 개념 없음

### Instruction Fine-tuning (지시문 학습)

- “질문-답변” 구조 학습
- 채팅, 명령 수행 능력 획득
- 예시:
  - `google/gemma-2-2b-it`

특징:
- 대화 가능
- 기본적인 Agent 역할 가능
- 하지만 **Function-Calling은 아직 아님**

### Alignment (정렬 / 선호도 학습)

- 무례한 답변 방지
- 정책·톤·행동 제약
- 기업용 챗봇, 고객센터 모델 등

Hugging Face에 공개된 모델들은  
이 중 **1~2단계까지만** 완료된 경우가 대부분이다.

## 3. 왜 Instruction-tuned 모델에서 시작하는가?

이번 실습에서는:

- 사용 안 할 것: `google/gemma-2-2b` (base)
- 사용할 것: `google/gemma-2-2b-it` (instruction-tuned)

를 기반으로 Function-Calling을 학습한다.

이유는 명확하다.

- Base 모델에서 시작하면:
  - instruction following
  - chat format
  - function calling  
  **전부 새로 학습해야 함**
- Instruction-tuned 모델에서 시작하면:
  - 이미 대화/지시 수행 가능
  - **Function-Calling만 추가로 학습하면 됨**

즉,
> **학습해야 할 정보량을 최소화**하기 위함이다.

## 4. LoRA란 무엇인가?

**LoRA (Low-Rank Adaptation)**는  
대규모 모델을 “통째로 다시 학습”하지 않고,  
**아주 작은 추가 가중치만 학습**하는 기법이다.

### 4.1 LoRA의 핵심 아이디어

- 기존 모델 가중치: Freeze
- 일부 선형 레이어에:
  - 저차원(rank) 행렬 쌍을 추가
- 학습 시:
  - **이 작은 어댑터만 업데이트**

결과:
- 학습 파라미터 수 ↓
- VRAM 요구량 ↓
- 학습 속도 ↑

### 4.2 LoRA의 장점

- 수백 MB 수준의 가중치
- Colab / 개인 GPU에서도 학습 가능
- 추론 시:
  - base + adapter 함께 사용
  - 또는 merge -> 추가 지연 없음

**Function-Calling 같은 “특정 행동 학습”에 최적**

## 5. Function-Calling 파인튜닝의 본질

Function-Calling 파인튜닝이란:

> “도구를 호출하는 출력 형식”을  
> **정답으로 포함한 데이터로 학습**시키는 것

즉, 모델은 다음을 배우게 된다.

- 언제:
  - 일반 답변을 할지
  - 도구 호출을 할지
- 어떻게:
  - 함수 이름
  - 인자(JSON)
- 를 **정확한 포맷으로 출력하는 법**

이제 도구 사용은 프롬프트 트릭이 아닌 추론 결과의 일부이다.


참고자료
Huggingface, agents course, https://huggingface.co/learn