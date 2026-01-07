---
layout: post
title:  "허깅페이스 에이전트 코스 - LLM"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# What are LLMs?

앞선 글에서는 **Agent는 반드시 핵심에 AI 모델을 가진다**는 점과,  
그중에서도 **LLM(Large Language Model)** 이 가장 일반적으로 사용된다는 사실을 살펴보았다.

이 글에서는 다음 질문에 답한다.

- LLM이란 정확히 무엇인가?
- LLM은 어떤 구조로 동작하는가?
- 왜 Agent의 “Brain”으로 LLM이 사용되는가?

이 문서는 **Agent 설계 관점에서 필요한 만큼의 기술적 설명**을 제공하는 것을 목표로 한다.  
Transformer 내부 수식이나 미분 수준의 설명은 다루지 않는다.

## 실습 목표

- LLM의 정의와 역할 명확히 이해
- Transformer 구조의 큰 그림 파악
- 토큰, 다음 토큰 예측, EOS 개념 정리
- LLM이 Agent에서 수행하는 책임 범위 인식

## LLM이란 무엇인가?

**LLM(Large Language Model)**이란,

> 대규모 텍스트 데이터로 학습되어  
> 인간 언어를 이해하고 생성하는 데 특화된  
> 신경망 기반 AI 모델

을 의미한다.

LLM은 다음과 같은 특징을 가진다.

- 방대한 텍스트 코퍼스로 학습
- 언어의 패턴, 구조, 맥락을 통계적으로 학습
- 수백만~수십억 개의 파라미터 보유

오늘날 사용되는 대부분의 LLM은  
**Transformer 아키텍처**를 기반으로 한다.

이 구조는 2018년 Google의 **:contentReference[oaicite:0]{index=0} BERT** 공개 이후 사실상 표준이 되었다.

## Transformer 아키텍처 개요

초기의 Transformer는 다음 두 블록으로 구성되어 있었다.

- Encoder
- Decoder

이 조합 방식에 따라 Transformer는 크게 세 가지 유형으로 나뉜다.

## Transformer의 3가지 유형

### 1. Encoder-only 모델

입력을 받아 **의미를 압축한 표현(embedding)** 으로 변환한다.

- 대표 모델: **:contentReference[oaicite:1]{index=1} BERT**
- 주요 용도:
  - 텍스트 분류
  - 의미 검색
  - 개체명 인식(NER)
- 파라미터 규모: 수백만 단위

-> “이 문장이 무엇을 의미하는가?”에 특화

### 2. Decoder-only 모델

이전 토큰들을 바탕으로 **다음 토큰을 하나씩 생성**한다.

- 대표 모델: **:contentReference[oaicite:2]{index=2} LLaMA**
- 주요 용도:
  - 텍스트 생성
  - 챗봇
  - 코드 생성
- 파라미터 규모: 수십억 단위

-> **대부분의 LLM은 이 유형**

### 3. Seq2Seq (Encoder–Decoder)

입력 시퀀스를 인코딩한 뒤,  
그 결과를 바탕으로 **다른 시퀀스를 생성**한다.

- 대표 모델: T5, BART
- 주요 용도:
  - 번역
  - 요약
  - 패러프레이징
- 파라미터 규모: 수백만 단위

## 대표적인 LLM들

실무 및 연구에서 자주 언급되는 LLM은 다음과 같다.


| Model | Provider |
|------|---------|
| **DeepSeek-R1** | DeepSeek |
| **GPT-4** | :contentReference[oaicite:3]{index=3} |
| **LLaMA 3** | :contentReference[oaicite:4]{index=4} |
| **SmolLM2** | :contentReference[oaicite:5]{index=5} |
| **Gemma** | :contentReference[oaicite:6]{index=6} |
| **Mistral** | Mistral |


공통점은 하나다.

> **모두 “다음 토큰 예측”을 수행한다.**

## LLM의 핵심 원리: 다음 토큰 예측

LLM의 목표는 매우 단순하다.

> **이전 토큰들이 주어졌을 때,  
> 다음에 올 토큰의 확률 분포를 예측한다.**

여기서 중요한 개념이 **토큰(token)** 이다.

## 토큰(Token)이란?

토큰은 LLM이 처리하는 최소 단위다.

- 단어 전체가 아닐 수도 있음
- 서브워드 단위로 분해됨

예:

- `interest` + `ing` → `interesting`
- `interest` + `ed` → `interested`

이 방식 덕분에,

- 수십만 개의 단어 대신
- 약 3만 개 내외의 토큰 사전으로

언어를 효율적으로 표현할 수 있다.

## Special Token과 EOS

모든 LLM은 **특수 토큰(Special Token)** 을 사용한다.

이 토큰들은 다음을 구분하기 위해 존재한다.

- 시퀀스 시작 / 종료
- 메시지 경계
- 대화 턴 종료

가장 중요한 토큰은 **EOS (End Of Sequence)** 이다.

> EOS가 생성되면  
> 모델은 “여기서 멈춰도 된다”고 판단한다.

### 모델별 EOS 예시


| Model | Provider | EOS Token | 의미 |
|------|--------|----------|------|
| GPT-4 | OpenAI | `<|endoftext|>` | 메시지 종료 |
| LLaMA 3 | Meta | `<|eot_id|>` | 시퀀스 종료 |
| DeepSeek-R1 | DeepSeek | `<|end_of_sentence|>` | 문장 종료 |
| SmolLM2 | Hugging Face | `<|im_end|>` | 메시지 종료 |
| Gemma | Google | `<end_of_turn>` | 대화 턴 종료 |


> 모든 토큰을 외울 필요는 없다.  
> 다만 **모델마다 구조 토큰이 다르다는 사실**은 반드시 인지해야 한다.

## Autoregressive 모델이란?

LLM은 **Autoregressive** 모델이다.

즉,

- 한 번에 문장을 생성하지 않는다
- 토큰을 하나 생성할 때마다
- 그 결과를 다시 입력으로 사용한다

이 과정은 다음과 같다.

1. 입력 텍스트 토큰화
2. 다음 토큰 확률 분포 계산
3. 토큰 선택
4. 선택된 토큰을 입력에 추가
5. EOS가 나올 때까지 반복

## 디코딩(Decoding) 전략

다음 토큰을 고르는 방법은 여러 가지가 있다.

- **Greedy decoding**  
  -> 가장 확률이 높은 토큰 선택
- **Beam search**  
  -> 여러 후보 시퀀스를 동시에 탐색
- **Sampling (temperature, top-p)**  
  -> 다양성과 창의성 조절

Agent 설계에서는  
**출력의 안정성 vs 다양성**을 어떻게 조절할지가 중요해진다.

## Attention: 왜 중요한가?

Transformer의 핵심은 **Attention 메커니즘**이다.

문장 예시:

> “The capital of France is …”

다음 토큰을 예측할 때 중요한 단어는

- France
- capital

이지, the, of 같은 단어가 아니다.

Attention은“지금 예측에 어떤 토큰이 중요한가”를 학습적으로 계산한다.

## Context Length와 Attention Span

LLM에는 **컨텍스트 길이 제한**이 있다.

- 한 번에 볼 수 있는 최대 토큰 수
- 이 범위를 넘으면 과거 정보를 “잊음”

Agent 설계 시:

- 메모리 전략
- 요약
- 외부 저장(RAG)

이 필요한 이유가 바로 여기서 나온다.

## 프롬프트가 중요한 이유

LLM의 유일한 임무는 다음이다.

> **입력된 모든 토큰을 보고  
> 다음 토큰을 예측하는 것**

따라서 입력 문장, 즉 **프롬프트**의 구조와 표현은  
출력 결과에 직접적인 영향을 미친다.

프롬프트 엔지니어링이 중요한 이유다.

## LLM은 어떻게 학습되는가?

### 1. 사전학습 (Pre-training)

- 대규모 텍스트
- 자기지도학습
- 다음 토큰 예측

이 단계에서 모델은

- 문법
- 의미 구조
- 일반 상식

을 학습한다.

### 2. 파인튜닝 (Fine-tuning)

이후 목적에 맞게 추가 학습한다.

- 대화 구조
- 지시 따르기
- 도구 사용
- 코드 생성

Agent에서 사용하는 LLM은  
대부분 **Instruction-tuned** 모델이다.

## LLM은 어떻게 사용되는가?

### 1. 로컬 실행

- 충분한 GPU 자원 필요
- 완전한 제어 가능

### 2. API / 클라우드 사용

- 빠른 시작
- 확장성
- 비용 발생

이 글 이후의 실습에서는  
주로 로컬 LLM을 사용한다.

## Agent에서 LLM의 역할

LLM은 Agent에서 다음 역할을 담당한다.

- 사용자 명령 해석
- 맥락 유지
- 계획 수립
- Tool 선택

즉,

> **LLM은 Agent의 Brain이다.**

하지만 다시 강조하면,

- LLM은 행동하지 않는다
- 행동은 Tool이 수행한다

## 정리

이 글에서 다룬 핵심은 다음이다.

- LLM은 “다음 토큰 예측기”다
- Transformer + Attention이 핵심 구조다
- EOS와 Special Token이 생성의 경계를 만든다
- LLM은 Agent의 Brain 역할을 담당한다

이제 다음 단계는 **Conversational Formatting & Tool Calling** 으로 이어진다.


참고자료
Huggingface, agents course, https://huggingface.co/learn