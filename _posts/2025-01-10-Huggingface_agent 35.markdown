---
layout: post
title:  "허깅페이스 에이전트 코스 - 실습 1"
date:   2025-01-10 00:10:22 +0900
categories: Huggingface_agent
---

# What is GAIA?

**GAIA**는 현실 세계에서의 복합적인 문제 해결 능력을 기준으로  
**AI 어시스턴트의 “범용성”을 평가하기 위해 설계된 벤치마크**다.

공식 명칭은  
**_GAIA: A Benchmark for General AI Assistants_** 이며,  
단일 언어 모델의 지식량이 아니라 **에이전트로서의 실제 수행 능력**을 측정하는 데 초점을 둔다.

## GAIA의 핵심 목적

GAIA는 다음 질문에 답하기 위해 만들어졌다.

> “AI는 실제 인간이 수행하는 **현실적 작업(real-world task)**을  
> 얼마나 잘 계획하고, 조사하고, 도구를 활용해 해결할 수 있는가?”

이를 위해 GAIA는  
단순한 QA나 추론 문제가 아닌 **복합 태스크 기반 문제**를 제시한다.

## 성능 격차가 말해주는 것

GAIA의 문제들은 인간에게는 비교적 단순하지만,  
현재 AI 시스템에게는 매우 어렵다.

| 시스템 | 성공률 |
|---|---|
| 인간 | 약 **92%** |
| GPT-4 + 플러그인 | 약 **15%** |
| OpenAI Deep Research | **67.36% (검증 세트)** |

이 격차는 **“단일 LLM ≠ 범용 AI 어시스턴트”** 라는 사실을 명확히 보여준다.

## GAIA의 설계 철학 (Core Principles)

GAIA는 다음 네 가지 원칙을 중심으로 설계되었다.

### 1. Real-world Difficulty
- 다단계 추론
- 멀티모달 이해 (텍스트 + 이미지)
- 웹 탐색 및 도구 활용이 필수

### 2. Human Interpretability
- 문제 자체는 인간에게 직관적
- “문제 이해 난이도”보다 “수행 난이도”가 핵심

### 3. Non-gameability
- 요령이나 패턴 암기로 해결 불가
- **전체 작업을 실제로 수행해야 정답 도달**

### 4. Simplicity of Evaluation
- 정답은 짧고 명확
- 자동 평가에 적합한 구조

## 난이도 레벨 구조

GAIA는 문제를 **3단계 난이도**로 구분한다.

### Level 1
- 5단계 미만
- 최소한의 도구 사용
- 단순 검색 + 추론 중심

### Level 2
- 5~10단계
- 여러 도구의 조합 필요
- 중간 수준의 계획 수립 요구

### Level 3
- 장기 계획 수립
- 멀티모달 + 멀티툴 통합
- 고난도 에이전트 아키텍처 요구

## GAIA 고난도 문제 예시

> *“2008년 회화 ‘Embroidery from Uzbekistan’에 등장하는 과일 중,  
> 1949년 10월 아침 메뉴로 제공되었고  
> 이후 영화 ‘The Last Voyage’의 부유 소품으로 사용된  
> 해양 여객선에서 제공된 과일은 무엇인가?”*

이 문제는 다음 능력을 동시에 요구한다.

- 멀티모달 분석 (회화 이미지)
- 다단계 사실 연결
- 웹/문헌 탐색
- 정확한 순서 지정
- 엄격한 출력 포맷 준수

- 단순 LLM로는 사실상 해결 불가  
- **Agentic 시스템만이 접근 가능한 문제 유형**

## GAIA가 중요한 이유

GAIA는 단순히 “어렵다”는 것이 목적이 아니다.
GAIA는 다음을 명확히 보여준다.

- 범용 AI 어시스턴트는
  - 추론
  - 검색
  - 도구 실행
  - 계획
  - 상태 유지
  를 통합적으로 수행해야 한다
- Agent 기반 시스템이 필수적이다
- LangGraph, Agentic RAG, Tool-Driven Agent의 필요성을 정량적으로 증명한다

## 라이브 평가와 리더보드

GAIA는 **Hugging Face에서 공개 리더보드**를 운영한다.

- 테스트 문제 수: **300문항**
- 실제 모델 및 에이전트 비교 가능

https://huggingface.co/spaces/gaia-benchmark/leaderboard


## 정리

> **GAIA는 “지식 많은 LLM”이 아니라  
> “현실을 처리할 수 있는 AI 에이전트”를 평가하는 벤치마크다.**

Agentic RAG, LangGraph, Tool-Driven Agent를 배우는 이유가  
GAIA에서 명확해진다.


참고자료
Huggingface, agents course, https://huggingface.co/learn