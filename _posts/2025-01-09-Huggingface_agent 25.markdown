---
layout: post
title:  "허깅페이스 에이전트 코스 - LangGraph 소개: LLM 워크플로우를 그래프로 설계하기"
date:   2025-01-9 00:10:22 +0900
categories: Huggingface_agent
---

# LangGraph 소개: LLM 워크플로우를 그래프로 설계하기

## 1. LangGraph란 무엇인가?

LangGraph는 LLM 기반 애플리케이션을 **그래프(Graph) 구조**로 설계하고 실행할 수 있도록 돕는 프레임워크이다.  
기존의 단순한 체인(chain)이나 단일 에이전트 구조를 넘어, **상태(State)와 흐름(Flow)을 명시적으로 제어**하는 데 목적이 있다.

LangGraph의 핵심 철학은 다음과 같다.

- LLM 애플리케이션을 상태 머신(Finite State Machine)처럼 다룬다
- 각 단계의 역할과 책임을 코드 수준에서 명확히 정의한다
- 복잡한 분기, 반복, 중단, 재개가 가능한 프로덕션 친화적 구조를 제공한다

## 2. LangGraph는 언제 사용하는가?

LangGraph는 다음과 같은 상황에서 특히 적합하다.

- 멀티 스텝 추론, 멀티 에이전트 협업이 필요한 경우
- 단일 프롬프트 → 단일 응답 구조로는 표현이 어려운 로직
- 조건 분기(if/else), 반복(loop), 정책 기반 흐름 제어가 필요한 경우
- 실험 단계를 넘어 운영 환경(Production)에서 안정적으로 동작해야 하는 시스템

즉, LangGraph는  
**“LLM을 호출하는 코드”가 아니라  
“LLM을 포함한 시스템을 설계하는 단계”** 에서 사용된다.

## 3. 모듈 구성 개요

이 단원 LangGraph의 개념부터 실습까지 다음과 같은 흐름으로 구성된다.

### 3.1 What is LangGraph, and when to use it?
- LangGraph의 등장 배경
- 기존 체인/에이전트 방식의 한계
- 그래프 기반 설계가 필요한 이유

### 3.2 Building Blocks of LangGraph
- **State**: 에이전트가 공유·누적하는 상태 데이터
- **Node**: 특정 작업을 수행하는 실행 단위
- **Edge**: 노드 간 전이 규칙
- **Conditional Routing**: 상태 값에 따른 분기 처리

### 3.3 Alfred, the mail sorting butler
- 간단한 그래프 기반 에이전트 예제
- 입력된 메일을 분석하고 분류
- LangGraph의 기본 사용 패턴 학습

### 3.4 Alfred, the document analyst agent
- 문서 분석이라는 실제 업무 시나리오
- 멀티 단계 판단과 상태 유지
- 단순 데모를 넘어선 실전형 에이전트 구조

## 4. 모델 요구 사항 및 주의점
- LangGraph 자체는 모델 독립적이지만,
  - 안정적인 분기 판단
  - 구조화된 출력
  - 함수 호출 및 툴 사용
  을 위해 상위급 모델이 사실상 필요

로컬 LLM 또는 소형 모델을 사용할 경우,  
일부 예제는 성능 저하 또는 비결정적 동작이 발생할 수 있다.

## 한 줄 요약

> LangGraph는 LLM을 “프롬프트”가 아니라  
> “상태와 흐름을 가진 시스템 컴포넌트”로 다루기 위한 프레임워크다.

참고자료
Huggingface, agents course, https://huggingface.co/learn