---
layout: post
title:  "허깅페이스 에이전트 코스 - Introduction to smolagents"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Introduction to smolagents

이 모듈에서는 Hugging Face의 경량 Agent 프레임워크인 **smolagents**를 사용해  
**실전에서 활용 가능한 AI Agent를 설계·구현하는 방법**을 다룬다.

앞선 글들에서 Agent의 개념(Thought–Action–Observation), Tool, Workflow를 이론적으로 이해했다면,  
이번 글에서는 그 개념을 **실제 프레임워크 위에서 구현하는 단계**로 넘어간다.

## 이 모듈의 목적

이 모듈의 핵심 목표는 다음과 같다.

- smolagents의 구조와 철학 이해
- 어떤 문제에 smolagents가 적합한지 판단할 수 있게 되기
- 다양한 Agent 유형(CodeAgent, ToolCallingAgent, RetrievalAgent 등) 이해
- 단일 Agent를 넘어 **멀티 Agent 시스템**으로 확장하는 사고방식 습득
- Vision, Web Browsing 등 고급 Agent 활용 가능성 파악

즉, “smolagents를 쓰는 법”이 아니라  
**“Agent 시스템을 설계하는 법”을 프레임워크 관점에서 이해하는 것**이 목적이다.

## smolagents란 무엇인가?

**smolagents**는 Hugging Face에서 개발한 **경량 Agent 프레임워크**다.
이 라이브러리는 다음과 같은 특징을 가진다.

- Agent 개발에 필요한 반복적·위험한 작업을 자동화
  - Stop & Parse
  - Tool 실행
  - Observation 주입
  - 반복 루프 관리
- Agent 설계의 핵심인
  - Tool 정의
  - System Prompt 설계
  - Agent 조합
  에 집중할 수 있도록 지원

smolagents는 “모든 것을 다 해주는 프레임워크”가 아니라,  
**Agent의 본질적인 구조를 최대한 단순하게 드러내는 프레임워크**라는 점이 중요하다.

## 프레임워크 선택 관점에서의 smolagents

Agent 프레임워크는 매우 다양하다.

- LangGraph
- LlamaIndex
- 기타 LangChain 기반 도구들

smolagents는 그중에서도 다음 상황에 특히 잘 맞는다.

- Agent의 내부 동작을 **명확히 이해하고 싶을 때**
- Code 기반 Action이 중요한 경우
- 단일 Agent → 멀티 Agent로 점진적 확장을 원할 때
- 복잡한 추상화보다 **직관적인 제어 흐름**이 필요한 경우

이 모듈에서는 smolagents의 장점뿐 아니라,  
**언제 다른 프레임워크가 더 적합한지도 함께 비교**한다.

---

## 이 Unit의 스토리라인

Unit 1에서 등장했던 Agent **Alfred**가 다시 등장한다.

이번에는 Alfred가:

- smolagents 프레임워크를 내부 엔진으로 사용하며
- 다양한 실제 업무를 수행한다

설정은 다음과 같다.

- Wayne 가족이 자리를 비운 사이
- Alfred는 Wayne Manor에서 파티를 준비해야 한다
- 검색, 일정 관리, 정보 수집, 의사결정 등 다양한 작업을 처리한다

이 스토리를 통해 추상적인 Agent 개념을  
**구체적인 사용 사례로 연결**한다.

---

## 이 Unit에서 다루는 주요 내용

### 1. Why Use smolagents

- smolagents의 장점과 한계
- LangGraph, LlamaIndex 등과의 비교
- 어떤 프로젝트에 smolagents가 적합한지 판단 기준

---

### 2. CodeAgents

- smolagents의 핵심 Agent 타입
- Action을 JSON이 아닌 **Python 코드로 생성**
- 코드 실행 결과를 Observation으로 활용
- 소프트웨어 개발, 자동화 작업에 특히 적합

---

### 3. ToolCallingAgents

- JSON/Text 기반 Action을 사용하는 Agent
- CodeAgent와의 구조적 차이
- Function-calling 스타일 Agent 설계 방식

---

### 4. Tools

- Tool의 구조와 역할
- `Tool` 클래스 vs `@tool` 데코레이터
- 기본 제공 ToolBox
- 커뮤니티 Tool 공유 및 로드 방법

---

### 5. Retrieval Agents

- 외부 지식에 접근하는 Agent
- Vector Store 기반 검색
- Retrieval-Augmented Generation(RAG) 패턴
- 웹 검색 + 자체 지식베이스 통합
- 실패 대비 fallback 전략

---

### 6. Multi-Agent Systems

- 여러 Agent를 조합하는 설계 방식
- 역할 분리(검색 Agent, 실행 Agent 등)
- Agent 간 협업과 오케스트레이션
- 복잡한 문제를 분산 처리하는 전략

---

### 7. Vision and Browser Agents

- Vision-Language Model(VLM)을 활용한 Agent
- 이미지 기반 추론
- 웹 브라우저를 실제로 탐색하는 Agent
- 멀티모달 Agent 설계 기초

---

## 이 Unit을 마치면 얻게 되는 것

이 Unit을 끝내면 다음이 가능해진다.

- smolagents 기반 Agent를 직접 설계·구현
- Tool 중심으로 Agent 능력 확장
- 단일 Agent를 멀티 Agent 시스템으로 확장
- Vision, Retrieval, Web browsing을 포함한 고급 Agent 설계
- 프레임워크에 휘둘리지 않고 **Agent 구조 자체를 설계하는 관점** 확보

---

## 참고 자료

- smolagents 공식 문서  
  https://huggingface.co/docs/smolagents

- Building Effective Agents (Anthropic)  
  https://www.anthropic.com/research/building-effective-agents

- Agent 설계 가이드라인  
  https://huggingface.co/docs/smolagents/tutorials/building_good_agents

- LangGraph Agents  
  https://langchain-ai.github.io/langgraph/

- Function Calling 가이드  
  https://platform.openai.com/docs/guides/function-calling

- RAG Best Practices  
  https://www.pinecone.io/learn/retrieval-augmented-generation/

참고자료
Huggingface, agents course, https://huggingface.co/learn