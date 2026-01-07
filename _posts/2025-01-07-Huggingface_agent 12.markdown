---
layout: post
title:  "허깅페이스 에이전트 코스 - Why use smolagents"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Why use smolagents

이 글에서는 **왜 smolagents를 사용하는지**, 그리고 **어떤 경우에 적합한 프레임워크인지**를 명확히 정리한다.  
목표는 smolagents가 “좋다/나쁘다”가 아니라, **언제 선택해야 하는 도구인지 판단 기준을 갖는 것**이다.

## smolagents란 무엇인가

**smolagents**는 LLM에 *행위 능력(agency)*을 부여해  
현실 세계(검색, 이미지 생성, 코드 실행 등)와 상호작용하도록 만드는 **경량 Agent 프레임워크**다.

이전 글에서 다룬 Agent의 핵심 구조인  
**Thought → Action → Observation** 사이클을 매우 직관적인 방식으로 구현한다.

- Thought: LLM의 추론
- Action: Tool 호출 또는 코드 실행
- Observation: 실행 결과를 다시 컨텍스트로 주입

smolagents는 이 흐름을 **과도한 추상화 없이 그대로 드러내는 것**을 목표로 한다.

## smolagents의 핵심 장점

### 1. 단순성 (Simplicity)

- 불필요한 추상화 최소화
- 코드 구조가 직관적
- Agent 내부 동작을 추적하기 쉬움

-> 학습용, 실험용, 구조 이해에 매우 유리

### 2. LLM 선택의 유연성

- Hugging Face Hub 모델
- Serverless Inference
- OpenAI API 호환 서버
- Azure OpenAI
- LiteLLM 기반 모델

-> 특정 벤더나 모델에 종속되지 않음

### 3. Code-First Agent 설계

smolagents의 가장 큰 특징은 **CodeAgent 중심 설계**다.

- Action을 JSON으로 기술하지 않고
- **Python 코드 자체를 Action으로 생성**
- 파싱 과정이 필요 없음
- 생성된 코드를 바로 실행 가능

이는 다음과 같은 장점을 만든다.

- Stop & Parse 오류 감소
- 복잡한 로직(조건문, 반복문)을 자연스럽게 표현
- Tool 호출을 코드 레벨에서 직접 제어

### 4. Hugging Face Hub와의 강력한 통합

- Gradio Space를 Tool처럼 사용 가능
- 커뮤니티 Tool 재사용
- Tool 공유 및 배포가 쉬움

-> 실험 결과를 바로 데모·공유 가능

## 언제 smolagents를 쓰는가

smolagents는 다음과 같은 상황에 특히 적합하다.

- 가볍고 빠른 Agent 실험이 필요할 때
- 복잡한 설정 없이 바로 Agent를 만들고 싶을 때
- Action 로직이 비교적 단순한 경우
- Code 기반 실행이 중요한 경우
- Agent 내부 동작을 명확히 이해하고 싶은 경우

반대로, 매우 복잡한 상태 머신이나  
대규모 워크플로 오케스트레이션이 필요하다면  
LangGraph 같은 프레임워크가 더 적합할 수 있다.

## Code Action vs JSON Action

기존 Agent 프레임워크 다수는 Action을 JSON으로 표현한다.

### JSON Action 방식
- LLM -> JSON 출력
- 파서가 JSON 해석
- Tool 호출 코드 생성
- 실행

### Code Action 방식 (smolagents)
- LLM -> Python 코드 출력
- 코드 직접 실행
- 결과를 Observation으로 사용

**핵심 차이점**

- JSON: 안전하지만 파싱과 규칙 관리가 필요
- Code: 유연하고 강력하지만 실행 보안 고려 필요

smolagents는 이 중 **Code Action을 1급 시민**으로 취급한다.

## smolagents의 Agent 타입

smolagents의 Agent는 기본적으로 **멀티스텝 Agent**다.

각 스텝은 다음을 수행한다.

- 하나의 Thought
- 하나의 Action
- 하나의 Observation

### 주요 Agent 타입

1. **CodeAgent**
   - Python 코드로 Action 생성
   - smolagents의 핵심 Agent 타입

2. **ToolCallingAgent**
   - JSON/Text 기반 Tool 호출
   - 기존 Function-calling 패턴과 유사

상황에 따라 두 방식을 혼합하거나 선택할 수 있다.

## Tool 정의 방식

smolagents에서 Tool은 다음 두 방식으로 정의한다.

- `@tool` 데코레이터로 Python 함수 감싸기
- `Tool` 클래스로 명시적 정의

공통 조건:
- 타입 힌트 필수
- docstring에 인자 설명 포함

이는 Tool 자동 문서화 및 LLM 이해를 위해 중요하다.

## 모델 통합 방식

smolagents는 다양한 모델 연결 클래스를 제공한다.

- **TransformersModel**
  - 로컬 transformers 파이프라인

- **InferenceClientModel**
  - Hugging Face Serverless Inference
  - 서드파티 Inference Provider 지원

- **LiteLLMModel**
  - LiteLLM 기반 경량 연결

- **OpenAIServerModel**
  - OpenAI API 호환 서버 연결

- **AzureOpenAIServerModel**
  - Azure OpenAI 연동

모델 선택과 실험이 매우 자유롭다.

## 정리

smolagents는 다음 철학을 가진 프레임워크다.

- Agent의 본질(Thought–Action–Observation)을 숨기지 않는다
- Code를 중심으로 Action을 설계한다
- 빠른 실험과 이해를 최우선으로 한다

따라서 smolagents는  
**Agent를 처음 제대로 이해하고, 직접 설계해보고 싶은 사람**에게  
매우 적합한 선택지다.

참고자료
Huggingface, agents course, https://huggingface.co/learn