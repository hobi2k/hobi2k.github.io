---
layout: post
title:  "허깅페이스 에이전트 코스 - Using Agents in LlamaIndex"
date:   2025-01-08 00:10:22 +0900
categories: Huggingface_agent
---

# Using Agents in LlamaIndex

## 1. LlamaIndex에서 말하는 “Agent”란 무엇인가

이전에 정의한 Agent 개념을 다시 정리하면 다음과 같다.

> **Agent란**
> LLM을 중심으로  
> - 추론(Reasoning)
> - 계획(Planning)
> - 행동(Action, Tool 호출)
> 을 결합하여 **사용자가 정의한 목표를 달성하는 시스템**이다.

LlamaIndex의 Agent는 단순한 “질문-응답”이 아니라  
**도구를 선택하고, 순서를 결정하고, 필요하면 다른 Agent에게 위임하는 실행 주체**다.


## 2. LlamaIndex의 Agent 유형 (3가지)

LlamaIndex는 내부 추론 방식에 따라 **세 가지 Agent 유형**을 지원한다.

### Function Calling Agents
- OpenAI / HF 등 **function calling API를 지원하는 모델** 전용
- Tool 스키마(JSON Schema)를 기반으로
- LLM이 직접 함수 호출을 생성

장점  
- 매우 정확한 Tool 호출  
- 프롬프트 엔지니어링 부담 감소  

단점  
- function calling을 지원하는 모델에서만 가능

### ReAct Agents
- 모든 chat / text LLM에서 동작
- **Reason -> Action -> Observation** 루프를 프롬프트로 구현
- 추론 과정이 비교적 명시적으로 드러남

장점  
- 모델 제약 없음  
- 복잡한 추론 흐름에 강함  

단점  
- 토큰 소모가 많음  
- 프롬프트 의존성 큼

### Advanced Custom Agents
- BaseWorkflowAgent 기반
- 비동기, 이벤트 기반, 복합 플로우 설계
- 실서비스/대규모 시스템용

## 3. Agent 초기화: AgentWorkflow

LlamaIndex에서 Agent는 **AgentWorkflow**를 통해 생성된다.

### 핵심 개념

- AgentWorkflow는:
  - Tool 목록
  - LLM
  - (선택적으로) system prompt
  를 받아 **적절한 Agent 타입(Function/ReAct)을 자동 선택**

### 기본 예제: FunctionTool 기반 Agent

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
)

agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)
```

내부 동작

- LLM이 function calling API를 지원하면 Function Calling Agent
- 아니면 ReAct Agent
- 개발자는 Agent 유형을 직접 고르지 않아도 된다

## 4. Agent의 상태(State)와 Context

기본 원칙

- Agent는 기본적으로 stateless
- 즉, 각 run() 호출은 독립적

하지만 실전에서는

- 대화형 Agent
- 작업 추적 Agent
- 장기 플래닝 Agent

가 필요하다.

이를 위해 Context 객체를 사용한다.

### Context 사용 예제

```python
from llama_index.core.workflow import Context

ctx = Context(agent)

await agent.run("My name is Bob.", ctx=ctx)
await agent.run("What was my name again?", ctx=ctx)
```

의미

- Context = Agent의 단기 메모리
- 동일 Context를 넘기면:
    - 이전 Tool 호출 결과
    - 이전 메시지
    - 내부 상태 유지 가능

5. RAG Agent (Agentic RAG)

기존 RAG vs Agentic RAG


| 구분   | 기존 RAG | Agentic RAG    |
| ---- | ------ | -------------- |
| 검색   | 자동 1회  | Agent가 필요 시 결정 |
| 전략   | 고정     | 동적             |
| Tool | 숨겨짐    | 명시적            |
| 확장성  | 낮음     | 매우 높음          |


### QueryEngineTool을 Agent에 연결

QueryEngine을 Tool로 감싸면,
Agent는 “지식 검색”을 하나의 선택 가능한 행동으로 인식한다.

```python
from llama_index.core.tools import QueryEngineTool

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3
)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="persona_database",
    description="Searches persona descriptions stored in the database",
    return_direct=False,
)

query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant with access to persona data."
)
```

핵심 포인트

- Tool 이름 / 설명이 Agent 행동 결정에 직접 영향
- Agent는:
    - RAG를 쓸지
    - 다른 Tool을 쓸지
    - 바로 답할지 스스로 판단

## 6. Multi-Agent 시스템 in LlamaIndex

### 왜 Multi-Agent인가?
하나의 Agent에게:

- 계산
- 검색
- 문서 이해
- 플래닝

모두 맡기면 성능이 떨어진다.

역할 분리를 통해:

- 정확도 상승
- 토큰 비용 하락
- 디버깅 용이

### Multi-Agent 구조 개념

각 Agent는:

- 이름(name)
- 역할(description)
- 전용 Tool

을 가진다.

AgentWorkflow가 발언권(Active Speaker)을 관리한다.

### 예제: 계산 Agent + 조회 Agent

```python
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

calculator_agent = ReActAgent(
    name="calculator",
    description="Performs arithmetic operations",
    system_prompt="Use tools for math.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use RAG to answer questions about XYZ.",
    tools=[query_engine_tool],
    llm=llm,
)

agent = AgentWorkflow(
    agents=[calculator_agent, query_agent],
    root_agent="calculator"
)

response = await agent.run("Can you add 5 and 3?")
```

중요한 설계 포인트

- Agent는 다른 Agent를 Tool처럼 사용 가능
- root_agent는 초기 발화 담당
- 필요 시 Agent 간 hand-off 발생

## 7. DL / 시스템 설계 관점 요약

### LlamaIndex Agent의 본질
- Agent = LLM + Tool Router + Planner
- QueryEngine은 지식 접근 모듈
- Context는 단기 메모리
- Multi-Agent는 역할 분리 아키텍처

### 실전 설계 원칙

1. Tool 수가 늘어나면 -> Agent 분리
2. RAG는 반드시 Tool로 노출
3. 상태가 필요하면 Context 사용
4. 고비용 Tool은 Utility Tool / 분리 Agent로 관리
5. 최종적으로는 Workflow 기반 구조로 확장

참고자료
Huggingface, agents course, https://huggingface.co/learn