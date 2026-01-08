---
layout: post
title:  "허깅페이스 에이전트 코스 - Using Tools in LlamaIndex"
date:   2025-01-08 00:10:22 +0900
categories: Huggingface_agent
---

# Using Tools in LlamaIndex

## 1. 왜 “Tool 설계”가 중요한가

LlamaIndex에서 Tool은 단순한 함수 래퍼가 아니다.  
Tool은 **Agent가 외부 세계와 상호작용하는 유일한 통로**다.

즉,
- Tool 설계가 명확하면 -> Agent는 정확하게 행동하고
- Tool 인터페이스가 모호하면 -> Agent는 틀린 도구를 선택하거나 잘못 호출한다

이는 인간 개발자에게 API 문서가 중요한 것과 동일하다.

> **Tool = LLM이 호출하는 API**
>  
> 이름 · 설명 · 인자 정의가 곧 “프롬프트의 일부”가 된다

## 2. LlamaIndex의 Tool 4대 분류

LlamaIndex는 Tool을 목적에 따라 네 가지로 분류한다.

### FunctionTool  
### QueryEngineTool  
### ToolSpecs  
### Utility Tools  

이 구분은 **Agent가 무엇을 할 수 있느냐**를 구조적으로 나눈 것이다.

## 3. FunctionTool — “파이썬 함수 -> Tool”

### 개념

`FunctionTool`은 **임의의 Python 함수를 Agent가 호출 가능한 Tool로 변환**한다.

- 동기 / 비동기 함수 모두 가능
- 함수 시그니처를 자동 분석
- docstring은 Agent 판단에 직접 영향

즉, **함수의 이름과 docstring = Agent의 사용 설명서**

### 예제 코드

```python
from llama_index.core.tools import FunctionTool

def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)

tool.call("New York")
```

핵심 포인트

- name: Agent가 “이 도구를 언제 써야 하는지” 판단하는 핵심 단서
- description: 도구 선택 정확도를 좌우
- 인자 타입 힌트: LLM이 argument를 정확히 작성하도록 도움

## 4. QueryEngineTool — “RAG를 Tool로 만들기”

개념

QueryEngine은:
- 검색 + LLM 응답을 묶은 RAG 실행 엔진

이를 Tool로 감싸면:
- Agent가 “필요할 때” RAG를 호출할 수 있다

즉, Agentic RAG의 핵심 연결고리

생성 흐름

1. VectorStoreIndex 생성
2. QueryEngine 생성
3. QueryEngineTool로 래핑

```python
from llama_index.core.tools import QueryEngineTool

tool = QueryEngineTool.from_defaults(
    query_engine,
    name="knowledge_base_search",
    description="Searches the internal knowledge base and answers questions."
)
```

중요한 설계 포인트

- QueryEngineTool은 Agent -> 지식 접근 인터페이스
- 여러 QueryEngine을 만들고 각각 Tool로 제공 가능
- Agent는 “어떤 지식 소스를 쓸지” 스스로 선택

## 5. ToolSpecs — “도구 묶음 (Tool Collection)”

개념
ToolSpec은 서로 연관된 여러 Tool의 패키지다.

예시:

- Gmail ToolSpec
- Google Calendar ToolSpec
- MCP ToolSpec

이는 단일 도구가 아니라 업무 단위 능력 묶음이다.

### Gmail ToolSpec 예제

```python
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tools = tool_spec.to_tool_list()
```

```python
[(tool.metadata.name, tool.metadata.description) for tool in tools]
```

의미
- Agent에게 “메일 처리 능력”을 한 번에 부여
- Tool 수가 많아질수록 ToolSpec이 필수
- Agent 프롬프트가 훨씬 간결해짐

## 6. MCP (Model Context Protocol) ToolSpec

### MCP란?
- 외부 도구 서버를 표준 프로토콜로 연결
- Tool 호출을 HTTP / SSE 기반으로 분리
- Agent <-> Tool 실행 환경 완전 분리

### MCP ToolSpec 사용 흐름

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool_spec = McpToolSpec(client=mcp_client)
```

의미
- Tool 실행을 완전히 외부화
- 보안 / 스케일링 / 언어 분리 용이
- 실제 서비스 환경에 매우 적합

## 7. Utility Tools — “컨텍스트 폭발 방지 장치”
왜 필요한가?
API 응답이나 데이터 로딩 결과는 종종:

- 너무 크고
- 너무 잡다하며
- LLM 컨텍스트를 초과한다

Utility Tool은 이를 지연 로딩 + 부분 검색 구조로 해결한다.

### OnDemandToolLoader
“필요할 때만 데이터를 로드하고 즉시 검색”

- Loader + Index + Query를 한 번의 Tool 호출로 처리
- 대규모 문서에 매우 효과적

### LoadAndSearchToolSpec
“기존 Tool을 Load / Search로 분리”

- Load Tool -> 데이터 수집 & 인덱싱
- Search Tool → 질의 전용

Agent는 상황에 따라:

- 새로 불러올지
- 기존 인덱스를 쓸지

선택할 수 있다.

## 8. 정리 — Agent 설계 관점 요약
Tool은 단순한 기능이 아니다


| 구성요소            | 의미          |
| --------------- | ----------- |
| FunctionTool    | 행동(Action)  |
| QueryEngineTool | 지식 접근(RAG)  |
| ToolSpec        | 능력 묶음       |
| Utility Tool    | 비용/컨텍스트 최적화 |


핵심 원칙
- Tool 이름과 설명은 프롬프트 설계
- QueryEngineTool은 Agentic RAG의 핵심
- ToolSpec은 대규모 Agent 설계의 필수 요소

Utility Tool 없이는 실전 서비스 불가능


참고자료
Huggingface, agents course, https://huggingface.co/learn