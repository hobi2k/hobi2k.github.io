---
layout: post
title:  "허깅페이스 에이전트 코스 - Creating Your Gala Agent"
date:   2025-01-10 00:10:22 +0900
categories: Huggingface_agent
---

# Creating Your Gala Agent

이 글에서는 지금까지 구축한 모든 구성 요소를 **하나의 완성된 에이전트 Alfred**로 통합한다.  
Alfred는 이제 단순한 QA 봇이 아니라, **복수의 도구를 상황에 맞게 활용하는 Agentic RAG 기반 갈라 호스트**가 된다.

## 목표 요약

최종적으로 Alfred는 다음 기능을 모두 수행할 수 있어야 한다.

1. **게스트 정보 조회** (RAG 기반)
2. **웹 검색** (최신 정보·뉴스)
3. **날씨 확인** (불꽃놀이 일정 판단)
4. **Hugging Face Hub 통계 조회** (AI 연구자와의 대화)
5. **멀티툴 조합 추론** (여러 도구를 연속/병렬 사용)
6. **대화 맥락 유지 (메모리)**

## 1. Alfred 조립하기: 전체 에이전트 구성

이미 앞선 섹션에서 각 도구를 `tools.py`, `retriever.py`로 분리해 두었으므로,  
여기서는 **재구현 없이 import 후 조립**하는 것이 핵심이다.

### 1.1 smolagents 기반 Alfred

```python
# 필수 라이브러리
from smolagents import CodeAgent, InferenceClientModel

# 커스텀 도구 import
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset

# 모델 초기화
model = InferenceClientModel()

# 도구 초기화
search_tool = DuckDuckGoSearchTool()
weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()
guest_info_tool = load_guest_dataset()

# Alfred 생성
alfred = CodeAgent(
    tools=[
        guest_info_tool,
        weather_info_tool,
        hub_stats_tool,
        search_tool
    ],
    model=model,
    add_base_tools=True,   # 기본 내장 도구 추가
    planning_interval=3    # 3 step마다 planning 수행
)
```

**핵심 포인트**
- `planning_interval` -> 에이전트가 주기적으로 계획을 재수립
- 모든 도구는 **동등한 선택지**로 제공됨

### 1.2 LlamaIndex 기반 Alfred

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

# 모델 초기화
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
)

# Alfred 생성
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)
```

**특징**
- Tool 중심 구성
- Context 객체를 통해 명시적 메모리 관리

### 1.3 LangGraph 기반 Alfred (Agentic RAG 정석)

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

# 검색 도구
search_tool = DuckDuckGoSearchRun()

# LLM + Chat
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])]
    }

# 그래프 구성
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

alfred = builder.compile()
```

**핵심 구조**
- `assistant ↔ tools` 루프
- LLM이 **스스로 도구 사용 여부를 결정**
- Agentic RAG의 대표적 구현 패턴

## 2. Alfred 활용 예시 (End-to-End)

### 2.1 게스트 정보 조회

```python
response = alfred.run("Tell me about Lady Ada Lovelace.")
print(response)
```

Alfred는 자동으로:
- RAG 도구 사용
- 게스트 문서 검색
- 요약 응답 생성

### 2.2 불꽃놀이용 날씨 판단

```python
response = alfred.run(
    "What's the weather like in Paris tonight? Will it be suitable for fireworks?"
)
print(response)
```

- 날씨 도구 호출
- 조건 해석
- 행사 적합성 판단까지 포함된 응답

### 2.3 AI 연구자와의 대화 준비

```python
response = alfred.run(
    "One of our guests is from Qwen. What is their most popular model?"
)
print(response)
```

- Hub 통계 도구 사용
- 다운로드 수 기반 대화 포인트 생성

### 2.4 멀티툴 조합 추론

```python
response = alfred.run(
    "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy."
)
print(response)
```

Alfred는 내부적으로:
1. 게스트 정보 검색
2. 웹 검색
3. 정보 종합
4. 대화 주제 제안

을 **자동으로 조합**한다.

## 3. 고급 기능: 대화 메모리

### 3.1 smolagents 메모리

```python
alfred.run("Tell me about Lady Ada Lovelace.")
alfred.run("What projects is she working on?", reset=False)
```

- `reset=False`로 명시적 유지

### 3.2 LlamaIndex 메모리

```python
from llama_index.core.workflow import Context

ctx = Context(alfred)
await alfred.run("Tell me about Lady Ada Lovelace.", ctx=ctx)
await alfred.run("What projects is she working on?", ctx=ctx)
```

### 3.3 LangGraph 메모리

```python
response = alfred.invoke({
    "messages": [
        HumanMessage(content="Tell me about Lady Ada Lovelace."),
        HumanMessage(content="What projects is she working on?")
    ]
})
```

또는 `MemorySaver` 컴포넌트 활용 가능.

## 메모리 설계가 분리된 이유


| 프레임워크 | 메모리 방식 |
|---|---|
| smolagents | 실행 단위, 명시적 유지 |
| LlamaIndex | Context 객체 |
| LangGraph | 메시지/스토어 기반 |

**에이전트 로직과 메모리는 의도적으로 분리**  
-> 확장성, 재현성, 운영 안정성 확보

## 최종 정리

이제 Alfred는 다음을 모두 수행할 수 있다.

- 게스트 맞춤 정보 제공
- 최신 뉴스·날씨 기반 판단
- AI 트렌드에 대한 대화
- 복수 도구를 조합한 추론
- 대화 맥락 유지

즉, Alfred는  
**“현실적인 Agentic RAG의 완성형 예제”**다.

이 구조는 그대로:
- 사내 비서
- 분석 에이전트
- 운영 자동화
- 리서치 어시스턴트

로 확장 가능하다.

참고자료
Huggingface, agents course, https://huggingface.co/learn