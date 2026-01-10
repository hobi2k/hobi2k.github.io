---
layout: post
title:  "허깅페이스 에이전트 코스 - Building and Integrating Tools for Your Agent"
date:   2025-01-10 00:10:22 +0900
categories: Huggingface_agent
---

# Building and Integrating Tools for Your Agent

이번 글에서는 Alfred에게 **외부 세계와 상호작용할 수 있는 도구(Tools)**를 제공한다.  
이를 통해 Alfred는 단순한 내부 지식 응답을 넘어, **최신 정보·날씨·AI 트렌드**까지 아우르는 진정한 “르네상스 호스트”로 거듭난다.

구체적으로 Alfred는 다음 능력을 갖추게 된다.

- 웹 검색을 통한 최신 뉴스 및 일반 지식 확인
- 날씨 정보를 활용한 행사 일정 판단 (불꽃놀이)
- Hugging Face Hub 통계를 활용한 AI 관련 대화

이 모든 기능은 **Agentic RAG 환경에서 “도구”로 통합**된다.

## 1. 에이전트에게 웹 접근 권한 부여하기

Alfred가 세계 정세와 최신 정보를 논할 수 있으려면,  
먼저 **웹 검색 도구**가 필요하다.

아래는 동일한 목적(웹 검색)을 서로 다른 프레임워크에서 구현한 예시들이다.

### 1.1 smolagents 기반 DuckDuckGo 검색 도구

```python
from smolagents import DuckDuckGoSearchTool

# DuckDuckGo 검색 도구 초기화
search_tool = DuckDuckGoSearchTool()

# 예시 사용
results = search_tool("Who's the current President of France?")
print(results)
```

**예상 출력**
```
The current President of France in Emmanuel Macron.
```

### 1.2 llama-index 기반 DuckDuckGo 검색 도구

```python
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool

# DuckDuckGo 검색 스펙 초기화
tool_spec = DuckDuckGoSearchToolSpec()

# FunctionTool로 래핑
search_tool = FunctionTool.from_defaults(
    tool_spec.duckduckgo_full_search
)

# 예시 사용
response = search_tool("Who's the current President of France?")
print(response.raw_output[-1]["body"])
```

### 1.3 LangChain 기반 DuckDuckGo 검색 도구

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("Who's the current President of France?")
print(results)
```

이처럼 **동일한 개념의 도구라도 프레임워크에 따라 구현 방식만 달라질 뿐**,  
Agentic RAG 관점에서는 모두 “웹 검색 도구”라는 동일한 역할을 수행한다.

## 2. 불꽃놀이를 위한 날씨 정보 도구 만들기

완벽한 갈라의 마무리를 위해서는 **날씨 확인**이 필수다.  
여기서는 이해를 돕기 위해 **더미(Dummy) 날씨 API**를 사용한다.

### 2.1 smolagents 기반 날씨 도구

```python
from smolagents import Tool
import random

class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # 더미 날씨 데이터
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]

        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# 도구 초기화
weather_info_tool = WeatherInfoTool()
```

### 2.2 llama-index 기반 날씨 도구

```python
import random
from llama_index.core.tools import FunctionTool

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]

    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

weather_info_tool = FunctionTool.from_defaults(get_weather_info)
```

### 2.3 LangChain 기반 날씨 도구

```python
from langchain_core.tools import Tool
import random

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]

    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)
```

## 3. Hugging Face Hub 통계 도구 만들기

갈라에는 유명 AI 연구자와 개발자들이 참석한다.  
Alfred는 **그들의 가장 인기 있는 모델**을 언급하며 대화를 이끌 수 있어야 한다.


### 3.1 smolagents 기반 Hub Stats 도구

```python
from smolagents import Tool
from huggingface_hub import list_models

class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            models = list(
                list_models(
                    author=author,
                    sort="downloads",
                    direction=-1,
                    limit=1
                )
            )

            if models:
                model = models[0]
                return (
                    f"The most downloaded model by {author} is "
                    f"{model.id} with {model.downloads:,} downloads."
                )
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"

hub_stats_tool = HubStatsTool()
```

### 3.2 llama-index 기반 Hub Stats 도구

```python
from llama_index.core.tools import FunctionTool
from huggingface_hub import list_models

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        models = list(
            list_models(
                author=author,
                sort="downloads",
                direction=-1,
                limit=1
            )
        )

        if models:
            model = models[0]
            return (
                f"The most downloaded model by {author} is "
                f"{model.id} with {model.downloads:,} downloads."
            )
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)
```

### 3.3 LangChain 기반 Hub Stats 도구

```python
from langchain_core.tools import Tool
from huggingface_hub import list_models

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        models = list(
            list_models(
                author=author,
                sort="downloads",
                direction=-1,
                limit=1
            )
        )

        if models:
            model = models[0]
            return (
                f"The most downloaded model by {author} is "
                f"{model.id} with {model.downloads:,} downloads."
            )
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)
```

## 4. Alfred에게 모든 도구 통합하기

이제 Alfred에게 다음 도구들을 모두 장착한다.

- 웹 검색 도구
- 날씨 정보 도구
- Hugging Face Hub 통계 도구

### 4.1 smolagents 기반 통합

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()

alfred = CodeAgent(
    tools=[search_tool, weather_info_tool, hub_stats_tool],
    model=model
)

response = alfred.run(
    "What is Facebook and what's their most popular model?"
)

print(response)
```

### 4.2 llama-index 기반 통합

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
)

alfred = AgentWorkflow.from_tools_or_functions(
    [search_tool, weather_info_tool, hub_stats_tool],
    llm=llm
)

response = await alfred.run(
    "What is Facebook and what's their most popular model?"
)

print(response)
```

### 4.3 LangGraph 기반 통합 (Agentic RAG)

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm)
chat_with_tools = chat.bind_tools(
    [search_tool, weather_info_tool, hub_stats_tool]
)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [
            chat_with_tools.invoke(state["messages"])
        ]
    }

builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(
    [search_tool, weather_info_tool, hub_stats_tool]
))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

alfred = builder.compile()

messages = [
    HumanMessage(
        content="Who is Facebook and what's their most popular model?"
    )
]

response = alfred.invoke({"messages": messages})
print(response["messages"][-1].content)
```

## 결론

이제 Alfred는 다음을 모두 수행할 수 있다.

- 최신 웹 정보 검색
- 실시간(또는 유사 실시간) 날씨 판단
- AI 트렌드 및 인기 모델 언급

이는 **Agentic RAG의 핵심 가치**를 잘 보여준다.

> 에이전트는 더 이상 “답변 생성기”가 아니라  
> **필요한 정보를 스스로 찾아 행동하는 시스템**이다.

참고자료
Huggingface, agents course, https://huggingface.co/learn