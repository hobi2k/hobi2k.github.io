---
layout: post
title:  "허깅페이스 에이전트 코스 - Tools in smolagent"
date:   2025-01-07 00:10:22 +0900
categories: Huggingface_agent
---

# Tools in smolagents

Agent는 스스로 행동하지 않는다.  
Agent는 **도구(Tool)**를 통해서만 환경과 상호작용한다.

`smolagents`에서 Tool은 단순한 보조 기능이 아니라,  
**LLM이 “행동(Action)”을 수행하기 위한 핵심 인터페이스**다.

이 문서에서는 다음을 다룬다.

- Tool이 무엇인지
- LLM이 Tool을 호출하기 위해 어떤 정보가 필요한지
- Tool을 정의하는 두 가지 방법
- 기본 제공 Toolbox
- Tool을 Hub / Space / LangChain / MCP에서 불러오는 방법

## 1. Tool이란 무엇인가

`smolagents`에서 Tool은 다음과 같이 정의된다.

> **Tool = LLM이 Agent 시스템 안에서 호출할 수 있는 함수**

LLM은 실제 세계에 접근할 수 없기 때문에,  
다음과 같은 작업은 반드시 Tool을 통해서만 가능하다.

- 웹 검색
- API 호출
- 계산
- 파일 / 웹 페이지 접근
- 이미지 생성

LLM이 Tool을 사용하려면, **명확한 인터페이스 설명**이 필요하다.

### Tool 인터페이스의 필수 구성 요소

모든 Tool은 최소한 다음 정보를 제공해야 한다.

- **Name**: Tool의 이름
- **Description**: Tool이 무엇을 하는지
- **Input types + descriptions**: 어떤 인자를 받는지
- **Output type**: 무엇을 반환하는지

예시 (웹 검색 Tool):

- Name: `web_search`
- Description: Searches the web for specific queries
- Input: `query` (string)
- Output: 검색 결과 문자열

이 정보는 **System Prompt에 주입**되어 LLM이 Tool을 “이해”하도록 만든다.

## 2. Tool 생성 방법 개요

`smolagents`에서는 Tool을 두 가지 방식으로 정의할 수 있다.

1. **`@tool` 데코레이터 (권장, 단순 Tool)**
2. **`Tool` 클래스를 상속 (복잡한 Tool)**

두 방식 모두 최종적으로는  
LLM이 이해할 수 있는 Tool 스펙을 생성한다.

## 3. `@tool` 데코레이터로 Tool 만들기

`@tool`은 **가장 간단하고 권장되는 방식**이다.  
Python 함수 하나만 있으면 Tool을 만들 수 있다.

### `@tool`을 사용할 때 중요한 규칙

1. 함수 이름은 **의도를 잘 드러내야 한다**
2. **입력 / 출력 타입 힌트**를 반드시 명시한다
3. **docstring 안에 Args 설명을 정확히 작성한다**

LLM은 docstring을 읽고 Tool 사용법을 학습한다.

### 예제: 최고 평점 케이터링 업체 검색 Tool

Alfred는 파티를 준비하면서  
고담 시티에서 가장 평점이 높은 케이터링 업체를 찾고 싶다.

```python
from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    best_service = max(services, key=services.get)
    return best_service
```

이 Tool은 다음 정보를 LLM에 제공한다.

- Tool 이름: catering_service_tool
- 기능: 고담 시티 최고 평점 케이터링 반환
- 입력: query: str
- 출력: str

### Agent에 Tool 주입 및 실행

```python
agent = CodeAgent(
    tools=[catering_service_tool],
    model=InferenceClientModel()
)

result = agent.run(
    "Can you give me the name of the highest-rated catering service in Gotham City?"
)

print(result)
```

출력 예시:

```
Gotham Catering Co.
```

## 4. Tool 클래스를 상속해 Tool 만들기

복잡한 Tool이 필요한 경우,
함수 대신 Tool 클래스를 상속해 정의할 수 있다.

이 방식은 다음이 필요하다.

- name
- description
- inputs (dict)
- output_type
- forward() 메서드

### 예제: 슈퍼히어로 파티 테마 생성 Tool

```python
from smolagents import Tool, CodeAgent, InferenceClientModel

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea.
    """

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade with Batman villains.",
            "futuristic gotham": "Neo-Gotham Night: Cyberpunk-style Batman Beyond party."
        }

        return themes.get(
            category.lower(),
            "Theme not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'."
        )
```

### Agent 실행

```python
party_theme_tool = SuperheroPartyThemeTool()

agent = CodeAgent(
    tools=[party_theme_tool],
    model=InferenceClientModel()
)

result = agent.run(
    "What would be a good superhero party idea for a villain masquerade theme?"
)

print(result)
```

## 5. Default Toolbox (기본 제공 Tool)

smolagents는 자주 쓰이는 Tool을 기본으로 제공한다.
대표적인 기본 Tool은 다음과 같다.

- PythonInterpreterTool
- FinalAnswerTool
- UserInputTool
- DuckDuckGoSearchTool
- GoogleSearchTool
- VisitWebpageTool

예를 들어 Alfred는 다음 흐름으로 파티를 준비할 수 있다.

- 웹 검색: DuckDuckGoSearchTool
- 계산: PythonInterpreterTool
- 결과 정리: FinalAnswerTool

## 6. Tool 공유와 재사용 (Hub / Space / LangChain)

### 6.1 Tool을 Hub에 공유하기

```python
party_theme_tool.push_to_hub(
    "{your_username}/party_theme_tool",
    token=""
)
```

### 6.2 Hub에서 Tool 불러오기

```python
from smolagents import load_tool, CodeAgent, InferenceClientModel

image_generation_tool = load_tool(
    "m-ric/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    tools=[image_generation_tool],
    model=InferenceClientModel()
)

agent.run(
    "Generate an image of a luxurious superhero-themed party at Wayne Manor."
)
```

### 6.3 Hugging Face Space를 Tool로 사용하기

```python
from smolagents import CodeAgent, InferenceClientModel, Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(
    tools=[image_generation_tool],
    model=model
)

agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={
        "user_prompt": "A grand superhero-themed party at Wayne Manor"
    }
)
```

### 6.4 LangChain Tool 불러오기

```python
pip install -U langchain-community
```

```python
import os
os.environ['SERPAPI_API_KEY'] = '...'
```

```python
from langchain.agents import load_tools
from smolagents import CodeAgent, InferenceClientModel, Tool

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(
    tools=[search_tool],
    model=InferenceClientModel()
)

agent.run(
    "Search for luxury entertainment ideas for a superhero-themed event."
)
```

### 6.5 MCP 서버에서 Tool 컬렉션 불러오기

```python
pip install "smolagents[mcp]"
```

```python
import os
from smolagents import ToolCollection, CodeAgent, InferenceClientModel
from mcp import StdioServerParameters

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(
    server_parameters,
    trust_remote_code=True
) as tool_collection:
    agent = CodeAgent(
        tools=[*tool_collection.tools],
        model=model,
        add_base_tools=True
    )

    agent.run("Please find a remedy for hangover.")
```

## 7. 정리

- Tool은 Agent의 행동 그 자체
- LLM은 Tool 설명을 통해 “무엇을 할 수 있는지”를 배운다
- @tool은 가장 단순하고 강력한 방법
- Tool 클래스는 복잡한 Tool에 적합
- Hub, Space, LangChain, MCP를 통해 Tool은 재사용 자산이 된다

참고자료
Huggingface, agents course, https://huggingface.co/learn