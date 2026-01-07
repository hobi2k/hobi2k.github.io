---
layout: post
title:  "허깅페이스 에이전트 코스 - Tools"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# What are Tools?

AI Agent의 가장 중요한 특징 중 하나는 **실제로 행동(Action)을 수행할 수 있다는 점**이다.  
그리고 이 행동은 전부 **Tool**을 통해 이루어진다.

이 글에서는 다음을 정리한다.

- Tool이란 무엇인가
- Tool을 어떻게 설계해야 하는가
- Tool을 LLM에 어떻게 “알려주는가”
- 왜 Tool 설계가 Agent 품질을 좌우하는가

## 실습 목표

- Tool과 Action의 차이 명확히 구분
- LLM이 Tool을 “직접 실행하지 않는다”는 구조 이해
- Tool description이 왜 중요한지 이해
- Python 기반 Tool 자동 정의 방식 이해
- MCP(Model Context Protocol)의 등장 배경 이해

## Tool이란 무엇인가?

**Tool이란, LLM에게 제공되는 “외부 기능 함수”**다.

중요한 점은 다음이다.

> Tool은 **LLM이 할 수 없는 일을 대신 수행**하기 위해 존재한다.

대표적인 Tool 예시는 다음과 같다.


| Tool | 설명 |
|-----|-----|
| Web Search | 최신 정보를 인터넷에서 검색 |
| Image Generation | 텍스트 기반 이미지 생성 |
| Retrieval | 외부 DB / 문서 검색 |
| API Interface | GitHub, YouTube, Spotify 등 외부 API 호출 |


핵심은 이것이다.

- Tool은 무엇이든 될 수 있다
- 단, **LLM의 한계를 보완해야 한다**

## 왜 Tool이 필요한가?

LLM은 다음과 같은 한계를 가진다.

1. **계산에 약하다**
   - 산술, 정확한 수치 계산
2. **실시간 정보가 없다**
   - 학습 시점 이후의 정보는 모름
3. **환경에 직접 개입할 수 없다**
   - 파일, API, DB를 스스로 조작 불가

따라서 다음과 같은 경우 Tool이 필수다.

- 계산기 -> 계산 Tool
- 오늘 날씨 -> 검색 Tool
- 이메일 전송 -> 메일 Tool

Tool 없이 “오늘 파리 날씨”를 물으면  
LLM은 **확률적으로 그럴듯한 거짓말(hallucination)**을 할 뿐이다.

## Tool이 반드시 포함해야 할 정보

LLM이 Tool을 올바르게 사용하려면, Tool에 관한 **정확한 정보**를 알아야 한다.
최소한 다음 정보가 필요하다.

- Tool 이름
- Tool이 무엇을 하는지에 대한 설명
- 입력 인자와 타입
- 출력 값과 타입 (선택)

정리하면:

> Tool = 명확한 목적 + 호출 가능한 함수 + 정확한 입출력 계약

---

## Tool은 어떻게 동작하는가?

중요한 구조적 사실부터 짚자.

> **LLM은 Tool을 직접 호출하지 못한다.**  
> LLM은 오직 “텍스트”만 생성한다.

Tool 호출의 실제 흐름은 다음과 같다.

1. 사용자가 질문을 한다
2. LLM이 판단한다  
   -> “이건 Tool을 써야겠다”
3. LLM이 **텍스트로 된 Tool 호출 요청**을 생성한다
4. Agent 런타임이 그 텍스트를 해석한다
5. Agent가 실제 Tool(함수)을 실행한다
6. 실행 결과를 **새 메시지로 추가**한다
7. LLM이 그 결과를 읽고 자연어 응답을 생성한다

사용자 입장에서는  
“모델이 직접 Tool을 쓴 것처럼” 보이지만,  
실제로는 **Agent가 모든 실행을 대신 처리**한다.

## Tool을 LLM에게 어떻게 알려주는가?

핵심은 **System Message**다.

System Message 안에 다음 정보를 넣는다.

- 사용 가능한 Tool 목록
- 각 Tool의 설명
- 입력 인자와 타입

이때 가장 중요한 조건은 다음이다.

> **정확하고, 일관된 형식으로 설명할 것**

그래서 Tool description은 보통 다음처럼 작성된다.

```text
Tool Name: calculator,
Description: Multiply two integers.
Arguments: a: int, b: int
Outputs: int
```

이 텍스트는 **사람을 위한 설명이 아니라, LLM을 위한 사양(spec)**이다.

## 예제: 간단한 계산기 Tool

1. Python 함수 구현

```python
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

이 함수에는 이미 Tool에 필요한 정보가 모두 들어 있다.

- 함수 이름 -> Tool 이름
- docstring -> 설명
- type hint -> 입력/출력 타입

2. 수동 Tool 설명 (비권장)

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

이 방식은 실수하기 쉽고 유지보수가 어렵다.

## 자동 Tool 정의: Decorator 방식

그래서 대부분의 프레임워크는
코드에서 Tool 사양을 자동 추출한다.

이를 위해 `@tool` 같은 데코레이터를 사용한다.

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

출력:

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

즉, 사람이 쓴 코드에서 LLM이 이해할 Tool 설명을 자동 생성

## Generic Tool 클래스 개념

Tool을 일반화하면 다음과 같은 구조를 가진다.

- name: Tool 이름
- description: 역할 설명
- func: 실제 실행 함수
- arguments: 입력 인자 정보
- outputs: 출력 타입

```python
from typing import Callable


class Tool:
    """
    A class representing a reusable piece of code (Tool).

    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of arguments.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool,
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])

        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)
```

핵심은 이것이다.
Tool은 **실행 로직 + LLM용 사양(spec)**을 함께 묶은 객체다.

## Tool description은 어디로 가는가?

결국 Tool description은 System Message에 주입된다.

예시:

```
You are an AI agent with access to the following tools:

Tool Name: calculator
Description: Multiply two integers.
Arguments: a: int, b: int
Outputs: int

When you need to use a tool, output a JSON object with the tool name and arguments.
```

이렇게 해서 LLM은 다음을 학습한다.

- 어떤 Tool이 있는지
- 언제 사용해야 하는지
- 어떤 형식으로 호출해야 하는지

## MCP(Model Context Protocol)

Model Context Protocol(MCP)은
Tool 인터페이스를 표준화하기 위한 오픈 프로토콜이다.

MCP의 목표는 다음과 같다.

- Tool 정의 방식 통일
- 프레임워크 간 Tool 재사용
- LLM 공급자 변경에 대한 유연성
- 보안 및 인프라 베스트 프랙티스 제공

즉, 한 번 정의한 Tool을
여러 Agent 프레임워크에서 재사용하기 위한 규약이다.

## 정리

이 글에서 다룬 핵심은 다음이다.

- Tool은 LLM의 한계를 보완하는 외부 기능이다
- LLM은 Tool을 직접 실행하지 않고, “호출 요청 텍스트”를 생성한다
- Tool description은 System Message를 통해 전달된다
- 정확한 Tool 사양이 Agent 품질을 결정한다
- 자동 Tool 정의(@tool)는 사실상 필수다
- MCP는 Tool 인터페이스의 표준화를 목표로 한다


참고자료
Huggingface, agents course, https://huggingface.co/learn