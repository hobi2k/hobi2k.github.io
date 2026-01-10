---
layout: post
title:  "허깅페이스 에이전트 코스 - Document Analysis Graph"
date:   2025-01-9 00:10:22 +0900
categories: Huggingface_agent
---

# Document Analysis Graph

이번 글에서는 LangGraph를 사용해 **문서 분석(Document Analysis) 시스템**을 구축한다.  
집사 **Alfred**는 Mr. Wayne이 남긴 각종 문서(훈련 계획, 식단, 이미지 메모 등)를 분석하고,  
필요 시 계산을 수행하며, 요약과 실행 지시까지 처리한다.

이 예제의 목적은 다음과 같다.

- 이미지 문서를 입력으로 받는다
- Vision Language Model(VLM)을 사용해 텍스트를 추출한다
- 필요 시 계산과 같은 일반 도구를 호출한다
- 문서 내용을 분석하고 요약 및 지시를 수행한다
- 위 과정을 **LangGraph 기반 제어 흐름**으로 안정적으로 구성한다

## Alfred의 문서 분석 워크플로우

이 시스템은 다음과 같은 구조적 흐름을 따른다.

1. 문서(이미지/PDF 등)를 입력으로 받는다
2. 필요하면 Vision 모델을 사용해 텍스트를 추출한다
3. 계산이나 추가 분석이 필요한 경우 도구를 호출한다
4. 결과를 종합해 사용자 요청에 답한다
5. 더 이상 도구 호출이 필요 없으면 종료한다

이 구조는 **ReAct(Reason -> Act -> Observe)** 패턴을 LangGraph로 구현한 예제다.

## 환경 설정

### 필수 패키지 설치

```python
# LangGraph, OpenAI 연동, LangChain 핵심 모듈 설치
%pip install langgraph langchain_openai langchain_core
```

### 모듈 임포트

```python
import base64
from typing import List, TypedDict, Annotated, Optional

# OpenAI LLM
from langchain_openai import ChatOpenAI

# LangChain 메시지 타입
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

# LangGraph 관련 모듈
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 시각화용
from IPython.display import Image, display
```

## Agent State 정의

이번 State는 이전 예제보다 복잡하다.  
특히 `Annotated[..., add_messages]`를 사용해 **메시지를 누적(add)**하는 방식이 핵심이다.

```python
class AgentState(TypedDict):
    # 분석할 문서 파일 경로 (이미지 또는 PDF)
    input_file: Optional[str]

    # LLM과의 모든 대화 메시지
    # add_messages 연산자를 통해 이전 메시지 위에 누적된다
    messages: Annotated[list[AnyMessage], add_messages]
```

### 핵심 포인트

- `AnyMessage`  
  -> LangChain에서 정의한 메시지 추상 타입  
- `add_messages`  
  -> 새 메시지를 기존 메시지 리스트에 **덮어쓰지 않고 추가**
- 이를 통해 에이전트는 **이전 맥락을 유지**한 채 반복 실행 가능

## 도구(Tools) 준비

### Vision LLM 초기화

```python
# 이미지 입력을 처리할 수 있는 Vision Language Model
vision_llm = ChatOpenAI(model="gpt-4o")
```

### 이미지에서 텍스트 추출 도구

```python
def extract_text(img_path: str) -> str:
    """
    이미지 파일로부터 텍스트를 추출하는 도구.
    Vision Language Model을 사용한다.

    Args:
        img_path: 로컬 이미지 파일 경로

    Returns:
        이미지에서 추출된 텍스트 문자열
    """
    all_text = ""
    try:
        # 이미지 파일을 바이너리로 읽기
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        # base64 인코딩 (OpenAI 이미지 입력 형식)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 텍스트 + 이미지 입력을 동시에 전달
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Vision 모델 호출
        response = vision_llm.invoke(message)

        # 결과 텍스트 누적
        all_text += response.content + "\n\n"

        return all_text.strip()

    except Exception as e:
        # 오류 발생 시에도 빈 문자열 반환 (에이전트 안정성 확보)
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""
```

### 계산용 도구

```python
def divide(a: int, b: int) -> float:
    """
    단순 나눗셈 도구.
    문서 분석 중 필요한 계산을 시연하기 위한 예제.
    """
    return a / b
```

### 도구 등록 및 LLM 바인딩

```python
# Alfred가 사용할 수 있는 도구 목록
tools = [
    divide,
    extract_text
]

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o")

# 도구를 바인딩한 LLM
# parallel_tool_calls=False → 한 번에 하나의 도구만 호출
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

## Assistant 노드 정의

이 노드는 **에이전트의 두뇌 역할**을 한다.  
LLM에 시스템 프롬프트를 제공하고, 필요 시 도구 호출을 유도한다.

```python
def assistant(state: AgentState):
    """
    Alfred의 핵심 추론 노드.
    문서 분석 요청을 이해하고,
    필요 시 도구 호출을 생성한다.
    """

    # 도구 설명을 텍스트로 제공 (LLM 이해용)
    textual_description_of_tool = """
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

divide(a: int, b: int) -> float:
    Divide a and b
"""

    image = state["input_file"]

    # 시스템 메시지: 역할, 도구, 현재 이미지 정보 명시
    sys_msg = SystemMessage(
        content=(
            "You are a helpful butler named Alfred that serves Mr. Wayne and Batman. "
            "You can analyse documents and run computations with provided tools:\n"
            f"{textual_description_of_tool}\n"
            f"You have access to some optional images. "
            f"Currently the loaded image is: {image}"
        )
    )

    # LLM 호출 결과를 messages에 추가
    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }
```


## ReAct 패턴 설명

이 에이전트는 **ReAct(Reason–Act–Observe)** 패턴을 따른다.

1. **Reason**  
   문서와 요청을 분석
2. **Act**  
   필요 시 도구 호출
3. **Observe**  
   도구 결과를 확인
4. **Repeat**  
   추가 도구 호출이 필요하면 반복

LangGraph는 이 반복 구조를 **그래프 루프**로 안전하게 구현한다.


## LangGraph 구성

```python
# StateGraph 생성
builder = StateGraph(AgentState)

# 노드 등록
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 시작 지점
builder.add_edge(START, "assistant")

# assistant 결과에 따라 분기
# - 도구 호출이면 tools 노드로
# - 아니면 END로 종료
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

# 도구 실행 후 다시 assistant로 복귀
builder.add_edge("tools", "assistant")

# 그래프 컴파일
react_graph = builder.compile()

# 그래프 구조 시각화
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```

### 흐름 요약

- assistant 실행
- 도구 호출이 있으면 -> tools
- tools 실행 후 -> assistant
- 더 이상 도구 호출이 없으면 -> END

## 실행 예제 1: 계산 요청

```python
messages = [HumanMessage(content="Divide 6790 by 5")]

result = react_graph.invoke({
    "messages": messages,
    "input_file": None
})

# 메시지 출력
for m in result["messages"]:
    m.pretty_print()
```

### 내부 흐름

```
Human -> assistant
assistant -> divide 도구 호출
tools -> 결과 반환
assistant -> 최종 응답 생성
```

## 실행 예제 2: 문서 이미지 분석

```python
messages = [
    HumanMessage(
        content=(
            "According to the note provided by Mr. Wayne in the provided images. "
            "What's the list of items I should buy for the dinner menu?"
        )
    )
]

result = react_graph.invoke({
    "messages": messages,
    "input_file": "Batman_training_and_meals.png"
})
```

### 내부 흐름

1. assistant가 이미지 분석 필요 판단
2. extract_text 도구 호출
3. 추출된 텍스트 관찰
4. 분석 결과를 종합해 최종 답변 생성

## 핵심 정리

- LangGraph는 ReAct 패턴을 **명시적 제어 흐름**으로 구현할 수 있다
- 도구 호출 여부에 따라 안전하게 루프를 구성할 수 있다
- `add_messages` 연산자를 통해 대화 맥락을 자연스럽게 유지할 수 있다
- Vision 모델 + 일반 도구를 하나의 그래프로 통합할 수 있다


참고자료
Huggingface, agents course, https://huggingface.co/learn