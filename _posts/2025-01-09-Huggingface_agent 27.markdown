---
layout: post
title:  "허깅페이스 에이전트 코스 - Building Blocks of LangGraph"
date:   2025-01-9 00:10:22 +0900
categories: Huggingface_agent
---

# Building Blocks of LangGraph

LangGraph 애플리케이션을 구축하기 위해서는 몇 가지 **핵심 구성 요소(Building Blocks)**를 이해해야 한다.  
LangGraph에서 하나의 애플리케이션은 **엔트리포인트(entrypoint)**에서 시작하여, 실행 흐름에 따라 여러 함수(Node)를 거친 뒤 **END**에 도달하는 구조를 가진다.

이 흐름 전체를 구성하는 기본 요소는 다음과 같다.

## 1. State

**State**는 LangGraph의 가장 핵심적인 개념이다.  
애플리케이션 전반에 걸쳐 **공유되고 전달되는 모든 정보의 집합**을 의미한다.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

- State는 **사용자가 직접 정의(User-defined)**한다
- 노드 간 전이와 분기 판단의 기준이 된다
- 각 노드는 State를 입력으로 받고, 일부를 갱신하여 반환한다

즉, State 설계는 곧 **애플리케이션의 사고 구조를 설계하는 일**과 같다.

> Tip  
> 각 단계에서 무엇을 판단해야 하는지 먼저 생각하고,  
> 그 판단에 필요한 정보만 State에 포함시키는 것이 중요하다.

## 2. Nodes

**Node**는 하나의 **Python 함수**이다.  
각 노드는 다음과 같은 공통 규칙을 따른다.

- State를 입력으로 받는다
- 특정 작업을 수행한다
- State의 변경 사항을 반환한다

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " sad!"}
```

Node 안에는 다음과 같은 로직이 들어갈 수 있다.

- LLM 호출 (텍스트 생성, 판단 수행)
- 외부 Tool 호출 (API, DB, 파일 등)
- 조건 판단 로직
- 사용자 입력(Human-in-the-loop)

LangGraph에는 워크플로우 전체를 위해 기본 제공되는 노드도 존재한다.

- **START**
- **END**

이들은 사용자가 직접 정의하지 않아도 된다.

## 3. Edges

**Edge**는 노드와 노드를 연결하며,  
**다음에 어떤 노드를 실행할지 결정하는 규칙**을 정의한다.

```python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    user_input = state["graph_state"]

    # 예시: 50 / 50 확률로 다음 노드 선택
    if random.random() < 0.5:
        return "node_2"
    else:
        return "node_3"
```

- Edge는 **결정적(deterministic)** 일 수도 있고
- **LLM 출력이나 상태 값에 따라 비결정적(non-deterministic)** 일 수도 있다

State를 기준으로 다음 실행 노드를 선택하는 것이 핵심이다.

## 실행 예시

그래프를 실행할 때는 초기 State를 전달한다.

```python
graph.invoke({"graph_state": "Hi, this is Lance."})
```

출력 예시는 다음과 같다.

```text
---Node 1---
---Node 3---
{'graph_state': 'Hi, this is Lance. I am sad!'}
```

이 과정에서:

1. START -> node_1 실행
2. Edge 로직에 따라 node_3 선택
3. END에 도달하며 최종 State 반환

## 정리

LangGraph의 기본 구성 요소는 다음과 같이 정리할 수 있다.

- **State**  
  -> 애플리케이션 전반에 걸쳐 유지되는 정보

- **Node**  
  -> 하나의 처리 단계(함수)

- **Edge**  
  -> 상태를 기준으로 다음 단계를 결정하는 전이 규칙

이 세 가지를 조합해,  
LangGraph는 LLM 애플리케이션을 **명확한 흐름을 가진 시스템**으로 설계할 수 있게 해준다.

## 다음 단계

다음 섹션에서는 이 개념들을 실제로 활용하여,  
**Alfred가 이메일을 분류하고 응답 초안을 작성하는 첫 번째 그래프**를 직접 만들어본다.


참고자료
Huggingface, agents course, https://huggingface.co/learn