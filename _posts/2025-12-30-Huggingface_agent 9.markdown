---
layout: post
title:  "허깅페이스 에이전트 코스 - Dummy Agent Library"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Dummy Agent Library: 프레임워크 없이 Agent 구조 이해하기

이번 글의 목적은 **특정 프레임워크 사용법이 아니라, Agent의 핵심 개념을 이해하는 것**이다.  
그래서 이 과정은 의도적으로 **프레임워크에 종속되지 않은(dummy) Agent 구현**을 사용한다.

이 글에서는 다음을 정리한다.

- 왜 Dummy Agent Library를 쓰는지
- Serverless API로 LLM을 호출하는 방식
- System Prompt가 Agent의 “엔진”이라는 사실
- Stop & Parse가 없으면 왜 환각(hallucination)이 발생하는지
- Agent를 직접 구현하는 것이 왜 번거로운지

이 과정을 이해하면,  
이후 `smolagents`, `LangGraph`, `LlamaIndex` 같은 라이브러리가 **무엇을 대신해주는지**가 명확해진다.

## 왜 Dummy Agent Library를 쓰는가?

여기서 프레임워크 중립적인 이유는 명확하다.

- Agent의 본질은 **개념과 구조**
- 프레임워크는 구현상의 편의일 뿐
- 개념을 이해하면 어떤 프레임워크든 쓸 수 있다

그래서 여기서는 **Agent의 작동 원리 자체**에 집중한다.

## Serverless API: 설치 없는 LLM 호출

Hugging Face의 **Serverless API**를 사용하면:

- 모델 설치 x
- 서버 배포 x
- 즉시 추론 가능 o

### 기본 설정

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
)
```

### chat 메서드 사용 (권장)

```python
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
)

print(output.choices[0].message.content)
```

출력:

```
Paris.
```

chat 메서드는
모델별 chat template 차이를 자동 처리해주므로 권장된다.

## Dummy Agent의 핵심: System Prompt

Agent 라이브러리의 핵심은 단 하나다.

*System Prompt에 무엇을 넣느냐*

Dummy Agent에서는 System Prompt 안에 이미 다음이 들어 있다.

- Tool 설명
- Action 포맷(JSON)
- Thought–Action–Observation 사이클 규칙
- 출력 형식 강제

## System Prompt 예제 (핵심 구조)

```python
SYSTEM_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.

ALWAYS use the following format:

Question:
Thought:
Action:
$JSON_BLOB

Observation:
...

Thought: I now know the final answer
Final Answer:
"""
```

이 System Prompt는 다음을 모두 강제한다.

- Tool 사용 방법
- JSON Action 포맷
- 단일 Action 원칙
- Thought / Action / Observation 반복 구조
- Final Answer로 종료

이것이 바로 Agent의 행동 규칙 엔진이다.

## 사용자 입력은 어떻게 결합되는가?

Agent 호출 시 실제로 전달되는 메시지는 다음과 같다.

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]
```

LLM 입장에서는:

- “규칙 + 도구 설명”을 읽고
- 그 다음에 사용자 질문을 읽는 구조

## 첫 번째 문제: Observation 환각

LLM을 그대로 실행하면 이런 출력이 나온다.

```python
Thought: I need to get the weather in London.
Action:
{
  "action": "get_weather",
  "action_input": {"location": "London"}
}
Observation: The current weather in London is partly cloudy with 12°C.
```

문제점

- Observation은 실제 Tool 실행 결과가 아님
- 모델이 자기 혼자 만들어낸 환각

즉, LLM은 Action만 “설명”해야 하는데
Observation까지 상상으로 생성해버린다.

## 해결책: Stop & Parse

이를 막기 위해 stop 토큰을 사용한다.

```python
output = client.chat.completions.create(
    messages=messages,
    max_tokens=150,
    stop=["Observation:"]
)
```

이제 출력은 여기서 멈춘다.

```python
Thought: I need to get the current weather in London.
Action:
{
  "action": "get_weather",
  "action_input": {"location": "London"}
}
```

- Action 정의까지만 생성
- Observation은 Runtime이 책임짐

## Dummy Tool 실행

이제 실제로 Tool을 실행한다.

```python
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures.\n"
```

## Observation을 직접 주입하고 재호출

이제 Agent Runtime이 하는 일은 다음이다.

1. Action 실행
2. 결과를 Observation으로 추가
3. 다시 LLM 호출

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
    {
        "role": "assistant",
        "content": output.choices[0].message.content
        + "Observation:\n"
        + get_weather("London")
    },
]

output = client.chat.completions.create(
    messages=messages,
    max_tokens=200,
)

print(output.choices[0].message.content)
```

출력:

```python
Final Answer: The weather in London is sunny with low temperatures.
```

## 이 예제가 보여주는 핵심 교훈

1. Agent는 결국 “프롬프트 조립기 + 실행기”다
- LLM은 텍스트 생성
- Runtime은:
   - stop
   - parse
   - execute
   - observe
   - re-prompt

2. Observation은 반드시 Runtime이 넣어야 한다
- LLM에게 Observation 생성을 맡기면 안 된다.
- 실제 환경 결과만 Observation으로 주입해야 한다.

3. 직접 구현은 매우 번거롭다
이 Dummy Agent 구현만 봐도:

- stop 토큰 관리
- JSON 파싱
- Tool 실행
- 메시지 재조립
- 재호출

실수할 지점이 너무 많다

## Agent 라이브러리가 필요한 이유

이 모든 귀찮은 작업을 대신해주는 것이:

- smolagents
- LangGraph
- LlamaIndex

같은 Agent 프레임워크다.

이들은 자동으로:

- Tool 스펙 관리
- Stop & Parse
- Observation 주입
- Loop 제어
- 에러 처리

를 해준다.

## 정리

- Dummy Agent는 Agent의 본질을 이해하기 위한 학습용 구현이다
- System Prompt가 Agent의 행동 규칙을 결정한다
- Stop & Parse 없이는 Observation 환각이 발생한다
- 실제 Agent 구현은 매우 번거롭다
- 그래서 Agent 라이브러리가 존재한다

참고자료
Huggingface, agents course, https://huggingface.co/learn