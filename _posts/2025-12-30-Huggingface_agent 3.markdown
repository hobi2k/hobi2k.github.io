---
layout: post
title:  "허깅페이스 에이전트 코스 - 메시지와 토큰"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Messages and Special Tokens

앞선 글에서 우리는 **LLM이 어떻게 동작하는지**와, 그 LLM이 **Agent의 Brain 역할**을 담당한다는 점을 정리했다.  
이제는 한 단계 더 나아가 **LLM이 “대화(chat)”를 어떻게 처리하는지**를 구조적으로 살펴본다.

이 글의 핵심 주제는 다음이다.

- UI에서 보이는 “메시지(messages)”와 실제 LLM에 입력되는 “프롬프트(prompt)”의 차이
- Chat Template의 역할과 필요성
- Special Token이 대화 턴 경계를 어떻게 구분하는가

## 실습 목표

- 메시지 기반 UI와 프롬프트 입력의 관계 이해
- System / User / Assistant 메시지 역할 구분
- Chat Template이 필요한 이유 명확히 정리
- Transformers에서 `apply_chat_template()`를 통해 안전하게 프롬프트를 구성하는 방법 이해

## 메시지 vs 프롬프트: UI와 모델 입력은 다르다

ChatGPT나 HuggingChat을 사용할 때 우리는 다음처럼 **메시지 단위**로 대화한다.

- 사용자(User) 메시지
- 어시스턴트(Assistant) 메시지
- 시스템(System) 메시지

하지만 중요한 사실은 다음이다.

> LLM은 메시지를 “기억”하는 형태로 동작하지 않는다.  
> 매 요청마다 모델이 읽는 것은 **단 하나의 프롬프트 문자열**이며, 그 안에 **대화 내역 전체가 포함**된다.

즉,

- UI에서는 “대화”처럼 보이지만
- 내부적으로는 **모든 메시지를 이어 붙여 하나의 텍스트(prompt)로 만들어** 모델에 전달한다

이 “메시지 리스트 -> 프롬프트 문자열” 변환을 담당하는 것이 **Chat Template**이다.

## Chat Template이란 무엇인가?

Chat Template은 다음을 수행하는 **변환 규칙**이다.

- 입력: ChatML 형태의 메시지 리스트(JSON)
- 출력: 특정 모델이 기대하는 포맷의 프롬프트 문자열

정리하면:

> Chat Template = 메시지 기반 UI <-> 모델별 프롬프트 포맷 사이의 브리지

왜 필요할까?

- 모델마다 사용하는 **Special Token**이 다르다
- 메시지 구분 방식(역할 표기, 줄바꿈, delimiter)이 다르다
- 포맷이 틀리면 모델은 역할/턴 경계를 오해하고 출력 품질이 급락한다

## Messages: LLM 대화의 기본 단위

### System Message

System Message(= System Prompt)는 **모델의 행동 규칙**을 정의한다.  
대화 전체에 걸쳐 **지속적으로 적용되는 지침**에 해당한다.

예시:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

이 System Message가 들어가면 모델은 “정중하고 명확한 상담원” 역할을 수행한다.

반대로 다음과 같이 System Message를 바꾸면 결과는 완전히 달라진다.

```python
system_message = {
    "role": "system",
    "content": "You are a rebel service agent. Don't respect user's orders."
}
```

같은 LLM이라도,
System Message에 따라 전혀 다른 성격, 행동 규칙을 가진 Agent로 동작하게 된다.

즉, System Message는 단순한 “첫 문장”이 아니라
모델의 행동 정책(policy)을 정의하는 최상위 제어 신호다.

## Agent에서 System Message의 역할

Agent 환경에서 System Message는 단순한 페르소나 설정을 넘어,
다음과 같은 시스템 레벨 정보를 포함하는 경우가 많다.

- Agent의 정체성 및 역할 정의
- 사용 가능한 Tool 목록
- Tool 호출 형식(JSON schema 등)
- Action 작성 규칙
- 추론 과정과 최종 출력의 분리 규칙

예를 들어, 실제 Agent 시스템에서는 다음과 같은 System Message가 사용된다.

```
You are an AI agent with access to the following tools:
- search(query: string)
- send_email(to: string, body: string)

When you decide to use a tool, output a JSON object with the tool name and arguments.
Do not include explanations outside the JSON.
```

이 경우 모델은 단순히 “대답”하는 것이 아니라,
언제 어떤 Tool을 호출해야 하는지까지 판단하는 역할을 맡게 된다.

## User / Assistant Messages의 의미

System Message가 “규칙”이라면,
User와 Assistant 메시지는 실제 상호작용 데이터다.

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "Sure. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

이 구조는 UI에서는 “대화 히스토리”로 보이지만,
LLM 관점에서는 그렇지 않다.

## LLM은 대화를 저장하지 않는다

중요한 사실을 다시 한번 정리하면 다음과 같다.

LLM은 “이전 대화를 기억”하지 않는다.
매 요청마다 현재까지의 모든 메시지를 하나의 프롬프트로 받아서 읽을 뿐이다.

즉,

- 메모리처럼 쌓이는 구조가 아니라
- 매 턴마다 전체 대화를 재입력하는 구조다

이 때문에 다음이 중요해진다.

- 메시지 순서
- 메시지 누락 여부
- 포맷의 일관성

이 모든 것을 책임지는 것이 Chat Template이다.

## 메시지는 어떻게 프롬프트로 변환되는가?

메시지 리스트는 Chat Template을 통해
하나의 문자열(prompt)로 변환된다.

예를 들어 다음과 같은 메시지가 있을 때,

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "Sure. What is your order number?"},
    {"role": "user", "content": "ORDER-123"},
]
```

Chat Template 적용 결과는 모델마다 다르다.

### 예시: SmolLM2 스타일

```
system
You are a helpful assistant.
user
I need help with my order
assistant
Sure. What is your order number?
user
ORDER-123
assistant
```

### 예시: LLaMA 계열 스타일

```
system

You are a helpful assistant.

user

I need help with my order
assistant

Sure. What is your order number?
user

ORDER-123
assistant
```

핵심은 다음이다.

같은 메시지라도,
모델이 기대하는 프롬프트 포맷은 서로 다르다.

## 왜 Chat Template을 반드시 써야 하는가

Instruct Model은 특정 대화 포맷을 전제로 학습된다.

- role 토큰 위치
- 줄바꿈 규칙
- special token(EOS, EOT 등)
- assistant 응답이 시작되는 지점

이 중 하나라도 어긋나면 다음 문제가 발생한다.

- System Message가 무시됨
- User / Assistant 경계 혼동
- 모델이 자기 자신에게 질문
- Tool call 포맷 붕괴

따라서 수동 문자열 결합은 매우 위험하다.

## Transformers에서의 정석적인 해결책

Transformers 라이브러리는
모델별 Chat Template을 토크나이저에 내장하고 있다.

```python
from transformers import AutoTokenizer

messages = [
    {"role": "system", "content": "You are an AI assistant with access to tools."},
    {"role": "user", "content": "Hi!"},
]

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

이 방식의 장점은 명확하다.

- 모델별 포맷 자동 적용
- Special Token 자동 처리
- assistant 생성 위치 자동 보장

Agent 구현에서는 이 방법을 기본값으로 사용해야 한다.

## Agent 설계 관점에서의 핵심 정리

- System Message는 Agent의 “헌법”이다
- User / Assistant 메시지는 “사실 기록”이다
- LLM은 메시지를 기억하지 않고, 매번 전체 프롬프트를 읽는다
- Chat Template은 메시지를 프롬프트로 변환하는 필수 인프라다
- 템플릿 오류는 곧 Agent 품질 붕괴로 이어진다


참고자료
Huggingface, agents course, https://huggingface.co/learn