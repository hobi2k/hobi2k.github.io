---
layout: post
title:  "허깅페이스 에이전트 코스 - What is Function Calling?"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# What is Function Calling?

## 정의: “LLM이 환경에 행동(Action)을 취하는 표준 인터페이스”

**Function Calling**은 LLM이 단순히 텍스트를 생성하는 것을 넘어, 외부 세계(도구/시스템/DB/API 등)에 **구체적 행동을 요청**할 수 있게 하는 메커니즘이다.  
OpenAI의 GPT-4에서 널리 알려졌고, 이후 다양한 모델/플랫폼(Mistral 등)에서도 유사한 형태로 확산되었다.

- 에이전트의 “Tool 사용”과 같은 목적(환경에 영향)  
- 차이점은 **“도구 호출 형식과 타이밍”을 모델이 학습(파인튜닝)으로 내재화**했다는 점

즉, 프롬프트로 “도구를 써라”를 강하게 유도하는 방식보다,  
모델 자체가 **‘필요 시 도구 호출’이라는 행동 양식을 더 직접적으로 체득**하도록 만든다.

## Tool-Using Agent vs Function Calling의 차이

### (A) 일반적인 Tool-Using Agent
- 우리가 도구 목록을 제공해두고,
- 모델이 “계획 -> 도구 선택 -> 호출”을 **일반화 능력**으로 수행하도록 기대한다.
- 도구를 어떻게/언제 쓸지의 핵심은 **프롬프트 설계 + 에이전트 런타임 로직**에 크게 의존한다.

### (B) Function Calling 기반 에이전트
- 모델이 **도구 호출을 위한 구조화된 출력(예: JSON arguments, tool name)을 생성**하도록 학습되어 있다.
- 프롬프트 유도보다 **학습된 행동(형식/역할/토큰)**에 더 의존한다.
- “언제 도구를 호출해야 하는지”를 포함해 **도구 사용 패턴 자체가 모델의 능력으로 편입**된다.

핵심 요약:
- Tool-Using(프롬프트 중심): “도구를 쓸 수 있다”를 **설명하고 유도**해서 쓰게 함  
- Function Calling(학습 중심): “도구를 쓰는 법”을 **모델이 학습해서** 자연스럽게 호출 형식을 출력함

## 모델은 어떻게 “행동”을 학습하는가? (Think -> Act -> Observe 루프)

에이전트의 전형적인 동작 루프는 다음과 같다.

1. **Think**: 목표 달성을 위해 어떤 행동/도구가 필요한지 결정  
2. **Act**: 도구 호출을 위한 형식으로 출력(이때 텍스트 생성을 멈추고 “호출 포맷”을 냄)  
3. **Observe**: 도구 실행 결과(관측)를 입력으로 다시 받아 후속 진행

Function Calling은 이 루프를 “채팅 메시지 역할(role)” 또는 “특수 토큰” 체계로 **프로토콜화**한다.

## 일반 채팅과 Function Calling 채팅의 메시지 구조 비교

### (A) 일반적인 대화 (User ↔ Assistant)
```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

메시지 역할(role)이 보통 user, assistant만으로 충분하다.
“행동”이 아니라 “대화” 자체가 목적이기 때문.

### (B) Function Calling이 들어간 대화 (Action/Observation 단계 추가)
Function calling은 대화에 “행동”과 “관측” 개념이 들어가므로,
도구 호출/결과를 담는 메시지 구조가 추가된다.

Mistral API 스타일 예시:

```python
conversation = [
    {
        "role": "user",
        "content": "What's the status of my transaction T1001?"
    },
    {
        "role": "assistant",
        "content": "",
        "function_call": {
            "name": "retrieve_payment_status",
            "arguments": "{\"transaction_id\": \"T1001\"}"
        }
    },
    {
        "role": "tool",
        "name": "retrieve_payment_status",
        "content": "{\"status\": \"Paid\"}"
    },
    {
        "role": "assistant",
        "content": "Your transaction T1001 has been successfully paid."
    }
]
```

이 흐름을 단계별로 해석하면:

- user: 문제 제기(질문)
- assistant (function_call 포함): Action = 어떤 함수(도구)를 어떤 인자로 호출할지 선언
- tool: Observation = 호출 결과를 시스템이 되돌려줌
- assistant: 관측 결과를 반영해 최종 답변 생성

### “새 role이 있다면서요?”에 대한 정확한 답
문서에서 말하는 “새로운 역할(role)”은 개념적으로는 맞다.
하지만 구현은 API마다 차이가 있다.

어떤 시스템은 도구 호출을 assistant 메시지에 특수 필드(function_call)로 담는다.
그리고 tool 결과는 role="tool"로 별도 메시지로 돌아온다.

내부적으로는 채팅 템플릿(chat template)이 이를 특수 토큰으로 직렬화해서 모델 입력에 넣는다.

대표적인 특수 토큰 예시:

- [AVAILABLE_TOOLS] / [/AVAILABLE_TOOLS] : 사용 가능한 도구 목록 구간
- [TOOL_CALLS] : 도구 호출(Action) 선언 구간
- [TOOL_RESULTS] / [/TOOL_RESULTS] : 도구 결과(Observation) 구간

즉, “role이 추가된다”는 건

- 외부 API 수준에서는 tool 같은 role로 드러나기도 하고,
- 모델 내부 입력에서는 특수 토큰/템플릿으로 표현되는 경우가 많다.

## 다음 단계: Function Calling을 ‘없는 모델’에 붙이려면?

Function calling 능력이 없는 모델(예: google/gemma-2-2b-it)에
**특수 토큰 + 학습(파인튜닝/LoRA)**을 통해
"도구 호출 형식"을 출력할 수 있는 능력을 부여하는 것

참고자료
Huggingface, agents course, https://huggingface.co/learn