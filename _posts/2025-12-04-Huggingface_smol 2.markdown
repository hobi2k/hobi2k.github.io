---
layout: post
title:  "허깅페이스 스몰 코스 - 쳇 탬플릿"
date:   2025-12-04 00:10:22 +0900
categories: Huggingface_smol
---

# 쳇 탬플릿

쳇 탬플릿은 대규모 언어 모델이 대화의 흐름을 이해하는 데 필요한 규칙을 제공한다.
인스트럭션 튜닝에서는 데이터셋의 품질뿐만 아니라, 모델이 학습하는 대화 형식의 일관성이 성능에 직접적으로 영향을 준다.
이 글에서는 SmolLM3의 Chat Template 구조를 중심으로 대화 데이터 방식, 파이프라인 사용법, 고급 기능을 설명한다.

## Base Model과 Instruct Model의 차이

먼저, Chat Template이 왜 필요한지를 이해하기 위해 Base 모델과 Instruct 모델의 차이를 살펴보아야 한다.

**Base Model**

SmolLM3-3B-Base

- 사전학습 과정에서 next-token prediction을 학습한다.
- 입력 문장을 자연스럽게 이어가는 데는 능하지만, 대화 규칙이나 질의응답 형식은 이해하지 못한다.

**Instruct Model**

SmolLM3-3B

- Instruction Tuning(SFT)을 거쳐 “질문에는 대답”을, “요청에는 작업 수행”을 학습한다.
- 역할(role) 구분, 메시지 경계, 대화 맥락 유지 등을 이해한다.

이 변환 과정에서 가장 중요한 요소가 쳇 탬플릿이다.

## Chat Template이 하는 일

Chat Template은 다음과 같은 기능을 수행한다.

- 사용자(user), 시스템(system), 어시스턴트(assistant) 역할을 구분
- 메시지의 시작과 끝을 명확히 표시
- 모델이 대화 기록을 순서대로 이해하도록 구조화
- 외부 도구(tool) 호출을 표현
- Reasoning Mode 등 특수 기능을 활성화

SmolLM3는 업계 표준으로 자리잡은 ChatML 형식을 사용한다.

## Pipeline을 통한 자동 템플릿 적용

Transformers의 pipeline API는 쳇 탬플릿을 자동 처리한다.
즉, 템플릿을 직접 작성할 필요 없이 모델을 자연스럽게 대화형으로 사용할 수 있다.

```python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceTB/SmolLM3-3B", device_map="auto")

messages = [
    {"role": "system", "content": "Respond in the style of a pirate."},
    {"role": "user", "content": "How many helicopters can a human eat?"}
]

response = pipe(messages, max_new_tokens=128)
print(response[0]["generated_text"][-1])
```

Pipeline은 다음 작업을 자동으로 수행한다.

- 모델에 맞는 쳇 탬플릿 자동 적용
- Message-to-text 변환
- Tokenization
- 생성 결과를 다시 role 기반 구조로 반환

Pipeline은 프로덕션 환경에서 특히 유용하다.

## ChatML 구조 이해하기

SmolLM3의 Chat Template은 ChatML 기반이다.
실제 메시지는 다음과 같은 구조로 변환된다.

```python
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
```

**구성 요소 설명**

- `<|im_start|>`: 메시지 시작
- system, user, assistant: 역할(role)
- `<|im_end|>`: 메시지 종료
- `<|thinking|>`: reasoning 모드 시작

이 구조는 추론과 학습 둘 다에서 중요한 역할을 한다.

## Dual Reasoning Mode

SmolLM3는 두 가지 동작 모드를 지원한다.

1. Standard Mode

직접적인 정답만 응답한다.

2. Thinking Mode

`<|thinking|>` 블록을 사용해 reasoning을 먼저 생성한 뒤 답변을 제공한다.

```python
<|thinking|>
Detailed reasoning here.
</|thinking|>

Final answer here.
```

이 기능은 수학, 코드, 논리적 문제 해결에 유용하다.

## 쳇 탬플릿 직접 적용하기

tokenizer.apply_chat_template()를 사용하면 템플릿이 실제로 어떻게 문자열로 변환되는지 확인할 수 있다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

messages = [
    {"role": "system", "content": "Technical assistant."},
    {"role": "user", "content": "Explain chat templates."},
    {"role": "assistant", "content": "Chat templates define conversation structure."}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(formatted)
```

## 중요한 파라미터: add_generation_prompt

언제 사용해야 하는가?

- 학습 데이터 준비 시: False
- 모델 추론(inference) 시: True
- 평가 시: True

이 설정을 잘못 사용하면 모델이 역할을 혼동하거나, 사용자 메시지를 계속 이어서 잘못된 답변을 생성할 수 있다.

## continue_final_message: 응답 프리필(prefill)

모델에게 “해당 메시지를 이어서 작성하라”고 명령하는 기능이다.

예시: JSON 구조 완성, 코드 completion, reasoning step-by-step 등

```python
messages = [
    {"role": "user", "content": "Write JSON."},
    {"role": "assistant", "content": '{"name": "'}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    continue_final_message=True
)
```

이 기능은 고정된 형식을 요구하는 작업에서 특히 유용하다.

## Tool Calling (Function Calling)

SmolLM3는 함수 호출을 지원하며, Chat Template에 tool 호출 기록이 포함될 수 있다.

예시: 함수 정의

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]
```

예시: 메시지에서 tool 호출

```python
{
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}'
            }
        }
    ]
}
```

Tool 사용 여부는 템플릿 내부에서 자동 처리된다.

## Template Debugging

쳇 탬플릿 문제는 모델 오류의 주요 원인이므로, 디버깅이 중요하다.

```python
def debug_chat_template(messages, tokenizer):
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(formatted, return_tensors="pt")
    
    print("Formatted text:")
    print(repr(formatted))
    print("Token count:", tokens["input_ids"].shape[1])
```

## Chat Template 사용 시 주의할 점

1. 템플릿 불일치

훈련에 사용된 템플릿과 추론에 사용하는 템플릿이 다르면 모델 성능이 크게 저하된다.

2. 특수토큰 중복

`<|im_start|>` 등을 직접 넣어서는 안 된다. 템플릿에서 자동 처리된다.

3. 구조 혼합

다른 형식의 대화를 섞어서 SFT 데이터셋을 구성하면 모델이 규칙을 혼동한다.

4. system message 부족

모델 스타일이 흔들리거나 일관되지 않게 답한다.

## 핵심 정리

- 쳇 탬플릿은 instruction-tuned 모델의 성능을 좌우한다.
- SmolLM3는 ChatML 기반 구조를 사용한다.
- add_generation_prompt는 추론에서 반드시 필요하다.
- continue_final_message는 구조화된 출력 생성에 유용하다.
- Reasoning Mode는 `<|thinking|>`로 제어한다.
- Tool calling은 현대 LLM에서 필수 기능이며 쳇 탬플릿에 통합되어 있다.
- 템플릿 불일치나 특수토큰 오사용은 모델 품질을 크게 낮춘다.


참고자료
Huggingface, Audio Course, https://huggingface.co/learn