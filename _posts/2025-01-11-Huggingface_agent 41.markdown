---
layout: post
title:  "허깅페이스 에이전트 코스 - Function Calling 모델 파인튜닝 전체 코드 해설"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# Function Calling 모델 파인튜닝 전체 코드 해설

이 글은 `bonus_unit1.ipynb`에 포함된 
**Function Calling 파인튜닝 코드** 전체를 해설한 문서이다.

목표는 다음과 같다.

- Function Calling이 **프롬프트가 아니라 학습 대상**임을 이해
- LoRA가 **어떤 파라미터를, 왜 학습하는지** 명확히 이해
- 이후 **LangGraph + Function Calling LLM**으로 확장 가능한 기반 확보

## 0. 필수 라이브러리 임포트

```python
import torch
```

- PyTorch 메인 프레임워크
- 모델 로딩, 학습, 텐서 연산에 사용

```python
from datasets import load_dataset
```

- Hugging Face datasets
- function-calling 학습 데이터셋 로딩용

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
```

- Hugging Face Transformers 핵심 구성요소
- tokenizer / base LLM / 학습 설정 관리

```python
from peft import LoraConfig, get_peft_model
```

- LoRA(저랭크 어댑터) 설정 및 적용
- base model 가중치는 고정, adapter만 학습

```python
from trl import SFTTrainer
```

- Supervised Fine-Tuning 전용 Trainer
- “정답이 있는 출력 포맷 학습”에 최적

## 1. 베이스 모델 선택

```python
model_id = "google/gemma-2-2b-it"
```

왜 instruction-tuned 모델인가?

- 이미 대화 구조 / 명령 이해가 학습되어 있음

우리는 언어 능력이 아닌 함수 호출 행동 패턴만 추가한다.

## 2. 토크나이저 로드
```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

- Gemma 전용 토크나이저 로드
- 이후 특수 토큰 추가 예정

## 3. Function Calling용 특수 토큰 정의
```python
from enum import Enum
```

- 특수 토큰을 명시적으로 관리하기 위한 Enum

```python
class ChatmlSpecialTokens(Enum):
    think = "<think>"
    tool_call = "<tool_call>"
    tool_response = "<tool_response>"
```

이 Enum의 진짜 의미

- <think> -> 사고 단계 시작
- <tool_call> -> 행동(Action) 출력
- <tool_response> -> 관측(Observation)

이 문자열들은 일반 텍스트가 아니라 “문법 기호”

## 4. 토크나이저에 특수 토큰 등록

```python
special_tokens_dict = {
    "additional_special_tokens": [token.value for token in ChatmlSpecialTokens]
}
```
Enum에 정의한 모든 특수 토큰을 tokenizer에 추가

```python
tokenizer.add_special_tokens(special_tokens_dict)
```

이 시점 이후:

- <think>는 하나의 토큰
- 쪼개지지 않음
- 모델이 구조 신호로 인식 가능

## 5. 베이스 모델 로드

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

- bfloat16 -> 메모리 절약
- device_map="auto" -> GPU 자동 분산
- base model 가중치는 이후 고정됨

```python
model.resize_token_embeddings(len(tokenizer))
```

- 특수 토큰 추가로 vocab 크기 변경
- embedding layer 크기 동기화 필수

## 6. Function Calling 데이터셋 로드
```python
dataset = load_dataset("NousResearch/hermes-function-calling-v1")
```

- 이미 함수 호출 예제가 포함된 데이터셋
- 단, thinking 과정이 없음
> 직접 추가해야 함

## 7. 데이터 전처리 함수 정의
```python
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"


def preprocess(sample):
      messages = sample["messages"]
      first_message = messages[0]

      # Instead of adding a system message, we merge the content into the first user message
      if first_message["role"] == "system":
          system_message_content = first_message["content"]
          # Merge system content with the first user message
          messages[1]["content"] = system_message_content + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n" + messages[1]["content"]
          # Remove the system message from the conversation
          messages.pop(0)

      return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

- 이 함수는 이 노트북 전체에서 가장 중요하다.
- 단순 포맷 변경이 아니라 “모델이 어떤 사고 구조를 배울지”를 결정한다.

### 7-1. system 메시지 병합
```python
messages = []
```

- 최종적으로 LLM에 입력될 메시지 리스트

```python
system_prompt = (
    "You are a helpful assistant with access to the following tools.\n\n"
    "Before calling a tool, think carefully about which tool to use.\n"
    "Write your reasoning between <think> and </think>."
)
```

- Gemma는 system role을 직접 지원하지 않음
- user 메시지 안에 system 역할 병합

### 7-2. tools 정의 삽입
```python
tools = sample["tools"]
```
- 함수 스키마(JSON)
- 모델은 이 스키마를 보고:
  - 어떤 함수가 있는지
  - 어떤 인자를 요구하는지 학습

```python
user_message = f"{system_prompt}\n\nAvailable tools:\n{tools}\n\nUser query:\n{sample['query']}"
```

- system + tools + query를 하나의 user 메시지로 통합
- chat template 제약 때문

### 7-3. assistant 출력 구성
```python
assistant_message = (
    f"<think>\n{sample['thoughts']}\n</think>\n"
    f"<tool_call>\n{sample['tool_call']}\n</tool_call>"
)
```

이 부분이 학습의 핵심
모델은 여기서 다음을 배운다:

- 먼저 <think>로 사고
- 그 다음 <tool_call>로 행동
- 순서를 어기면 틀린 답

### 7-4. 최종 포맷 반환
```python
return {
    "text": tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        tokenize=False,
    )
}
```

- Gemma 전용 chat template 적용
- 학습 데이터는 이미 완성된 정답 포맷

## 8. 데이터셋 매핑
```python
processed_dataset = dataset["train"].map(preprocess)
```

- 모든 샘플에 동일한 사고/행동 구조 강제
- Function Calling은 “학습된 습관”이 됨

## 9. LoRA 설정
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head",
    ],
)
```
각 파라미터 의미
- r=16: 학습 가능한 저차원 공간 크기
- target_modules
  - Attention + FFN + 출력 헤드
  - 사고 + 결정 + 출력 전부에 영향

## 10. LoRA 적용
```python
model = get_peft_model(model, lora_config)
```

- base model 가중치 freeze
- adapter만 학습

```python
model.print_trainable_parameters()
```

- 실제 학습되는 파라미터 수 확인
- 보통 1~2% 수준

## 11. 학습 설정
```python
training_args = TrainingArguments(
    output_dir="./fc_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    report_to="none",
)
```
- batch size 작음 -> accumulation으로 보완
- epoch 1 -> 구조 학습에는 충분

## 12. SFTTrainer 초기화

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=processed_dataset,
    args=training_args,
    packing=True,
)
```
- packing=True -> 여러 샘플을 하나의 시퀀스로 묶음

작은 데이터셋에서 매우 중요

### 13. 학습 시작
```python
trainer.train()
```

- 이 시점에서 모델은 학습한다:
    - 언제 <think>를 써야 하는지
    - 언제 <tool_call>을 출력해야 하는지
    - 함수 이름과 인자 포맷

## 14. 최종 요약

이 코드는 단순한 SFT가 아니다.
“LLM에게 행동 문법(Action Grammar)을 가르치는 코드”


참고자료
Huggingface, agents course, https://huggingface.co/learn