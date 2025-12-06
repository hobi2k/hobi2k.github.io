---
layout: post
title:  "허깅페이스 스몰 코스 - SmolLM3 파인튜닝"
date:   2025-12-06 00:10:22 +0900
categories: Huggingface_smol
---

# SmolLM3 파인튜닝

이 실습 글은 SmolLM3의 챗 템플릿, 데이터 준비, Supervised Fine-Tuning(SFT),
그리고 TRL CLI 기반의 실전 워크플로우를 정리했다.

SmolLM3는 Hybrid Reasoning 모델로, 상황에 따라 /no_think 또는 /think 모드로
자동 전환해 추론 과정을 다르게 처리하는 것이 특징이다.

이 실습은 실제 LLM 개발자가 수행하는 작업 전체 흐름을 그대로 따라간다.

## 학습 목표

이 실습을 모두 수행하면 다음을 할 수 있게 된다.

- SmolLM3의 Chat Template 적용 방식 이해 및 활용
- Python API + TRL CLI 두 방식 모두로 SFT 수행
- SmolTalk2 및 GSM8K 등 다양한 데이터셋을 일관된 메시지 형식(messages) 으로 변환
- Base 모델 vs Instruct 모델의 응답 차이를 실험적으로 관찰
- LoRA/PEFT 기반 메모리 절약형 SFT 수행
- 모델을 Hugging Face Hub에 배포하는 실전 workflow 이해

## 실습 1: Chat Template 실제 동작 이해

SmolLM3는 모든 대화 입력을 chat template(jinja 기반)로 감싸야
정상적인 instruct-style 응답을 생성한다.
즉, prompt 자체를 모델이 기대하는 구조로 바꿔주는 과정이 반드시 필요하다.

예시

- Base 모델 = 순수 텍스트 이어쓰기(next-token prediction)
- Instruct 모델 = system/user/assistant 구조를 포함한 chat template 기반 동작

이 실습의 핵심은 apply_chat_template() 함수로 메시지를 포맷팅하는 과정을 체계적으로 이해하는 것이다.

환경 설치

```python
pip install "transformers>=4.36.0" "trl>=0.7.0" "datasets>=2.14.0" "torch>=2.0.0"
pip install "accelerate>=0.24.0" "peft>=0.7.0" "trackio"
```

여기서 accelerate는 GPU/CPU 자동 할당 및 분산 설정을 처리하고
peft는 LoRA 학습을 위해 필요하다.

장치 확인

SmolLM3-3B는 최소 8GB VRAM 이상 GPU를 요구한다.

```python
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

모델 로드

```python
base_model_name = "HuggingFaceTB/SmolLM3-3B-Base"
instruct_model_name = "HuggingFaceTB/SmolLM3-3B"

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, dtype=torch.bfloat16, device_map="auto"
)
instruct_model = AutoModelForCausalLM.from_pretrained(
    instruct_model_name, dtype=torch.bfloat16, device_map="auto"
)
```

## Base 모델 vs Instruct 모델 비교하기

SmolLM3에는 두 가지 주요 버전이 있다.


| 모델                        | 설명                                                |
| ------------------------- | ------------------------------------------------- |
| **SmolLM3-3B-Base**       | 순수 언어 모델. 템플릿 미사용 시 의미 없는 이어쓰기 발생 가능              |
| **SmolLM3-3B (Instruct)** | Instruction-following 후처리(Post-training)가 적용되어 있음 |


Chat Template을 적용하지 않은 Base 모델은 보통 질문에 답하지 않고
문장을 계속 이어서 쓰는 경향을 보인다.

반면 Instruct 모델은 아래 형식처럼 system/user 메시지를 기반으로 응답한다.

```python
<|im_start|>system
...
<|im_start|>user
...
<|im_start|>assistant
...
```

## SmolLM3의 Reasoning Mode 탐구

SmolLM3는 다음 두 모드로 동작할 수 있다.


| 모드            | 의미                                        |
| ------------- | ----------------------------------------- |
| **/no_think** | 일반 instruct 생성. reasoning token 사용하지 않음   |
| **/think**    | `<think>...</think>` 태그 내부에서 내부 추론을 먼저 생성 |


Reasoning 태그는 모델의 내부 사고 과정을 포함하지만
final answer는 태그 밖에서 추출된다.

훈련 시 이 태그 포맷을 잘못 처리하면 학습이 오염되므로
데이터 전처리에서 반드시 유지, 정확히 분리해야 한다.

## 실습 2: 데이터셋 전처리 — 메시지 형식 통일하기

SFT에서는 다음과 같은 단일 규격 메시지 구조를 만들 필요가 있다.

```python
messages = [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

그러나 현실에서는 다양한 데이터셋이 서로 다른 형식으로 제공된다.

예시

- GSM8K: question, answer
- Instruction datasets: instruction, response
- Dialogue datasets: 여러 turns의 messages

이 실습에서는 서로 다른 데이터셋을 동일한 messages 구조로 변환하는 방법을 배운다.

핵심 함수

```python
tokenizer.apply_chat_template(messages, tokenize=False)
```

이 함수가 모델에 전달할 최종 학습용 텍스트(text)로 변환한다.

SmolTask2 데이터로드

```python
dataset_dict = load_dataset("HuggingFaceTB/smoltalk2", "SFT")
```

메시지 구조

```python
{
  "messages": [
     {"role": "...", "content": "..."},
     ...
  ]
}
```

외부 데이터의 전처리

```python
def process_gsm8k(examples):
    messages = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
```

### Chat Template 적용

모든 messages는 학습 전에 chat template을 적용한 text 문자열로 변환해야 한다.

```python
def apply_chat_template_to_dataset(dataset, tokenizer):
    def format_messages(examples):
        texts = []
        for m in examples["messages"]:
            t = tokenizer.apply_chat_template(m, tokenize=False)
            texts.append(t)
        return {"text": texts}
```

SFTTrainer는 "text" 컬럼만 읽기 때문에
messages 컬럼을 제거하고 text만 남긴다.



## 실습 3: SmolLM3 Fine-Tuning

Fine-tuning은 TRL의 SFTTrainer를 사용한다.
이 Trainer는

- Chat template 자동 적용
- Packed training(optional)
- Gradient accumulation
- FP16/BF16 메모리 최적화

등을 자동으로 처리한다.

### 중요한 설정

1. dataset_text_field="text"

학습 데이터셋은 반드시 "text" 컬럼만 있어야 한다.

2. Tokenizer 설정

```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

Right padding을 쓰는 이유는
LLM의 causal mask 특성 상 좌측 padding은 학습 효율을 떨어뜨리기 때문이다.

3. 학습 설정에서 가장 핵심 파라미터

- max_length
- per_device_train_batch_size
- gradient_accumulation_steps
- learning_rate

SmolLM3-3B 기준 최소 GPU 요구량

- 8GB: LoRA 필수
- 12–16GB: Full SFT도 가능하나 batch size 제한 있음

### SFT 설정

```python
training_config = SFTConfig(
    output_dir="./SmolLM3-Custom",
    dataset_text_field="text",
    max_length=2048,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=1,
)
```

## LoRA/PEFT 기반 SFT

GPU가 부족하다면 PEFT(특히 LoRA)를 사용해야 한다.

LoRA는 다음 구조로 동작한다.

```python
W' = W + BA
```

- W: 동결된 원본 weight
- B,A: rank r의 저차원 업데이트 행렬

이 방식으로 전체 파라미터의 1~5%만 학습하면 되므로
VRAM 및 속도 면에서 압도적으로 유리하다.

### LoRA 설정 예시

```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

일반적으로 r=4~16 사이가 가장 안정적이다.

### 훈련

```python
trainer = SFTTrainer(model, train_dataset, args=training_config, peft_config=peft_config)
trainer.train()
```

## 실습 4: TRL CLI 기반 생산 환경 학습

실제 기업/팀 환경에서는 Python API보다 CLI 기반 학습이 효율적이다.

CLI 사용

```python
trl sft \
  --model_name_or_path HuggingFaceTB/SmolLM3-3B-Base \
  --dataset_name HuggingFaceTB/smoltalk2_everyday_convs_think \
  --output_dir ./smollm3-sft-cli
```

추가로 YAML 설정 파일을 사용하면
재현성(Reproducibility)이 크게 향상된다.

## 실전 Troubleshooting

다음은 실제 실습에서 가장 자주 발생하는 문제와 해결법이다.

1. GPU OOM(Out of Memory)

- batch size를 1로 낮춘다
- max_length를 512~1024로 줄인다
- LoRA 사용

2. 모델 로딩 실패

- 인터넷 문제
- trust_remote_code=True 필요 여부 확인
- device_map을 CPU로 임시 로드

3. 학습 불안정

- 데이터 text 길이 불균형 -> group_by_length=True 활용
- 학습 데이터에 잘못된 구조 포함 -> messages 포맷 점검

참고자료
Huggingface, a smol course, https://huggingface.co/learn