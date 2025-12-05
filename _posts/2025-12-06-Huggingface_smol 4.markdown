---
layout: post
title:  "허깅페이스 스몰 코스 - LoRA와 PEFT"
date:   2025-12-06 00:10:22 +0900
categories: Huggingface_smol
---

# LoRA와 PEFT

대규모 언어 모델(LLM)을 그대로 풀파인튜닝(full fine-tuning)하려면 엄청난 GPU 메모리와 시간이 필요하다.
**Parameter-Efficient Fine-Tuning(PEFT)**는 이런 문제를 해결하기 위해 등장한 방법으로,
기존 베이스 모델의 대부분은 그대로 두고, 극히 일부의 추가 파라미터만 학습하도록 만든 기법들의 집합이다.

그중 가장 널리 쓰이는 방법이 바로 **LoRA(Low-Rank Adaptation)**이다.
LoRA는 선형층(linear layer)에 저랭크(low-rank) 업데이트 행렬을 삽입해서,
학습해야 하는 파라미터 수를 약 90% 이상 줄이면서도 성능을 유지하거나 가끔은 개선까지 한다.

## 언제 PEFT를 써야 할까

다음 중 여러 항목에 해당한다면, PEFT(특히 LoRA)를 고려할 만하다.

- 사용 가능한 GPU 메모리(VRAM)가 넉넉하지 않다.
- 하나의 베이스 모델을 여러 도메인/태스크에 빠르게 적응시키고 싶다.
- 다양한 하이퍼파라미터 실험을 자주 돌려야 한다.
- 학습 결과물을 “작은 파일(몇 MB)” 단위로 관리하고 싶다.
- 이미 잘 만든 베이스 모델(gemma, smollm3, mistral 등)을 존중하면서, 그 위에만 살짝 얹고 싶다.

반대로,

- 처음부터 모델을 새로 만들거나,
- 연구 목적으로 전체 architecture를 크게 바꾸고 싶다면

그때는 여전히 full fine-tuning이 필요하다.

## LoRA

### 풀파인튜닝과의 차이

일반 풀파인튜닝에서는,
어떤 선형층의 weight를 W라고 할 때, 학습 과정에서 W 전체를 업데이트한다.

LoRA에서는 이렇게 하지 않는다. 대신,

- W(베이스 가중치)는 동결(freeze)
- 업데이트 부분 ΔW만 따로 작은 두 개의 행렬 B, A로 표현
- 그리고 B, A만 학습한다.

텍스트로 표현하면 대략 이렇게 쓸 수 있다.

- W' = W + BA
- 여기서 W'는 LoRA 적용 후 weight
- B, A는 저랭크(low-rank) 행렬 (rank r << 원래 차원)

즉, LoRA는 “W 전체를 바꾸는 대신 BA만 더해서 바꾼 효과를 내자”라는 아이디어다.

### 파라미터 수 감소

원래 선형층의 weight 차원을 d x k 라고 하자.

- 풀파인튜닝에서 학습하는 파라미터 수:

`d * k`

- LoRA에서 학습하는 파라미터 수:

`r * (d + k)`

여기서 r은 rank인데, 보통 4, 8, 16 정도의 작은 값이다.

따라서 r이 d, k에 비해 아주 작기 때문에,
실제로 학습되는 파라미터 수는 원래의 1~10% 수준에 불과하다.

논문 기준으로는 GPT-3 175B 모델에 LoRA를 적용했을 때,

- 학습 파라미터 수: 약 10,000배 감소
- GPU 메모리 사용량: 약 3배 감소

라는 결과가 보고되었다.

자세한 내용은 다음 논문을 참고하면 좋다.

LoRA 논문: https://huggingface.co/papers/2106.09685

### Forward Pass 관점

Forward Pass 입장에서 보면, LoRA는 단순히 다음과 같은 형태로 동작한다.

- 기존 출력: output = W x
- LoRA 적용: output = W x + (BA) x

기존 모델은 그대로 두고, BA x만 “추가 토핑”처럼 얹어주는 구조라고 보면 된다.

## LoRA는 어디에 붙는가? (Attention 중심)

Transformer 기반 LLM에서 LoRA는 보통 다음 위치에 많이 붙는다.

- Self-Attention의 Query, Key, Value, Output projection
- 또는 이 중 일부(Q, V)만 선택적으로 적용

예를 들어 Q, V projection에만 적용하는 경우, 개념적으로 다음과 같다.

- Q' = Q + (B_Q * A_Q)
- V' = V + (B_V * A_V)

즉, 각 projection layer에 작은 LoRA 모듈이 하나씩 달려 있고,
그 모듈만 학습되는 구조라고 이해하면 된다.

## LoRA Adapter: 로딩과 스위칭

LoRA는 “베이스 모델 + 어댑터(adapter)” 구조를 쓰기 때문에,
하나의 베이스 모델에 여러 개의 작업용 어댑터를 얹어놓고 바꿔가면서 쓸 수 있다는 점이 매우 강력하다.

### 기본 로딩 예시

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"

model = PeftModel.from_pretrained(base_model, peft_model_id)
```

이렇게 하면, 

- base_model은 그대로 두고
- LoRA adapter 가중치만 추가로 로드한 model을 얻게 된다.

일반적으로는,

- base 모델 하나 로드
- 태스크마다 adapter를 하나씩 준비
- 상황에 따라 adapter만 교체

하는 식으로 사용한다.

### 어댑터 전환

PEFT/LoRA에서는 보통 다음과 같은 패턴을 쓸 수 있다.

- set_adapter("name") : 특정 adapter 활성화
- unload() : 모든 LoRA 모듈 제거, 순수 base 모델로 복귀

즉, “도메인 A용 adapter, 도메인 B용 adapter”를 여러 개 만들어두고
필요한 것만 골라 끼우는 방식으로 운용이 가능하다.

## LoRA Adapter 병합(Merging)

학습 단계에서는 보통 “베이스 + adapter” 구조로 두지만,
배포(deployment)에 편의를 위해 둘을 합쳐서 하나의 모델로 만들 수도 있다.

이 과정을 “merge”라고 한다.

### 왜 병합하는가

- 추론 서버에서 adapter를 별도로 로드할 필요가 없다.
- 단일 weight 파일만 관리하면 된다.
- Latency 관점에서 추가적인 연산 비용이 없다(실질적으로 W에 BA를 더한 결과만 사용).

### 병합 시 주의할 점

- 베이스 모델과 adapter를 동시에 메모리에 올려야 하므로 GPU 혹은 CPU 메모리가 충분한지 확인해야 한다.
- 학습했던 precision과 동일한 dtype을 유지하는 것이 좋다. (예: float16, bfloat16 등)
- 병합 후에는 반드시 출력 검증을 해야 한다. (adapter 방식일 때와 결과가 비슷한지, 성능(정확도/품질)이 유지되는지 확인 필요)

### 병합 구현 예시

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Base 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    dtype=torch.bfloat16,
    device_map="auto"
)

# 2. LoRA 어댑터 로드
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    dtype=torch.bfloat16
)

# 3. 병합 시도
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # 메모리 최적화나 CPU offload 등을 고려

# 4. 병합된 모델 저장
merged_model.save_pretrained("path/to/save/merged_model")

# 5. 토크나이저도 함께 저장
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## OLoRA: LoRA 초기화의 개선판

**OLoRA(Orthogonal LoRA)**는 기존 LoRA보다 학습 안정성을 개선하기 위한 변형 기법이다.
핵심 아이디어는 다음과 같다.

- Base weight를 QR 분해와 같은 형태로 분해
- 그 결과를 이용해 LoRA 어댑터 초기값을 더 “좋은 위치”에 잡는다

이렇게 하면:

- 학습 초기에 더 안정적
- 수렴 속도가 빨라질 수 있음
- 최종 성능도 LoRA보다 조금 더 나아지는 경우가 많음

즉, OLoRA는 “어댑터를 어떻게 초기화하느냐”를 개선한 LoRA의 변형이라고 이해하면 된다.

논문: https://huggingface.co/papers/2406.01775

## TRL과 PEFT 결합하기

**TRL(Transformers Reinforcement Learning)**은 Hugging Face에서 제공하는
SFT, DPO, PPO 등 alignment 작업용 라이브러리다.

TRL은 PEFT/LoRA와 자연스럽게 통합되도록 설계되어 있어서,
SFTTrainer에 LoRA 설정(peft_config)만 넘겨주면 자동으로 PEFT 기반 SFT를 수행할 수 있다.

### 기본 SFT + LoRA 예시

```python
from peft import LoraConfig
from trl import SFTTrainer

# 1. LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 2. Trainer 생성
trainer = SFTTrainer(
    model="your-model-name",
    train_dataset=dataset["train"],
    peft_config=lora_config,
)

# 3. 학습 시작
trainer.train()
```


여기서 중요한 점은,

- model에 LoRA를 직접 씌우지 않고
- peft_config만 넘겨주면,
- SFTTrainer가 내부에서 알아서 PEFT를 설정해 준다는 것이다.

## TRL + LoRA Quick Start 예시

조금 더 구체적인 설정 예시는 아래와 같다.

```python
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1) LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 2) SFT 설정
sft_config = SFTConfig(
    output_dir="lora-adapter",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    packing=True,
)

# 3) Trainer 생성
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    peft_config=peft_config,
)

# 4) 학습
trainer.train()
```

학습이 끝난 뒤에는 선택지가 두 가지 있다.

1. 어댑터 방식으로 사용

- base 모델 + adapter 를 함께 로드
- 여러 태스크 adapter를 동시에 관리하기 좋음

2. 병합 후 하나의 모델로 사용

- merge_and_unload()로 병합
- 배포 환경에서 단순함, latency 최소화

## 실전 팁: LoRA/PEFT 도입할 때 고려할 점

- rank(r)는 처음에는 4~8 정도의 작은 값부터 시작하는 것이 좋다.
- validation loss로 과적합 여부를 항상 확인해야 한다.
- 가능하다면 full fine-tuning 또는 기존 instruct 모델과 성능을 비교해 보는 것이 좋다.
- 8bit/4bit quantization과 LoRA(특히 QLoRA)를 함께 사용하면 VRAM을 극도로 줄이면서도 훈련이 가능하다.
- LoRA를 적용할 모듈(q, k, v, o, ffn 등)을 어디까지 넓힐지는 태스크 난이도와 리소스에 따라 조정할 수 있다.

## 마무리 요약

- PEFT는 “베이스 모델은 그대로, 작은 추가 파라미터만 학습하는” 접근이다.
- LoRA는 그중 가장 널리 쓰이는 방법으로, 저랭크 업데이트 BA를 통해 파라미터 수를 크게 줄인다.
- 어댑터(adapter) 구조 덕분에 하나의 베이스 모델로 여러 태스크를 쉽게 관리할 수 있다.
- 필요하면 어댑터를 base 모델에 병합하여 단일 모델로 배포할 수 있다.
- OLoRA는 LoRA 초기화를 개선하여 학습 안정성과 성능을 높이는 변형이다.
- TRL + LoRA 조합은 SFT, RLHF 등 현대 LLM 미세조정 워크플로우에서 사실상 표준에 가깝다.

참고 자료

LoRA: Low-Rank Adaptation of Large Language Models
https://huggingface.co/papers/2106.09685

PEFT Documentation
https://huggingface.co/docs/peft

Hugging Face Blog – PEFT 소개
https://huggingface.co/blog/peft

Huggingface, a smol course, https://huggingface.co/learn