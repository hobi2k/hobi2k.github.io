---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 음악 분류를 위한 모델 파인튜닝"
date:   2025-11-26 00:10:22 +0900
categories: Huggingface_Audio
---

# 음악 분류를 위한 모델 파인튜닝

이 글에서는 음악 분류(Music Classification) 작업을 위해 Encoder-only Transformer 기반 음성 모델을 파인튜닝하는 전체 과정을 다룬다.

특히 DistilHuBERT 같은 경량 모델을 사용하여, Google Colab(T4 16GB)이나 일반 소비자 GPU에서도 실행 가능한 형태로 구성되어 있다.

또한 데이터 전처리, feature extractor 구조, sampling rate 문제, normalization, batch memory 최적화 등 실전에서 부딪히는 포인트들을 함께 설명한다.

## 데이터셋: GTZAN

음악 장르 분류(Music Genre Classification)에서 가장 널리 사용되는 데이터셋 중 하나가 GTZAN이다.

- 총 1,000곡(실제로는 999곡. 1곡은 손상되어 제거됨)
- 각 곡은 10개 장르 중 하나
(blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

### 데이터 불러오기

```python
from datasets import load_dataset
gtzan = load_dataset("marsyas/gtzan", "all")
```

출력

```python
Dataset(features: ['file','audio','genre'], num_rows: 999)
```

### Train/Validation 분할

GTZAN에는 기본 validation 분할이 없으므로 직접 분할해야 한다.
데이터가 장르별로 균형 잡혀 있으므로 단순 90/10 비율 split이 적합하다.

```python
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
```

### 샘플 구조 이해

GTZAN의 오디오 파일은 다음과 같이 구성된다:

```python
{
  "file": ".../pop.00098.wav",
  "audio": {
      "array": [...],
      "sampling_rate": 22050
  },
  "genre": 7
}
```

### 중요한 특징

1. Raw Audio Array

1차원 float32 배열이며, 각 값이 “특정 시점의 진폭(amplitude)”이다.

2. Sampling Rate = 22,050 Hz

1초에 22,050개의 샘플을 기록한다는 의미.
모델이 요구하는 sampling rate가 다를 수 있어 resampling이 필요하다.

3. 장르 라벨은 정수(integer)

모델은 정수(class id)를 출력하므로, 장르 이름을 매핑하는 int2str() 함수가 필요하다.

## 오디오 탐색

초기 데이터 감각을 잡기 위해 Gradio로 샘플을 재생해볼 수 있다.

```python
import gradio as gr

def generate_audio():
    example = gtzan["train"].shuffle()[0]
    audio = example["audio"]
    return (audio["sampling_rate"], audio["array"]), id2label_fn(example["genre"])
```

음악 장르 간의 차이가 확실히 느껴진다.
우리가 할 일은 “사람의 귀가 구분할 수 있는 차이를 Transformer도 구분하게 만드는 것”이다.

## 사전학습 모델 선택

음성 기반 Transformer 모델은 보통 대규모 음성 데이터(예시: LibriSpeech, VoxPopuli)로 pretraining된다.

사용 가능한 모델 중 이번 실습에서는 다음을 사용한다.

### DistilHuBERT

- HuBERT를 distillation하여 가볍게 만든 모델
- 약 73% 더 빠름
- 성능은 HuBERT 대비 대부분 유지
- Colab Free Tier에서도 문제 없이 학습 가능

이는 Encoder-only Transformer이므로 음악 분류 같은 단일 라벨 문제에 매우 적합하다.

## 오디오 -> 모델 입력 변환

Transformer는 raw audio를 그대로 입력받지 않고, feature extractor를 통해 적절한 형태로 정규화(normalization)된 입력을 받는다.

### Feature Extractor 준비

```python
from transformers import AutoFeatureExtractor
model_id = "ntu-spml/distilhubert"

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, 
    do_normalize=True, 
    return_attention_mask=True
)
```

### Why feature extractor?

오디오 값의 **동적 범위(dynamic range)**는 제각각이다.
따라서 정규화가 필요하다.

모델이 기대하는 sampling rate를 확인하고
입력 오디오를 강제로 맞춰야 한다.

다른 샘플 길이가 섞여 있을 때 attention mask를 생성해야 한다.

## Resampling (샘플링 레이트 통일)

- GTZAN: 22050 Hz
- DistilHuBERT: 16000 Hz

반드시 downsampling하여 맞춰야 한다.

```python
sampling_rate = feature_extractor.sampling_rate  # 16000
gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
```

Hugging Face Datasets는 이 변환을 즉각적으로 진행하므로 매우 편리하다.

## Feature Extractor의 실제 동작 확인

Normalization 전후 비교

```python
sample = gtzan["train"][0]["audio"]

print(np.mean(sample['array']), np.var(sample['array']))
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
print(np.mean(inputs['input_values']), np.var(inputs['input_values']))
```

출력에서 다음이 바뀐다.

- normalization 전: variance는 0.0493
- normalization 후: variance는 1.0
- mean은 거의 0에 가까워짐

이러한 feature scaling 덕분에 모델의 학습 안정성이 높아진다.

## 전체 데이터 전처리 함수 정의

GTZAN의 각 오디오는 30초 길이이므로, feature extractor의 max_length를 이용해 30초를 최대 길이로 지정한다.

```python
max_duration = 30.0

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs
```

맵핑 적용

```python
gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=100,
    num_proc=1,
)
```

### 메모리 최적화 팁

RAM 부족 시:

- batch_size를 절반씩 줄인다 (100 -> 50 -> 25)
- writer_batch_size도 절반씩 줄인다

실전에서도 매우 자주 사용하는 전략이다.

## 라벨 컬럼 이름 변경

Trainer는 label 이름을 반드시 "label"로 인식한다.

```python
gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
```

또한 id2label과 label2id 매핑도 생성한다.

```python
id2label = { str(i): id2label_fn(i) for i in range(num_labels) }
label2id = { v: k for k, v in id2label.items() }
```

## 모델 로드 (Classification Head 자동 추가)

```python
from transformers import AutoModelForAudioClassification

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
```

## Trainer를 활용한 파인튜닝

### TrainingArguments 설정

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    "distilhubert-finetuned-gtzan",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
)
```

- fp16: GPU 메모리 절감
- push_to_hub: 자동 업로드 가능
- warmup: 오디오 모델에서 convergence 안정화에 매우 중요

## 정확도 Metric 정의

```python
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

## 학습 시작

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Out-of-memory가 발생한다면?

- batch size를 8 -> 4 -> 2 등으로 줄이고
1 gradient_accumulation_steps로 보정한다

이 방식은 LLM fine-tuning에서도 똑같이 활용하는 기술이다.

## 학습 결과 예시

약 10 epoch 후 최고 accuracy 약 0.83이 나온다.
이는 매우 적은 데이터(899곡)로는 준수한 성능이다.

epoch 수 증가, dropout, 데이터 증강(Audio Augmentation) 등을 사용하면 더 상승될 수 있다.

음악 데이터를 30초 그대로 쓰는 것보다
15초 단위로 잘라 데이터 양을 늘리는 방식도 실전에서 종종 사용된다.

## 모델을 Hub에 업로드

```python
trainer.push_to_hub(**kwargs)
```

이후 사용자는 pipeline으로 바로 불러올 수 있다.

```python
from transformers import pipeline

pipe = pipeline(
    "audio-classification",
    model="your-username/distilhubert-finetuned-gtzan"
)
```

## 정리

이 글에서 다룬 전체 프로세스:

1. GTZAN 데이터셋 분석 및 Train/Validation 분할
2. DistilHuBERT 선택 이유 (경량, Encoder-only)
3. Sampling rate 통일 (22050 -> 16000)
4. Feature Extractor로 normalization
5. Dataset 전체 전처리
6. Hugging Face Trainer로 학습
7. 모델을 Hub에 업로드하고 pipeline으로 활용

이 과정은 음악 분류뿐 아니라 모든 오디오 분류 작업에 그대로 적용할 수 있다.

- Keyword Spotting
- Language Identification
- 환경음 분류
- 감정 분류
- 소리 이벤트 탐지(Audio Event Classification)

데이터셋만 바꾸면 된다. 참고로 학습 파이프라인에는 Trainer를 사용하지 않고 PyTorch의 전통적인 파이프라인을 사용해도 무방하다(데이터셋도 마찬가지).

참고자료
Huggingface, Audio Course, https://huggingface.co/learn