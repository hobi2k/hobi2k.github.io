---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 실습 과제 2"
date:   2025-11-28 00:10:22 +0900
categories: Huggingface_Audio
---

# 실습과제 2

이 글에서는 실습과제 2를 풀이한 과정을 해설한다.  
특히 파형, 스펙트럼, 멜스펙트럼 시각화와
코드 전체 및 파라미터에 관한 해설을 포함했다.

## 오디오 분야

오디오 AI는 텍스트/이미지보다 덜 직관적이다. 오디오 모델을 다룰 때 가장 중요한 요소는 다음 네 가지이다.

### 오디오 데이터는 “신호(signal)”이다

음성/음악은 시간에 따른 진폭 변화로 표현된다.

- 1초 동안 22,050개의 숫자(샘플링 레이트 22.05kHz라면)
- 각 숫자는 공기의 압력(음압)의 상대적 크기
- 즉, 1차원 배열(벡터) 형태

코드에서 처음 등장하는 오디오 데이터는 다음과 같다.

```python
first_array = first_item['audio']['array']
```

여기서 array는 **파형(waveform)**이다.

### 왜 주파수(frequency) 분석이 필요한가?

인간이 듣는 소리는 "주파수" 성분의 조합이다.

파형을 보면 "시간에 따른 변화"는 알 수 있지만,
소리가 어떤 악기·보컬·음색으로 이루어졌는지는 잘 드러나지 않는다.

- 저음(20~250Hz)
- 중음(250~2000Hz)
- 고음(2k~8kHz+)
- 타악·보컬의 패턴
- 장르 특성(예: 메탈의 강한 고주파, 힙합의 베이스 강조)

이 때 사용하는 것이 다음이다.

- 푸리에 변환(FFT, DFT)
- 스펙트로그램(STFT)
- 멜 스펙트로그램(Mel-spectrogram)

파형을 그대로 모델에 넣는 대신 “주파수 기반 특징(스펙트럼)”을 뽑으면 다음의 이점이 있다.

- 악기/장르/사람 목소리의 패턴이 더 잘 드러남
- 모델이 분류하기 쉬워짐

### 스펙토그램은 “시간 × 주파수 × 에너지”

한 곡의 음악을 아주 많은 작은 조각으로 잘라서
각 조각이 어떤 주파수를 얼마나 포함하고 있는지를 표시한 이미지라 할 수 있다.

즉, 오디오를 이미지처럼 바꾼 형태이다.

### 멜(Mel) 스펙토그램은 사람 귀에 맞춘 버전

사람의 청각은 고주파보다 저주파 차이를 더 예민하게 느낀다.
멜스펙트럼은 그 심리음향적 기준을 반영한 스펙트럼이며,
최근 TTS, 음성 인식, 장르 분류에서 기본 전처리로 사용된다.

## 시각화 코드

오디오 시각화에는 세 가지 종류가 있다.

### 파형(waveform)

```python
plt.figure().set_figwidth(16)
librosa.display.waveshow(first_array, sr=item_sr)
```

- x축: 시간(sec)
- y축: 음압(amplitude)
- 값이 클수록 더 큰 소리(음량)
- 갑자기 폭이 커지면 강한 타악이나 보컬 강세

그래프에서 알 수 있는 것

- 무음 구간 존재 여부
- 큰 드럼/베이스의 패턴
- 스트리밍 서비스에서 사용하는 "노멀라이징" 여부

장르 분류에서는 파형만 보고 장르를 알기 어렵다.
하지만 전반적인 역동성과 리듬의 형태는 확인할 수 있다.

### 주파수 스펙트럼(DFT, FFT)

```python
window = np.hanning(len(first_array))
windowed_input = first_array * window
dft = np.fft.rfft(windowed_input)

amp = np.abs(dft)
amp_db = librosa.amplitude_to_db(amp, ref=np.max)
frequency = librosa.fft_frequencies(sr=item_sr, n_fft=len(first_array))

plt.figure().set_figwidth(16)
plt.plot(frequency, amp_db)
plt.xlabel("Frequency")
plt.ylabel("Amp (db)")
plt.xscale("log")
```

- x축: 주파수(Hz)
- y축: 해당 주파수의 에너지(dB)

그래프에서 알 수 있는 것

- 강한 저음(EDM, 힙합)
- 활발한 고음(클래식/어쿠스틱)
- 스네어/킥 패턴의 에너지 분포

로그 스케일(log-scale)을 쓰는 이유는
인간 청각은 주파수 차이를 비율로 인식하기 때문이다.

### 스펙토그램(STFT)

```python
stft = librosa.stft(first_array)
s_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

plt.figure().set_figwidth(16)
librosa.display.specshow(s_db, x_axis="time", y_axis="hz")
plt.colorbar()
```

- x축: 시간
- y축: 주파수
- 색깔: 에너지(진하기)

그래프에서 알 수 있는 것

- 진한 부분이 많으면 해당 주파수가 강함
- 킥드럼: 60–120Hz 근처에 강한 짧은 세로줄
- 하이햇: 고주파 영역(8kHz)에서 짧은 스파이크
- 보컬 폼란트(화음 구조)도 보임

### 멜(Mel) 스펙토그램

```python
mel_s = librosa.feature.melspectrogram(y=first_array, sr=item_sr, n_mels=80, fmax=8000)
mel_s_db = librosa.power_to_db(mel_s, ref=np.max)

plt.figure().set_figwidth(16)
librosa.display.specshow(mel_s_db, x_axis="time", y_axis="mel", sr=item_sr, fmax=8000)
plt.colorbar()
```

장르 분류에서 가장 중요한 이유

- 모델이 2차원 CNN처럼 이미지로 학습 가능
- 인간 귀에 가까운 주파수 표현
- 잡음에 강함

그래프에서 알 수 있는 것

- 밝은 색: 강한 주파수
- 멜축(Mel scale)이라 실제 Hz가 아니라 “사람 귀 느낌”에 가까움

## 전체 코드 및 파라미터

### 라이브러리 설치

```python
# librosa: 오디오 분석 기본 라이브러리
!pip install librosa

# Hugging Face datasets: 데이터셋 로딩을 위한 라이브러리
!pip install datasets==2.18.0

# transformers: 오디오 모델 포함한 HF 모델 라이브러리
!pip install transformers==4.57.3

# evaluation library
!pip install evaluate

# 충돌 방지를 위해 기존 torch 제거
!pip uninstall -y torch torchvision torchaudio

# RTX 5080 + CUDA 12.8 환경에 맞춘 PyTorch 설치(Nightly)
!pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

- librosa: 오디오 전처리 기본 라이브러리
- datasets: Hugging Face dataset 로드
- transformers: 오디오 모델 제공
- nightly torch(cu128): RTX 5080 환경에서 CUDA 12.8 필요

### 라이브러리 임포트

```python
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import evaluate
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

# 오디오 처리 라이브러리
import librosa
import librosa.display

# HuggingFace dataset, 오디오 로딩, 모델
from datasets import Audio, load_dataset
from transformers import (
    pipeline, Trainer, AutoFeatureExtractor,
    AutoModelForAudioClassification, TrainingArguments,
    EarlyStoppingCallback, WavLMForSequenceClassification
)

# HuggingFace Hub 업로드용
from huggingface_hub import notebook_login

# 오디오 재생
from IPython.display import Audio as Au
```

### Config 클래스

```python
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 여부
    SPLIT_SIZE = 0.1  # test set 비율
    SEED = 42  # 재현성 확보 위한 시드값
```

### GTZAN 데이터셋 로드

```python
# GTZAN 음악 장르 데이터셋 불러오기
gtzan = load_dataset("marsyas/gtzan", "default")

# train split만 사용 (원본 구조는 train만 존재)
gtzan = gtzan["train"]
```

### 파형 시각화

```python
first_item = gtzan[0]
first_array = first_item['audio']['array']
item_sr = first_item['audio']['sampling_rate']

plt.figure().set_figwidth(16)
librosa.display.waveshow(first_array, sr=item_sr)
```

### 주파스 스펙트럼(FFT) 시각화

```python
# Hanning window를 적용하여 경계 부근 신호 왜곡을 완화
window = np.hanning(len(first_array))

# 원 신호에 윈도우 적용
windowed_input = first_array * window

# rfft: 실수 신호에 대해 양쪽이 아닌 양수 주파수만 계산하는 FFT
dft = np.fft.rfft(windowed_input)

# 복소수 FFT 결과 -> magnitude(진폭)
amp = np.abs(dft)

# dB scale 변환 (log scale)
amp_db = librosa.amplitude_to_db(amp, ref=np.max)

# 해당 FFT에서 사용된 주파수 bin 계산
frequency = librosa.fft_frequencies(sr=item_sr, n_fft=len(first_array))

# 그래프 그리기
plt.figure().set_figwidth(16)
plt.plot(frequency, amp_db)
plt.xlabel("Frequency")
plt.ylabel("Amplitude (dB)")

# 주파수 특성은 로그 스케일이 더 유용
plt.xscale("log")
```

### 스펙토그램 시각화

```python
# STFT (Short-Time Fourier Transform) 계산
stft = librosa.stft(first_array)

# magnitude -> dB 변환
s_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

plt.figure().set_figwidth(16)

# 시간-주파수 2D 이미지로 시각화
librosa.display.specshow(
    s_db, x_axis="time", y_axis="hz"
)
plt.colorbar()  # 색 범례 추가
```

### 멜 스펙트로그램 시각화

```python
mel_s = librosa.feature.melspectrogram(
    y=first_array, # 입력 오디오
    sr=item_sr, # 샘플링 레이트
    n_mels=80, # 멜 필터 은닉 수 (일반적으로 80~128)
    fmax=8000 # 최대 주파수
)

# 멜 스펙트로그램 magnitude를 dB scale로 변환
mel_s_db = librosa.power_to_db(mel_s, ref=np.max)

plt.figure().set_figwidth(16)
librosa.display.specshow(
    mel_s_db, x_axis="time", y_axis="mel",
    sr=item_sr, fmax=8000
)
plt.colorbar()
```

### Feature Extractor 설정

```python
model_id = "ntu-spml/distilhubert"

# DistilHuBERT의 입력 규격에 맞춘 전처리기
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True, # 입력 오디오 정규화 여부
    return_attention_mask=True # attention mask 반환
)
```

### 리샘플링

```python
sampling_rate = feature_extractor.sampling_rate

dataset = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
```

핵심 개념

- 모델에 입력되는 모든 오디오의 샘플링레이트는 동일해야 한다.
- DistilHuBERT: 16kHz 고정

따라서 22kHz 음악 파일도 16kHz로 변환한다.

### Data Preprocessing

```python
def data_preprocessing(examples):
    # batch 내 오디오 배열 가져오기
    audio_arrays = [x["array"] for x in examples["audio"]]

    # feature extractor를 이용하여 패딩, 정규화, 길이 통일 수행
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate,
        max_length=int(sampling_rate * 30.0), # 30초 제한
        padding="max_length", # 짧으면 뒤에 0 채움
        truncation=True, # 길면 자름
        return_attention_mask=True
    )
    return inputs
```


| 파라미터                   | 설명                    |
| ---------------------- | --------------------- |
| `sampling_rate`        | 모델 입력 샘플링레이트          |
| `max_length`           | 최대 입력 길이(여기선 30초로 고정) |
| `padding="max_length"` | 길이가 짧으면 뒤에 0을 채움      |
| `truncation=True`      | 30초 넘으면 자름            |


### Train / Test Split

```python
dataset_processed = dataset_processed.train_test_split(
    seed=Config.SEED,
    test_size=Config.SPLIT_SIZE,
    stratify_by_column="label"  # 장르 비율 유지
)
```

- stratify_by_column: 장르 비율이 train/test에서 동일하도록 유지

음원 장르 데이터는 불균형이 있으므로 반드시 필요하다.

### 라벨 매핑

```python
int2str_fn = dataset_processed["train"].features["label"].int2str
id2label = {str(i): int2str_fn(i) for i in range(len(dataset_processed["train"].features["label"].names))}
label2id = {v: k for k, v in id2label.items()}
```

### 모델 초기화

```python
model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=len(id2label),
    label2id=label2id,
    id2label=id2label,
)
```

DistilHuBERT란?

- Wav2Vec2 기반의 경량화 음성 모델
- 오디오 분류/음성 인식 기초 모델로 많이 사용됨

### TrainingArguments

```python
training_args = TrainingArguments(
    f"{model_name}-finetuned-gtzan",
    eval_strategy="epoch", # epoch마다 평가
    save_strategy="epoch", # epoch마다 저장
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    warmup_ratio=0.1, # 학습 안정화
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True, # half precision
    push_to_hub=True
)
```


| 옵션                            | 의미                           |
| ----------------------------- | ---------------------------- |
| `fp16=True`                   | 메모리 절약 + 빠른 훈련               |
| `warmup_ratio=0.1`            | LR warm-up 구간                |
| `load_best_model_at_end=True` | validation accuracy 최고 모델 로딩 |
| `eval_strategy="epoch"`       | 매 epoch 평가                   |
| `save_strategy="epoch"`       | 매 epoch 저장                   |
| `learning_rate=5e-5`          | 미세조정 기본값 (작을수록 안전함)          |


### Custom Trainer (CrossEntropy 강제 적용)

```python
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # DistilHuBERT 입력 포맷 맞추기 위해 long 타입 강제
        labels = labels.to(model.device, dtype=torch.long)

        # 입력 텐서 디바이스 이동
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)

        outputs = model(**inputs)
        logits = outputs.logits

        # 일반적인 분류 문제 손실
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.float(), labels)

        return (loss, outputs) if return_outputs else loss

```

- DistilHuBERT 구조 특성상 입력 텐서 타입 문제를 회피
- 라벨 타입을 torch.long으로 강제하여 오류 방지

### 모델 훈련

```python
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
```

파형 -> 특성 추출 -> Hidden states -> 분류 레이어

장르(10개) 중 1개 선택

### 허깅페이스 모델 업로드

```python
trainer.push_to_hub(
    dataset_tags="marsyas/gtzan",
    dataset="GTZAN",
    model_name=f"{model_name}-finetuned-gtzan",
    finetuned_from=model_id,
    tasks="audio-classification"
)
```

참고자료
Huggingface, Audio Course, https://huggingface.co/learn