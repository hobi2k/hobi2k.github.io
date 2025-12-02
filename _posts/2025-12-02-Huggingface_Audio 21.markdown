---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 실습 과제 3"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_Audio
---

# 실습 과제 3

Whisper Tiny로 Minds14(영어) 데이터셋을 ASR 파인튜닝한다.

### 개요

- 데이터셋: PolyAI/minds14
 (여기서는 en-US 언어 서브셋 사용)
- 모델: openai/whisper-tiny

Minds14 데이터셋의 영어 음성 -> 영어 텍스트 변환(ASR) 작업을 위해 Whisper Tiny를 파인튜닝한다.

호환성 문제로 전통적인 PyTorch 학습 루프로 훈련한다.
베스트 모델을 저장하고 Hugging Face Hub에 업로드한다.
pipeline("automatic-speech-recognition")으로 결과를 테스트한다.

## 환경 설정

Colab/노트북 환경을 기준으로 필요한 라이브러리를 설치한다.

```python
# librosa: 오디오 신호 처리(스펙트로그램, 로딩 등)
!pip install librosa

# datasets: Hugging Face Datasets 라이브러리 (데이터셋 로딩/전처리)
!pip install datasets

# 최신 transformers: Whisper 포함 최신 기능 사용을 위해 GitHub 버전 설치
!pip install git+https://github.com/huggingface/transformers

# Hugging Face Hub 관련 유틸
!pip install huggingface_hub

# Gradio: 간단한 웹 데모용 (이 미션 코드에서는 크게 사용 X)
!pip install gradio

# jiwer: WER(Word Error Rate) 지표 계산용
!pip install jiwer

# transformers + accelerate 최신 버전 업데이트
!pip install -U transformers accelerate
```

## 라이브러리 임포트 및 기본 설정

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import evaluate

# PyTorch 관련
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

# 오디오 및 데이터셋
import librosa
import librosa.display
from datasets import Audio, load_dataset, DatasetDict
from functools import partial

# Transformers: Whisper 관련 클래스
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Hugging Face Hub 로그인
from huggingface_hub import notebook_login

# 노트북에서 오디오 재생용
from IPython.display import Audio as Au

# Gradio (이번 예제에서는 필수는 아님)
import gradio as gr
```

### Hugging Face Hub 로그인

```python
# 허깅페이스 허브 로그인 (토큰 입력 필요)
notebook_login()
```

### Config 클래스 정의

훈련/평가에서 공통으로 사용할 상수를 모아둔 설정 클래스다.

```python
class Config:
    # GPU가 있으면 cuda, 없으면 cpu 사용
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 시드값 (재현성 확보용)
    SEED = 42
```

## 데이터셋 로딩 및 구조 확인

이번 미션에서는 PolyAI/minds14 데이터셋의 en-US 서브셋을 사용한다.

```python
# Minds14 데이터셋 로딩 (en-US 서브셋 사용)
minds = load_dataset("PolyAI/minds14", "en-US")

# 데이터셋 구조 확인 (train/validation/test 정보 등)
minds
```

Minds14의 각 샘플에는 대략 다음과 같은 필드가 들어 있다.

- path: 원본 파일 경로
- audio: 오디오 데이터(dict: "array"와 "sampling_rate")
- transcription: 원 언어 텍스트
- english_transcription: 영어로 번역된 텍스트
- intent_class, lang_id 등 메타 정보

## Whisper Processor 초기화 및 오디오 리샘플링

Whisper는 고정된 샘플링 레이트(일반적으로 16kHz)를 기대하기 때문에, 데이터셋의 오디오 컬럼을 Whisper의 샘플링 레이트로 맞춰야 한다.

```python
# WhisperProcessor: 토크나이저 + feature_extractor를 함께 제공
wprocessor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny",
    task="transcribe"  # 영어 -> 영어 음성 인식 (번역 모드 X)
)

# Whisper가 기대하는 샘플링 레이트
sampling_rate = wprocessor.feature_extractor.sampling_rate

# train split의 "audio" 열을 Whisper 샘플링 레이트로 캐스팅(리샘플링)
minds["train"] = minds["train"].cast_column("audio", Audio(sampling_rate=sampling_rate))
```

## 전처리 함수 정의

오디오 파형을 Whisper 입력(feature)과 레이블(텍스트 토큰)로 변환하는 부분이다.

```python
def preprocessing(instance):
    """
    하나의 샘플(instance)에 대해:
    - 오디오 배열과 샘플링 레이트를 꺼내고
    - WhisperProcessor를 통해 입력 특징(input_features)과 레이블(labels)을 생성
    - 추가로, 음성 길이(초 단위)를 계산하여 필터링에 사용

    반환되는 딕셔너리에는 대략 다음 키들이 포함된다:
    - input_features: 모델 입력용 로그 멜 스펙트로그램 특징
    - labels: 토크나이저를 통해 변환된 정수 시퀀스
    - input_length: 음성 길이(초)
    """
    audio = instance["audio"]  # {"array": np.ndarray, "sampling_rate": int}

    # WhisperProcessor를 호출하면 feature_extractor + tokenizer가 함께 동작한다.
    instance = wprocessor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=instance["english_transcription"],  # 영어 ASR 타깃
    )

    # 길이(초 단위) 저장 -> 뒤에서 max 길이 필터링에 사용
    instance["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return instance
```

이 전처리를 전체 데이터셋에 적용하면서, 원본 열들을 제거한다.

```python
processed_dataset = minds.map(
    preprocessing,
    remove_columns=minds.column_names["train"],  # train split의 모든 컬럼 제거 후, 새 키들만 남김
    num_proc=1  # 병렬 처리 프로세스 수 (Colab 환경 안정성을 위해 1로 설정)
)

processed_dataset
```

## 길이 필터링 및 Train/Validation 분할

길이가 너무 긴 음성은 훈련을 불안정하게 만들 수 있기 때문에, 30초를 기준으로 필터링한다.

```python
# 허용 최대 길이(초)
max_input_length = 30.0

def audio_filtering(length):
    """
    input_length가 max_input_length보다 작은 샘플만 남긴다.
    filter 함수에서 각 샘플의 length 값에 대해 True/False를 반환.
    """
    return length < max_input_length

# train split에 필터 적용
processed_dataset["train"] = processed_dataset["train"].filter(
    audio_filtering,
    input_columns=["input_length"]  # 필터 함수에 넘겨줄 열
)

processed_dataset
```

이제 필터링된 train split을 train/validation으로 나눈다.

```python
# 0 ~ 449번 샘플 -> train
train_raw = processed_dataset["train"].select(range(450))

# 450번 이후 ~ 끝까지 -> validation
val_raw = processed_dataset["train"].select(range(450, len(processed_dataset["train"])))

# DatasetDict 형태로 재구성
processed_dataset = DatasetDict({
    "train": train_raw,
    "val": val_raw
})

# 상태 확인
processed_dataset
processed_dataset["train"]
```

## Custom Data Collator 정의

Whisper 입력은 각 샘플마다 길이가 다르기 때문에, 배치로 묶을 때 패딩이 필요하다.
Hugging Face Trainer를 쓰면 data_collator를 자동으로 구성해주지만, 여기서는 직접 DataLoader + 커스텀 Collator를 구현한다.

```python
@dataclass
class CustomCollator:
    """
    가변 길이 오디오 특징과 텍스트 레이블을 패딩하여
    모델에 바로 넣을 수 있는 배치 텐서로 만들어주는 Collator.

    - processor.feature_extractor.pad: input_features 패딩
    - processor.tokenizer.pad: labels 패딩
    """
    processor: any  # WhisperProcessor

    def __call__(self, features: list) -> dict[str, torch.Tensor]:
        """
        features: Dataset에서 뽑힌 샘플 딕셔너리들의 리스트
        반환: PyTorch 텐서들로 구성된 배치 딕셔너리
        """

        # 전처리에서 "input_features"의 shape는 (1, T, D)에 해당하므로
        # [0] 인덱스로 첫 번째(유일한) 채널을 꺼내어 리스트를 만든다.
        input_features = [
            {"input_features": feature["input_features"][0]}
            for feature in features
        ]

        # feature_extractor.pad를 이용해 길이가 다른 시퀀스를 패딩 후 텐서로 변환
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # 레이블(텍스트)도 토크나이저의 pad 메서드로 패딩
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # 어텐션 마스크가 1이 아닌 위치(패딩 위치)를 -100으로 채워서
        # loss 계산 시 무시하도록 설정
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        # Whisper 토크나이저는 문장 앞에 BOS 토큰을 붙인다.
        # 모델 내부에서 또 BOS를 붙이므로, 여기서는 레이블에서 제거해 준다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # batch 딕셔너리에 labels 추가
        batch["labels"] = labels

        return batch
```

Data Collator 인스턴스를 만든다.

```python
data_collator = CustomCollator(processor=wprocessor)
```

## 평가 지표(WER) + 정규화


### WER 지표 로딩

```python
# Word Error Rate 지표 로딩
metric = evaluate.load("wer")

# 영어 텍스트 정규화 도구 (대소문자, 구두점 등 정리)
normalizer = BasicTextNormalizer()
```

### compute_metrics 함수

훈련 루프 안에서는 직접 쓰지 않지만, 전형적인 WER 계산 패턴이라 참고용으로 좋다.

```python
def compute_metrics(pred, processor=wprocessor):
    """
    Seq2SeqTrainer 스타일의 compute_metrics 함수 예시.
    (현재 코드는 직접 PyTorch 루프를 쓰기 때문에 호출되지는 않음)

    - pred.predictions: 모델에서 나온 로짓/토큰 ID
    - pred.label_ids: 레이블 토큰 ID
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100을 pad_token_id로 되돌려 디코딩 가능하게 만든다.
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 문자열 디코딩
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # 정규화 전 WER
    wer_ortho = metric.compute(predictions=pred_str, references=label_str)

    # 정규화 후 문자열
    pred_str_norm = [normalizer(p) for p in pred_str]
    label_str_norm = [normalizer(l) for l in label_str]

    # 정규화 후 길이가 0인 문장은 제거
    pred_str_norm = [
        pred_str_norm[i]
        for i in range(len(pred_str_norm))
        if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    # 정규화된 문자열 기준 WER
    wer = metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}
```

## DataLoader 구성

CustomCollator를 collate_fn으로 사용하여 PyTorch DataLoader를 만든다.

```python
train_loader = DataLoader(
    processed_dataset["train"],
    batch_size=15,          # 배치 크기
    shuffle=True,           # 에포크마다 셔플
    collate_fn=data_collator,
    drop_last=True          # 배치가 딱 나누어 떨어지지 않을 경우 마지막 배치 버림
)

val_loader = DataLoader(
    processed_dataset["val"],
    batch_size=15,
    shuffle=False,
    collate_fn=data_collator,
    drop_last=True
)
```

## 모델, 옵티마이저, 스케줄러 설정

```python
# Whisper Tiny 모델 로드 후 GPU/CPU로 이동
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny"
).to(Config.DEVICE)

# 캐시 사용 끔 (훈련 시 메모리 및 경고 회피)
model.config.use_cache = False

# 강제 디코더 토큰 설정 해제
# (기본적으로 Whisper는 language, task 등에 따라 forced_decoder_ids를 쓸 수 있음)
model.config.forced_decoder_ids = None

# AdamW 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=1e-5)
```

### 간단한 Warmup 스케줄러

초기 학습 단계에서 learning rate를 서서히 늘린 다음, 그 이후에는 일정하게 유지하는 매우 단순한 스케줄러다.

```python
def lr_lambda(step):
    """
    step < warmup_steps 동안은 선형으로 LR 상승,
    이후에는 1.0(기본 lr 유지).
    """
    warmup_steps = 50
    if step < warmup_steps:
        # 기본 lr × (계수) 형태로 실제 lr을 조절
        return float(step) / float(max(1, warmup_steps))
    return 1.0

# 람다 함수 기반 LR 스케줄러
scheduler = LambdaLR(optimizer, lr_lambda)
```

## 훈련 루프 정의

```python
def train_one_epoch(model, loader, optimizer, scheduler):
    """
    한 에포크 동안 train_loader 전체를 순회하면서:
    - 모델 forward
    - loss 계산
    - 역전파 및 optimizer.step()
    - 스케줄러로 learning rate 업데이트

    반환값:
    - 에포크 평균 훈련 손실
    """
    model.train()
    total_loss = 0.0

    # tqdm으로 진행 상황 출력
    pbar = tqdm(loader, desc="Training")

    for batch in pbar:
        # 배치의 모든 텐서를 GPU/CPU로 보냄
        batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}

        # WhisperForConditionalGeneration은
        # input_features와 labels를 입력하면 내부에서 loss를 계산해 준다.
        outputs = model(
            input_features=batch["input_features"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        loss.backward()  # 역전파

        optimizer.step()     # 파라미터 업데이트
        scheduler.step()     # LR 스케줄러 한 스텝 진행
        optimizer.zero_grad()  # 그래디언트 초기화

        total_loss += loss.item()

        # tqdm 진행 바에 현재까지의 평균 손실을 표시
        pbar.set_postfix({"loss": total_loss / (len(pbar) + 1)})

    # DataLoader 길이 기준 평균 손실 반환
    return total_loss / len(loader)
```

## 검증 루프 정의 (WER 계산)

```python
def validate(model, loader, processor):
    """
    검증용 루프.
    - 모델을 eval 모드로 두고
    - generate()로 토큰 ID를 생성한 뒤
    - 디코딩해서 WER를 계산한다.

    반환값:
    - 전체 검증 세트에 대한 평균 WER
    """
    model.eval()
    total_wer = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}

            # generate: 인퍼런스 모드에서 토큰 시퀀스 생성
            pred_ids = model.generate(
                input_features=batch["input_features"],
                max_length=225  # 디코딩 최대 길이
            )

            # 예측 문자열 디코딩
            pred_str = processor.batch_decode(
                pred_ids,
                skip_special_tokens=True
            )

            # 레이블에서 -100을 pad_token_id로 되돌림
            labels = batch["labels"].clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id

            # 레이블 문자열 디코딩
            label_str = processor.batch_decode(
                labels,
                skip_special_tokens=True
            )

            # WER 계산 (정규화 없이 바로 비교)
            wer_score = metric.compute(
                predictions=pred_str,
                references=label_str
            )
            total_wer.append(wer_score)

    # 평균 WER 반환
    return np.mean(total_wer)
```

## 전체 학습 루프 및 베스트 모델 저장

```python
best_wer = 999  # 매우 큰 값으로 초기화
EPOCHS = 30

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch + 1} / {EPOCHS} =====")

    # 훈련
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
    print(f"Train Loss: {train_loss:.4f}")

    # 검증
    val_wer = validate(model, val_loader, wprocessor)
    print(f"Validation WER: {val_wer:.4f}")

    # 베스트 모델 저장 (WER가 작을수록 성능이 좋음)
    if val_wer < best_wer:
        best_wer = val_wer
        torch.save(model.state_dict(), "my_whisper_tiny.pth")
        print(">>> Best model saved.")

print("Training finished.")
print(f"BEST WER = {best_wer:.4f}")
```


## 최고 모델 로드 및 로컬/Hub 저장

### 로컬에서 모델/프로세서 저장

```python
# 저장해 둔 베스트 모델 가중치 로드
model.load_state_dict(torch.load("my_whisper_tiny.pth"))

# 로컬 디렉토리에 모델과 프로세서 저장
save_dir = "whisper-tiny-finetuned"
model.save_pretrained(save_dir)
wprocessor.save_pretrained(save_dir)
```

### Hugging Face Hub에 업로드

```python
# 업로드할 리포지토리 이름 (username/repo 형식)
repo_name = "ahnhs2k/whisper-tiny-minds14"

# 모델 업로드
model.push_to_hub(
    repo_name,
    commit_message="Upload PyTorch finetuned Whisper model"
)

# 프로세서(토크나이저 + feature_extractor) 업로드
wprocessor.push_to_hub(repo_name)
```

## 파이프라인으로 간단 테스트

Hub에 올라간 모델을 pipeline으로 바로 불러와서, 데이터셋의 샘플 하나에 대해 결과를 확인해 볼 수 있다.

```python
# train split에서 첫 번째 샘플 가져오기
first_sample = minds["train"][0]
first_sample
```

### 오디오 재생

```python
# 노트북에서 오디오를 재생해 볼 수 있음
Au(first_sample["audio"]["array"], rate=first_sample["audio"]["sampling_rate"])
```

### 파이프라인 생성 및 예측

```python
# 허브에 업로드한 파인튜닝된 Whisper 모델로 파이프라인 생성
pipe = pipeline(
    "automatic-speech-recognition",
    model="ahnhs2k/whisper-tiny-minds14"
)

# 첫 번째 샘플에 대해 추론
pipe(first_sample["audio"].copy())
```

## 정리

- Hugging Face Datasets로 오디오 데이터셋 로딩 및 전처리
- WhisperProcessor를 이용한 입력(feature), 레이블 생성
- Custom Data Collator로 가변 길이 시퀀스 패딩 처리
- Hugging Face Trainer가 아닌 전통적인 PyTorch 훈련 루프 직접 구현
- WER 지표로 음성 인식 성능 평가
- Hugging Face Hub에 모델/프로세서 업로드 후, pipeline("automatic-speech-recognition")으로 손쉽게 재사용

참고자료
Huggingface, Audio Course, https://huggingface.co/learn