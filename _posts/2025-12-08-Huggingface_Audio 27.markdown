---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - SpeechT5 한국어 TTS 프로젝트 2"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_Audio
---

# SpeechT5 한국어 TTS 파이프라인 코드 해설

(Jamo Tokenizer + KSS Dataset + Speaker Embedding + Training + Hub Upload)

이 글에서는 스크립트의 구조, 흐름을 정리한다.

## 모듈 Import - 기능별 의존성 로딩

```python
import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import librosa
import soundfile as sf

from pathlib import Path
from dataclasses import dataclasses, dataclass
from tqdm.auto import tqdm

from datasets import load_dataset, Audio, DatasetDict

from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Wav2Vec2FeatureExtractor,
    WavLMForXVector,
    PreTrainedTokenizerFast,
)

from torch.nn.utils.rnn import pad_sequence

# Jamo WordLevel 토크나이저 생성
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from huggingface_hub import HfApi, Repository

import unicodedata
import argparse
```

이 스크립트는 크게 4개의 역할을 위해 import를 한다.

1. 데이터 & 오디오 처리 계층: librosa, soundfile, datasets
2. 모델 계층: SpeechT5, HifiGAN, WavLM
3. 토크나이징 계층: tokenizers 라이브러리 + HuggingFace 토크나이저 래퍼
4. 훈련 계층: PyTorch, DataLoader, AMP

여기서 가장 중요한 점은 SentencePiece를 사용하지 않고 직접 WordLevel 토크나이저를 만든다는 것이다.

## CLI Argument – optional training skip

```python
parser = argparse.ArgumentParser()
parser.add_argument("--skip_train", action="store_true")
args = parser.parse_args()
```

`--skip_train` 옵션이 있으면
모델 학습을 건너뛰고 바로 저장/배포 과정으로 진행할 수 있다.

즉,

- 이미 학습이 끝난 상태에서 재실행할 때
- checkpoint만 올릴 때

등에 사용한다.

## Config - 전역 설정 컨테이너

```python
@dataclass
class Config:
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_SR: int = 16000
    MAX_FRAMES: int = 2000
    BATCH_SIZE: int = 3
    NUM_EPOCHS: int = 35
    LR: float = 1e-4
    MAX_AUDIO_LEN: int = 10
    WEIGHT_DECAY: float = 1e-6
    MAX_TOKENS: int = 200
    TRAIN_RATIO: float = 0.98

    MODEL_SAVE: Path = Path("./speecht5_kss_korean_jamo")
    JAMO_VOCAB: Path = Path("./jamo_vocab.txt")

    HF_USER: str = "ahnhs2k"
    HF_REPO_NAME: str = "speecht5-korean"
    HF_PRIVATE: bool = False
    HF_REPO_ID: str = ""

    SEED: int = 42
```

이 dataclass는 프로젝트 전체에서 쓰이는 모든 설정을 저장한다.

특히 중요한 항목은:

- TARGET_SR = 16000

SpeechT5는 기본적으로 16kHz mel spectrogram을 사용한다.

- MAX_FRAMES = 2000

멜 spectrogram 타겟 길이를 제한하기 위한 파라미터.

- TRAIN_RATIO = 0.98

전체 데이터 중 98%를 사용하여 fine-tuning: TTS 모델은 일반적으로 Train이 매우 중요하고 Test가 크게 필요하지 않기 때문.

- HF_* 계열

HuggingFace에 자동 푸시할 설정들.

## 시드 고정 - 재현성 확보

```python
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(Config.SEED)
print("Device:", Config.DEVICE)
```

모델 학습은 재현성이 있어야 한다.
random, numpy, torch, CUDA 시드를 모두 동일하게 설정하여 반복 실행 가능하게 만든다.

## decompose_jamo - 자모 단위로 텍스트 분해

```python
def decompose_jamo(text: str):
    result = []
    for ch in text:
        if "HANGUL SYLLABLE" in unicodedata.name(ch, ""):
            code = ord(ch) - 0xAC00
            choseong = chr(0x1100 + (code // 588))
            jungseong = chr(0x1161 + ((code % 588) // 28))
            jong = code % 28

            result.append(choseong)
            result.append(jungseong)
            if jong > 0:
                result.append(chr(0x11A7 + jong))
        else:
            result.append(ch)
    return result
```

한국어 완성형 음절(가~힣)을 초성·중성·종성으로 분해하는 함수이다.

예시

```python
"안녕" -> ['ᄋ','ᅡ','ᆫ','ᄂ','ᅧ','ᆼ']
```

이 방식의 장점

- OOV 사라짐
- 발음 단위와 더 잘 맞음
- TTS 학습에서 미세 음운 단위 모델링 가능

## Jamo Tokenizer 생성

```python
vocab = {}
with open(Config.JAMO_VOCAB, "r", encoding="utf-8") as f:
    for idx, tok in enumerate(f.read().splitlines()):
        vocab[tok] = idx
```

jamo_vocab.txt를 읽어 {token: id} 맵을 구성한다.
이 파일은 다음 순서를 포함해야 한다.

```python
<pad>
<unk>
<bos>
<eos>
ᄀ
ᄁ
ᄂ
...
```

```python
tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
tok.pre_tokenizer = Whitespace()
```

여기서 중요한 점

- SentencePiece 없이 직접 WordLevel 토큰 매핑 생성
- pre_tokenizer=Whitespace

하지만 실제 토큰 입력은 “자모 리스트”이므로 크게 영향 없음.

```python
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(tokenizer_json_path),
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
)
```

tokenizers 라이브러리로 만든 tokenizer.json을
HuggingFace 토크나이저 래퍼로 감싼다.

Transformer 모델에 바로 호환시킬 수 있게 하는 필수 단계다.

## KSS Dataset 로드 + waveform 전처리

```python
kss = load_dataset("Bingsu/KSS_Dataset")
```

HuggingFace Datasets에서 바로 다운로드된다.
로컬 캐시가 없으면 자동 다운로드.

```python
def safe_audio(batch):
    arr = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]

    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

    arr = arr.astype(np.float32)

    if len(arr) > Config.MAX_AUDIO_LEN * sr:
        arr = arr[: Config.MAX_AUDIO_LEN * sr]

    batch["waveform"] = arr
    batch["sr"] = 16000
    return batch
```


핵심 전처리

1. 리샘플링 -> 16kHz
2. 길이 자르기 -> 10초 제한
3. waveform 저장

이후 학습이 안정되고 GPU 메모리 사용량이 일정하게 유지된다.

## 모델 로드 – SpeechT5, HifiGAN, Processor

```python
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```

- processor: mel extraction + tokenizer(여기선 교체 예정)
- model: Transformer 기반 Text-to-Mel
- vocoder: mel → waveform 변환기

```python
processor.tokenizer = tokenizer
model.resize_token_embeddings(len(tokenizer))
```

이 두 줄이 매우 중요하다.

- SpeechT5 기본 토크나이저는 영어 기반 -> 버린다
- 커스텀 자모 tokenizer로 교체
- 임베딩 layer 크기를 새 vocab 크기에 맞춰 재조정

이 과정 없이 학습하면 embedding 매트릭스 크기 mismatch 에러 발생한다.

## WavLM Speaker Embedding - 단일 화자 TTS

```python
spk_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
```

TTS를 단일 화자로 학습하려면
speaker embedding을 입력에 계속 넣어야 한다.

## preprocess - 텍스트·오디오를 모델 입력으로 변환

```python
def preprocess(batch):
    """
    하나의 샘플에 대해:
    - 'decomposed_script' 텍스트 사용 (필요시 컬럼명 수정)
    - decompose_jamo() 로 자모 시퀀스 생성
    - jamo 토크나이저로 input_ids, attention_mask 생성
    - processor(audio_target=...) 를 이용해 멜 스펙 타겟(labels) 생성
    """
    wf = batch["waveform"]

    # TODO: 실제 컬럼명 확인 필요. Bingsu/KSS_Dataset 기준으로:
    # 'script', 'normalized_script', 'decomposed_script' 등이 있을 수 있다.
    # 여기서는 사용자가 이전에 쓰던 decomposed_script로 맞춰 둔다.
    text_str = batch["decomposed_script"]

    # 1) 텍스트 -> 자모 시퀀스
    jamo_seq = decompose_jamo(text_str)

    # 2) 자모 토큰들을 그대로 토크나이저에 전달
    encoded = tokenizer(
        jamo_seq,
        is_split_into_words=True,      # jamo_seq 원소 각각을 1토큰으로 간주
        add_special_tokens=True,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 3) 멜 스펙 타겟 생성
    audio_out = processor(
        audio_target=wf,
        sampling_rate=Config.TARGET_SR,
        truncation=True,
        max_length=Config.MAX_FRAMES,
        padding="max_length",
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": audio_out["input_values"][0],
    }

dataset = kss_mod.map(preprocess, remove_columns=kss_mod.column_names)
```

이 함수는 데이터의 최종 형태를 결정하는 핵심 구성 요소다.

- 텍스트 -> 자모 -> 토큰
- 오디오 -> 멜 spectrogram (labels)
- 각 샘플을 {input_ids, attention_mask, labels} 형식으로 통일

이 단계가 끝나면 SpeechT5는 훈련 가능한 포맷으로 입력을 받는다.

## DataCollator - batch 생성기

```python
@dataclass
class DataCollator:
    spk_emb: torch.Tensor  # [emb_dim]

    def __call__(self, features):
        # input_ids: 가변 길이 -> pad_sequence 사용
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # labels: 멜 스펙 (이미 동일 길이)
        labels = torch.stack(
            [torch.tensor(f["labels"], dtype=torch.float32) for f in features]
        )
        labels = labels.masked_fill(labels.eq(0.0), -100.0)

        B = len(features)
        spk = self.spk_emb.unsqueeze(0).repeat(B, 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "speaker_embeddings": spk,
        }
```

이 collator는 DataLoader가 batch를 묶어줄 때 사용하는 클래스.

정확한 역할

- variable-length input_ids -> padding
- attention_mask 생성
3. label의 “0.0 패딩 영역”을 -100으로 바꾸어 loss 계산에서 무시
4. 화자 embedding을 batch size 만큼 복제

이 구조는 HuggingFace Trainer 대신 직접 PyTorch 학습 루프를 만들 때 필수다.

## Training Loop - mixed precision + AdamW

```python
with torch.amp.autocast(...):
    out = model(**b)

scaler.scale(out.loss).backward()
scaler.step(optimizer)
scaler.update()
```

AMP(mixed precision)의 장점

- GPU 메모리 절약
- 연산 속도 증가
- 안정적인 SpeechT5 학습

PyTorch에서는 gradient scaling이 필수이므로
GradScaler()를 함께 사용한다.

또한 수동 학습 루프를 선택한 이유:

- TTS 손실 구조가 특수함
- 커스텀 collator 필요
- 향후 multi-speaker 지원 등 확장성 확보

## checkpoint 저장 및 최종 모델 패키징

```python
model.save_pretrained(...)
tokenizer.save_pretrained(...)
vocoder.save_pretrained(...)
torch.save(speaker_embedding)
```

HuggingFace Hub에 올릴 수 있는 모델 구조를 그대로 생성한다.

장점

- from_pretrained()으로 손쉽게 inference 가능
- demo 스크립트 재사용성 강함
- Hub space에서 바로 테스트 가능

## demo_inference.py 자동 생성

이 파일은 사용자가 바로 inference 할 수 있도록 자동 생성된다.

핵심 기능

- 모델 + vocoder + tokenizer 로드
- 입력 텍스트 자모 분해
- model.generate_speech() 호출
- wav 파일 저장

훈련된 모델을 사용하는 모든 환경에서 이 스크립트 한 개면 충분하다.

## HuggingFace Hub 업로드

```python
api.create_repo(..., exist_ok=True)
api.upload_folder(...)
```

이 두 줄이 “배포 자동화의 핵심”이다.

- repo 없으면 생성
- 모델 전체 폴더 업로드

CI/CD 느낌의 자동화가 이미 구현된 것이다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn