---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 실습 과제 1"
date:   2025-11-24 00:10:22 +0900
categories: Huggingface_Audio
---

# 허깅페이스 실습 과제 1 정리

## 개요

이 미션은 HuggingFace Audio Course의 실습 과제이며,
오디오 데이터셋을 스트리밍 모드로 불러오고,
파형, 스펙트럼, 스펙트로그램 시각화, 자동 음성 인식(ASR)을 수행하는 것을 목표로 한다.

사용한 데이터셋은 다음과 같다.

- facebook/voxpopuli (deprecated)
- 복구 버전: ddyuudd/voxpopuli

## 라이브러리 설치 및 기본 

기본 라이브러리 설치

```python
!pip install librosa
!pip install datasets
```

사용 라이브러리

```python
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import librosa
import librosa.display
from datasets import Audio, load_dataset
from transformers import pipeline
from IPython.display import Audio
```

### 코드 설명

- librosa: 오디오 분석 필수 라이브러리
- datasets: Hugging Face 데이터셋 로더
- pipeline: 다양한 오디오/텍스트 작업을 수행하는 HuggingFace 고수준 API
- islice: 스트리밍 모드에서 인덱싱이 불가하므로 대체 접근 방식

## 스트리밍 모드로 오디오 데이터셋 불러오기

```python
voxpopuli_en = load_dataset("ddyuudd/voxpopuli", streaming=True)
train = voxpopuli_en["train"]
```

### 스트리밍 모드 특징

- 전체 파일 다운로드 X
- "필요한 샘플"만 즉석에서 로드
- 대규모 오디오에 매우 유리
- 인덱싱 불가 -> 따라서 next(iter()) 또는 islice() 사용해야 함

## 스트리밍 모드에서 데이터 접근하기

첫 번째 데이터 가져오기

```python
item = next(iter(train))
```

3번째 데이터 가져오기 (for-loop 방식)

```python
for i, item in enumerate(train):
    if i == 2:
        example = item
        break
```

3번째 데이터 가져오기 (islice 방식)

```python
third_item = next(islice(train, 2, 3))
```

### 추가 정보

스트리밍 모드는 데이터셋이 파이썬 iterator처럼 동작한다.
사용 데이터셋의 특징상 각 데이터에 접근하려면 train = voxpopuli_en["train"]으로 다시 불러야 한다.

## 오디오 재생

```python
Audio(third_item['audio']['array'], rate=16000)
```

### 오디오 포맷

- array: float32 waveform
- sampling_rate: 16 kHz로 리샘플링된 버전

## 파형(waveform) 시각화

```python
array = third_item['audio']['array']
sr = third_item['audio']['sampling_rate']

plt.figure().set_figwidth(16)
librosa.display.waveshow(array, sr=sr)
```

### 파형에서 알 수 있는 것

- 음량 변화
- 무음 구간
- 노이즈 여부
- 전체 발화 길이
- clipping 여부

## 주파수 스펙트럼(DFFT) 시각화

```python
window = np.hanning(len(array))
windowed_input = array * window
dft = np.fft.rfft(windowed_input)

amp = np.abs(dft)
amp_db = librosa.amplitude_to_db(amp, ref=np.max)

frequency = librosa.fft_frequencies(sr=sr, n_fft=len(array))

plt.figure().set_figwidth(16)
plt.plot(frequency, amp_db)
plt.xlabel("Frequency")
plt.ylabel("Amp (db)")
plt.xscale("log")
```

### 중요 개념 설명

- Hanning window: STFT(Short-Time Fourier Transform) 준비를 위해 신호 양 끝단이 매끄럽게 연결되도록 감쇠
- FFT -> rfft(): 복소수 푸리에 변환의 절반만 취한 실수 기반 FFT
- amplitude_to_db(): 선형 진폭 -> 로그 스케일(dB) 변환
- log scale x-axis: 인간의 청각 특성(고주파 둔감, 저주파 민감)에 적합

## 스펙트로그램(STFT) 시각화

```python
stft = librosa.stft(array)
s_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

plt.figure().set_figwidth(16)
librosa.display.specshow(s_db, x_axis="time", y_axis="hz")
plt.colorbar()
```

### 스펙트로그램 특징

- x축: 시간
- y축: 주파수
- 색깔: 진폭(dB)
- STFT를 시각화해서 얻는 결과

이것은 "주파수 변화"를 볼 수 있어 음성 분석에서 필수적이다.

## 멜 스펙트로그램 시각화

```python
mel_s = librosa.feature.melspectrogram(y=array, sr=sr, n_mels=80, fmax=8000)
mel_s_db = librosa.power_to_db(mel_s, ref=np.max)

plt.figure().set_figwidth(16)
librosa.display.specshow(mel_s_db, x_axis="time", y_axis="mel", sr=sr, fmax=8000)
plt.colorbar()
```

### mel-spectrogram을 쓰는 이유

- 인간 청각 체계를 기반으로 한 mel scale 사용
- ASR(Audio Speech Recognition)의 핵심 입력
- Whisper, Wav2Vec2 등 모델들이 mel spectrogram 또는 raw waveform을 사용

## 자동 음성 인식(ASR) 실행

파이프라인 생성

```python
my_asr = pipeline("automatic-speech-recognition")
```

기본적으로 Whisper 기반 영어 모델을 자동 선택한다.

예측과 정답 실행

```python
print(my_asr(array))
print(third_item["text"])
```

ASR의 품질을 비교해볼 수 있다.

## 전체 흐름 요약


| 단계 | 설명                                           |
| -- | -------------------------------------------- |
| 1  | 대규모 음성 데이터셋 스트리밍 로드                          |
| 2  | 스트리밍 모드 전용 데이터 접근 방식 익히기                     |
| 3  | waveform / spectrogram / mel-spectrogram 시각화 |
| 4  | Whisper 기반 ASR 모델로 음성 -> 텍스트 변환               |
| 5  | 예측 결과와 실제 텍스트 비교                             |


## 추가 팁

- 스트리밍 모드는 메모리 절약 + 대규모 데이터셋 처리 핵심

특히 GigaSpeech·CommonVoice·LibriLight 같은 TB 단위 데이터셋에서 필수다.

- STFT 파라미터(n_fft, hop_length)는 성능에 큰 영향

hop_length는 ‘시간축’을 얼마나 촘촘하게 샘플링할지를 결정한다.
작을수록 더 자주 STFT를 계산하고 시간 변화는 더 세밀하게 보인다.

n_fft는 ‘주파수축’을 얼마나 세밀하게 나눌지를 결정한다.
클수록 FFT 출력 bin 개수가 많아지고 주파수 해상도는 증가한다.

이 둘은 트레이드오프 관계에 있다.

- mel-spectrogram은 ASR 모델의 표준 입력

특히 Whisper는 80 mel bands × 3000 frames 고정 입력 구조를 사용한다.

- pipeline()은 간단하지만 속도는 느릴 수 있다

대규모 inference 작업에서는
processor + model.forward() 형태를 직접 사용하는 편이 성능상 유리하다.


참고자료
Huggingface, Audio Course, https://huggingface.co/learn