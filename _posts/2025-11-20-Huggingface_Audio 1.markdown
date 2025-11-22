---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 1일차"
date:   2025-11-20 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 데이터 이해하기

오디오 신호는 원래 연속적인(continuous) 형태지만, 디지털 장치나 머신러닝 모델은 이산(discrete) 값만 처리할 수 있다. 따라서 아날로그 신호를 디지털 표현으로 바꾸는 과정이 필요하며, 이 과정에서 샘플링, 비트뎁스, 스펙트럼, 스펙트로그램 같은 개념이 등장한다.

아래에서는 오디오 데이터가 디지털로 표현되는 전체 과정을 핵심 개념 중심으로 정리한다.

## 아날로그 신호에서 디지털 신호로

아날로그 오디오가 디지털 오디오로 변환되는 과정은 다음과 같다.

1. 마이크가 공기의 압력 변화를 전기 신호로 변환
2. ADC(Analog-to-Digital Converter)가 일정 간격으로 샘플링
3. 샘플들을 정수 또는 부동소수점 값으로 양자화

디지털 오디오 품질은 크게 두 요소로 결정된다.

1. 샘플링 속도(sampling rate): 시간을 얼마나 촘촘히 나누는가
2. 비트뎁스(bit depth): 샘플 값을 얼마나 정밀하게 저장하는가

## 샘플링(sampling)과 샘플링 속도

### 샘플링이란

연속적인 신호를 일정한 간격으로 측정해 숫자 리스트로 만드는 과정이다.
이산화된 숫자들이 바로 디지털 오디오의 **파형(waveform)**을 구성한다.

### 샘플링 속도(sampling rate)

1초 동안 얻는 샘플 개수이며 단위는 헤르츠(Hz)다.  


| 용도       | 샘플링 속도     |
| -------- | ---------- |
| 전화 음성    | 8 kHz      |
| 음성 인식 모델 | 16 kHz     |
| CD 음악    | 44.1 kHz   |
| 고해상도 음악  | 48~192 kHz |


샘플링 속도가 높을수록 높은 주파수까지 표현할 수 있다.
표현 가능한 최고 주파수는 샘플링 속도의 절반이며 이를 **나이퀴스트 한계(Nyquist limit)**라고 한다.

- 예시
16 kHz -> 최대 8 kHz까지 표현 가능
음성은 대부분 8 kHz 미만 -> 음성 모델에서 16 kHz가 널리 쓰이는 이유

### 실무 관점

오디오 데이터셋의 샘플링 속도는 반드시 통일해야 한다.
특히 사전학습된 오디오 모델을 파인튜닝할 때, 사전학습과 동일한 샘플링 속도를 유지해야 한다.
샘플링 속도가 다르면 입력 시퀀스 길이가 달라져 모델이 일반화하기 어렵다.

샘플링 속도를 맞추는 작업을 **리샘플링(resampling)**이라고 한다.

## 진폭(amplitude)과 비트뎁스(bit depth)

### 진폭

진폭은 특정 순간의 소리 **크기(압력)**이며 **데시벨(dB)** 단위를 사용한다.

### 비트뎁스

각 샘플을 얼마나 정밀하게 저장하는가를 나타낸다.


| 비트뎁스        | 가능한 값 개수      |
| ----------- | ------------- |
| 16bit       | 65,536개       |
| 24bit       | 16,777,216개   |
| 32bit float | 부동소수점, 높은 정밀도 |


비트뎁스가 높을수록 양자화 노이즈가 줄어들지만, 16bit만 되어도 음성 작업에는 충분하다.

실제로 머신러닝에서는 오디오는 대부분 부동소수점(float32) 형태로 변환되어 처리된다.
librosa나 torchaudio는 파형(waveform)을 로드할 때 자동으로 float(-1.0~1.0)으로 변환한다.

## 파형(waveform): 시간 영역(time domain) 표현

파형은 시간에 따라 진폭이 어떻게 변하는지를 보여준다.

```python
import librosa


array, sr = librosa.load(librosa.ex("trumpet"))
```

```python
import librosa.display
import matplotlib.pyplot as plt

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sr)
```

파형으로 알 수 있는 정보:
- 구간별 음량 변화
- 무음(silence) 여부
- 노이즈 존재 여부
- 전처리(정규화, 리샘플링 등) 확인 가능

다만 파형만으로는 어떤 음인지(주파수 구성)는 알기 어렵다.
그래서 주파수 영역 분석이 필요하다.

## 주파수 스펙트럼: 주파수 영역(frequency domain) 표현

DFT(Discrete Fourier Transform)은 한 구간의 오디오를 주파수 성분으로 분해하는 알고리즘이다.
FFT는 이를 빠르게 계산하는 방식이다. numpy의 rfft() 함수를 쓰면 DFT를 계산할 수 있다.

```python
import numpy as np

dft_input = array[:4096]

# DFT 계산
window = np.hanning(len(dft_input))
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)

# 데시벨로 진폭 스펙트럼 변환
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# 주파수
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

plt.figure().set_figwidth(12)
plt.plot(frequency, amplitude_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.xscale("log")
```

스펙트럼으로 알 수 있는 정보:
- 기본 주파수(F0)
- 고조파(harmonics) 구성을 통한 음색(timbre)
- 잡음 성분
- 특정 순간의 주파수 구조

하지만 스펙트럼은 시간 변화 정보를 제공하지 않는다.

## 스펙트로그램(spectrogram): 시간 + 주파수

스펙트로그램은 일정한 길이의 창(window) 단위로 DFT를 반복 적용하여
시간축에 따라 주파수 스펙트럼이 어떻게 변하는지를 보여주는 2D 표현이다.

```python
D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()
```

스펙트로그램에서 알 수 있는 것:
- 발음별 주파수 패턴(formant)
- 자음/모음 차이
- 악기별 주파수 대역
- 특정 이벤트(폭발음, 충격음 등)의 발생 시점

스펙트로그램은 2D 이미지에 가깝기 때문에
CNN 또는 비전 트랜스포머(ViT) 입력으로 적합하다.

## 멜 스펙트로그램(mel spectrogram)

사람의 청각은 저주파에 민감하고 고주파에는 둔감하다.
멜(Mel) 스케일은 이러한 인간의 청각 특성을 수학적으로 근사한 축이다.

멜 스펙트로그램은 다음 과정을 거쳐 계산된다.

1. STFT로 스펙트럼 계산
2. mel filterbank 적용
3. 파워 혹은 진폭에 log 변환 -> log-mel spectrogram

```python
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()
```

위의 예에서, n_mels는 mel band의 수를 정합니다. 
mel band는 필터를 이용해 스펙트럼을 지각적으로 의미있는 요소로 나누는 
주파수 범위의 집합을 정의합니다. 

이 필터들의 모양(shape)과 간격(spacing)은 
사람의 귀가 다양한 주파수에 반응하는 방식을 모방하도록 선택됩니다. 

흔히 n_mels의 값으로 40 또는 80이 선택됩니다. 
fmax는 우리가 관심을 가지는 최고 주파수(Hz 단위)를 나타냅니다.

### 멜 스펙트로그램의 특징
- 인간의 청각에 더 가까운 표현
- 음성 인식(ASR)과 TTS 모델에서 사실상의 표준 입력
- 2D 이미지 형태라 모델 입력으로 적합
- mel filterbank 때문에 정보 손실 존재 -> 원래 파형으로 복원이 어려움

따라서 멜 스펙트로그램을 파형으로 복원하려면
HiFi-GAN, WaveGlow 같은 vocoder 모델이 필요하다.

## 마무리

오디오는 파형, 스펙트럼, 스펙트로그램, 멜 스펙트로그램처럼 여러 관점에서 표현할 수 있다.
각 표현은 분석 목적과 모델링 목적에 따라 선택된다.

- Waveform -> 원본 신호 분석
- Spectrum -> 주파수 구조 분석
- Spectrogram -> 시간 변화+주파수
- Mel-spectrogram -> 음성 모델 입력 표준

오디오 기반 머신러닝(ASR, TTS, 음악 분석 등)을 다룬다면
이 기본 개념들은 필수적인 배경지식이다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn