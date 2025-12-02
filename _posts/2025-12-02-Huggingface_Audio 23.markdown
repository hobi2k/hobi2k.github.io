---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 사전학습 TTS 모델"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_Audio
---

# 사전학습 TTS 모델

Text-to-Speech(TTS)는 단순히 텍스트를 음성으로 변환하는 것이 아니라, 텍스트 기반 언어적 맥락, 화자 특성, 음성 구조, 음향 구조까지 모두 포함하는 다단계 생성 모델이다.

그리고 여기에 필요한 모든 요소가 모델 안에서 특정 기술 단위로 분해되어 존재한다.
SpeechT5, Bark, MMS(VITS), HiFi-GAN, X-Vectors는 이 거대한 파이프라인의
서로 다른 부분을 맡는 기술 블록이다.

이 글은 이 각각이 왜 필요하고, 서로 어떻게 연결되며, 내부적으로 어떤 역할을 하는지  정리한다.

## 대규모 TTS 모델이 적은 이유 - 구조적 난이도

ASR과 다르게, TTS는 다음 문제 때문에 선행 모델이 적다.

1. One-to-many 매핑 문제

- 텍스트 "Hello"는 수천 가지 음성으로 발화할 수 있다.
- 목소리, 억양, 속도, 감정, 녹음 환경, 화자 신체 특성 모두 포함.
- “정답 데이터가 하나가 아닌 문제”여서 학습이 어렵다.

2. 음성 생성은 신호 처리 난이도가 높다

- 음성은 16,000~48,000 sample/second
- 파형을 직접 생성하려면 초당 수만 개 값 예측 필요
- 언어 모델보다 훨씬 고차원

3. 음성과 텍스트는 서로 다른 modality

- 텍스트는 discrete token sequence
- 음성은 continuous time series
- 두 모달리티를 매핑하는 건 단순 Transformer보다 훨씬 복잡

4. 데이터 품질 영향도가 매우 큼

- 음성 잡음, 마이크 품질, room reverb, breath noise
- 작은 문제라도 TTS 결과 음질이 크게 망가짐
- 고품질 멀티스피커 데이터는 구하기 매우 어렵다

이 복잡한 문제를 해결하기 위해 등장한 것이
SpeechT5, X-Vectors, HiFi-GAN, Bark, MMS(VITS) 같은 모델이다.

## SpeechT5

SpeechT5는 텍스트와 음성을 모두 다룰 수 있는 멀티모달 Transformer이다.

핵심 아이디어는 이렇다.

“텍스트와 음성은 다르지만, Transformer 안에서는 공통된 ‘의미 표현’ 공간으로 변환할 수 있다.”

그래서 SpeechT5는 단일 백본으로 다음 모든 작업을 처리한다.

- Text -> Speech (TTS)
- Speech -> Text (ASR)
- Speech -> Speech (Voice Conversion, Enhancement)
- Speech -> Speaker Identity (speaker ID)

이 모든 것을 단일 모델이 한다.

### SpeechT5의 핵심: Pre-net / Post-net 시스템

SpeechT5는 Transformer Backbone 앞뒤로
6개의 모달 전용 Pre-Net과 Post-Net을 둔다.

왜 필요한가?

| 모달       | 형태              | 문제                                |
| -------- | --------------- | --------------------------------- |
| 텍스트      | 정수 토큰           | Transformer로 바로 넣으면 정보가 너무 분산 |
| 음성       | Mel Spectrogram | 차원이 매우 높음, normalization 필요       |
| Waveform | 수만 개 샘플         | Transformer가 직접 처리 불가능            |


Pre-netd은 이것을 Transformer가 이해할 수 있는 hidden state로 바꿔준다.

그리고 Transformer가 만든 hidden state를
다시 음성/텍스트 형태로 변환해야 하는 게 Post-net이다.

그래서 TTS는 다음 병렬 구조를 가진다.

TEXT -> Text Pre-net -> Transformer -> Speech Pre-net -> Mel Spectrogram -> (HiFi-GAN) -> Waveform


이 구조를 이해하면
왜 SpeechT5가 멜 스펙트로그램을 출력하고, 보코더가 필요한지 명확해진다.

## Speaker Embeddings - 화자 동일성을 수학적으로 표현하는 기술

TTS에서 “이 목소리로 말해줘”를 가능하게 하는 기술이 Speaker Embeddings이다.

Speaker Embedding은 다음 정보를 하나의 고정 길이 벡터로 요약한 것이다.

- 음색(timbre)
- 발화 속도
- 억양 패턴
- 포먼트(formant) 구조
- 특징적인 말투

즉, 사람이 말할 때의 모든 개성을 “디지털 DNA”처럼 저장한 것이 speaker embedding이다.

### X-Vectors

X-vector는 현대 음성 생성, 음성 인식, 화자 인식 시스템의 핵심이다.

X-vector는 어떻게 만들어지는가?

1. CNN/TDDN 기반 네트워크가
오디오를 프레임 단위로 처리한다.

2. Time pooling을 통해
전체 구간의 정보를 집계(summarize)

3. 512차원 벡터로 압축

이 벡터가 X-vector이다.

왜 X-vector가 중요한가?

- 화자 인식(Speaker ID, Verification)의 SOTA
- Voice conversion의 표준 입력
- SpeechT5, Bark, VITS 모두 X-vector 사용
- 다른 화자의 목소리를 “학습 없이 복제” 가능
- 멀티스피커 TTS의 필수 요소

실제로 SpeechT5 튜토리얼에서도
X-vector를 불러와 목소리 바꾸는 예제를 제공한다.

## Vocoder - HiFi-GAN

TTS 모델 대부분은 스펙트로그램을 출력한다.
스펙트로그램은 사람이 들을 수 없다.

따라서 이를 Waveform으로 변환하는 단계가 반드시 필요하다.

이 역할을 담당하는 것이 Vocoder이다.

### 왜 WaveNet 대신 HiFi-GAN인가?

WaveNet은 훌륭했지만 너무 느렸다.
HiFi-GAN은 다음 문제를 해결했다.

- WaveNet보다 수백 배 빠름
- Waveform 품질 WaveNet과 거의 동급
- 실시간 TTS 가능
- 연산량 적음
- 멀티스피커 모델에도 잘 맞음

그래서 지금 업계 TTS 대부분은 HiFi-GAN 기반이다.

### HiFi-GAN 구조

**Generator**

- Mel Spectrogram 입력
- Conv stack을 거쳐 Waveform 생성
- 목적: “진짜 음성처럼 보이는 오디오 만들기”

**Two Discriminators**

- Multi-Scale Discriminator: 다양한 해상도에서 오디오 검사
- Multi-Period Discriminator: 주기(periodicity)를 관찰하여 성문음, 발음 떨림 등을 평가
(보컬, 음성의 사실감에 매우 중요)

두 discriminator가 GAN 훈련 방식으로
generator의 출력을 계속 까다롭게 테스트하며
결국 고품질의 자연스러운 오디오가 만들어진다.

## Bark

SpeechT5는 스펙트로그램을 생성하고 vocoder가 필요하다.
하지만 Bark는 vocoder가 필요 없다.

왜냐하면 Bark는 EnCodec이라는 뉴럴 오디오 코덱을 사용하기 때문이다.

### EnCodec의 핵심 개념

EnCodec은 음성을 “정수 codebook 토큰”으로 압축한다.

- Waveform -> EnCodec Encoder -> Hidden Codebook
- Codebook은 integer vector
- Transformer는 integer를 다루는 데 최적화됨
- Bark는 codebook을 직접 생성
- EnCodec Decoder로 Waveform 복원

Bark는 내부적으로 이렇게 동작한다:

Text -> Semantic Tokens -> Acoustic Coarse Tokens -> Acoustic Fine Tokens -> EnCodec -> Waveform

### Bark의 강점

- 멜 스펙트로그램 필요 없음
- 보코더 필요 없음
- 텍스트 기반 제어(감정, 사운드 이펙트, 웃음, 한숨, 음악)
- 다국어
- 매우 자연스러운 발성
- 음성 표현 능력 뛰어남(“스타일”을 잘 표현함)

## MMS

만약 한국어, 일본어, 독일어, 프랑스어 등
다국어 TTS를 해야 한다면 MMS는 좋은 선택이다.

MMS는 현재 지구상에서 언어 수가 가장 많은 TTS 모델이다.

이 모델의 기반은 VITS이다.

## VITS - End-to-end TTS의 결정판

VITS는 TTS 모델 설계의 많은 문제를 해결한 혁신적 모델이다.
세부적으로는 아래 기술 3가지를 통합했다.

### Flow-based Length Regulator

기존 FastSpeech 계열은 길이 조절을 위해 duration predictor를 별도로 학습했다.
VITS는 이것을 probabilistic flow-based model로 대체한다.

- 문장 길이 변화
- 음절, 음소 길이
- 강세, 억양 조절이 자연스럽게 모델 내에서 처리됨.

### VAE-like Acoustic Latent

텍스트 -> Acoustic Latent Space로 매핑
이 latent는 스펙트럼과 발화 특성 전체를 담는다.

### HiFi-GAN 기반 Decoder 내장

VITS의 decoder는 HiFi-GAN의 구조를 흡수한 형태다.
따라서 vocoder가 필요 없다.

## 어떤 모델을 언제 써야 하는가?


| 모델             | 장점                             | 단점                 | 언어    | 보코더 |
| -------------- | ------------------------------ | ------------------ | ----- | --- |
| **SpeechT5**   | 안정적, 구조 명확, X-vector로 화자 제어 좋음 | 보코더 필요(HiFi-GAN)   | 영어 중심 | 필요  |
| **Bark**       | 감정·효과음·음악·다국어·스타일 모두 가능        | 학습·튜닝 어려움, GPU 요구  | 다국어   | 불필요 |
| **MMS (VITS)** | 1100+ 언어 지원, 고품질               | 컨트롤 자유도가 낮음        | 세계 최다 | 불필요 |
| **VITS (일반)**  | 빠름, 고품질, vocoder 불필요           | 멀티스피커는 X-vector 필요 | 다양    | 불필요 |
| **HiFi-GAN**   | 사실상 표준 vocoder                 | 스펙트로그램 필수          | -     | -   |
| **X-Vectors**  | 화자 정체성 제어의 핵심                  | 프레임 정렬 필요          | -     | -   |


## 결론

TTS 전체 시스템은 아래처럼 구성된다.

TEXT -> Linguistic Representation -> Acoustic Representation -> Waveform


그런데 이 3단계를
각 모델이 다음과 같이 맡는다.


| 단계     | 기술                                                             |
| ------ | -------------------------------------------------------------- |
| 텍스트 처리 | SpeechT5 Text Pre-net, Bark Semantic, VITS Encoder             |
| 화자 정보  | X-Vector                                                       |
| 음향 생성  | SpeechT5 Decoder (Mel), Bark Coarse/Fine, VITS Acoustic Latent |
| 보코더    | HiFi-GAN 또는 EnCodec                                            |
| 최종 파형  | Bark / VITS는 직접 생성, SpeechT5는 vocoder 필요                       |


참고자료
Huggingface, Audio Course, https://huggingface.co/learn