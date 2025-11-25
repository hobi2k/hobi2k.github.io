---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 오디오를 위한 트랜스포머 아키텍처"
date:   2025-11-25 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오를 위한 트랜스포머 아키텍처

트랜스포머(Transformer)는 원래 NLP(자연어 처리)를 위해 개발된 모델이지만,
현재는 음성 인식(ASR), 음성 합성(TTS), 오디오 분류, 음성 변환, 음성 향상 등
거의 모든 오디오 작업의 핵심 구조가 되었다.

오디오 모델이라고 해서 구조가 특별히 달라지는 것은 아니다.
핵심은 Transformer 백본이며, 입력과 출력 단계에서 오디오에 맞는 전처리와 후처리만 달라진다.

오디오 트랜스포머를 완전히 이해하려면 두 가지가 핵심이다.

- **임베딩(Embedding):** 오디오를 Transformer가 처리할 수 있는 벡터 시퀀스로 만드는 과정
- **어텐션(Attention):** 시퀀스를 전체적으로 바라보며 중요한 부분에 집중하는 핵심 연산

아래에서는 임베딩과 어텐션을 중심으로 음성 분야의 트랜스포머 구조를 상세히 설명한다.

## Transformer 기본 구조

Transformer는 크게 두 부분으로 구성된다.

### Encoder

입력을 이해하고 의미 있는 표현(embedding sequence)을 만든다.

### Decoder

Encoder의 표현을 참고해 다음 토큰(또는 스펙트로그램)을 생성한다.

대표적인 예시

- BERT = Encoder-only
- GPT = Decoder-only
- Whisper, T5 = Encoder + Decoder

구조와 사용 목적은 다르지만, 전체적인 처리 흐름은 같다.

- 입력 -> 임베딩 -> 어텐션 -> 출력

## 임베딩(Embedding)이란 무엇인가?

Transformer는 다음과 같은 입력 형태만 처리할 수 있다.

**(batch_size, sequence_length, embedding_dim)**

즉, 텍스트든 오디오든 임베딩 벡터의 시퀀스로 변환해야 한다.
임베딩은 단순한 숫자 변환이 아니라 오디오 신호의 핵심 패턴을 의미 있는 벡터 공간에 표현하는 과정이다.

### 임베딩이 하는 일

- 비슷한 발음 -> 가까운 벡터
- 다른 발음 -> 멀리 떨어진 벡터
- 잡음은 벡터 공간의 구석으로 밀려남
- formant, pitch, timbre 같은 음향 특징도 간접적으로 표현됨

Wav2Vec2, Whisper 같은 모델이 강력한 이유는
바로 이 임베딩 공간 자체가 매우 잘 학습되어 있기 때문이다.

## 오디오 임베딩 생성 방식

트랜스포머는 원시 파형(raw waveform)을 직접 처리하기 어렵다.
따라서 오디오 모델은 전처리를 통해 입력을 임베딩 시퀀스로 변환해야 한다.

대표 방식은 두 가지이다.

### Raw Waveform -> CNN Feature Encoder (Wav2Vec2, HuBERT)

**입력 형태:**

- 1D waveform (예: 30초 -> 480,000 samples)

**변환 과정:**

- waveform -> 정규화 -> 1D CNN 여러 층 -> subsampling -> embedding sequence

**CNN을 사용하는 이유**

- waveform은 너무 길며, Transformer는 O(n<sup>2</sup>)이므로 비효율적이다.
- CNN으로 길이를 줄이면서도 음성 특징(발음, formant)을 압축 및 보존할 수 있다.
- 일반적으로 약 20~25ms마다 하나의 feature 벡터를 생성한다.

즉 CNN은 waveform을 짧은 시간 단위의 벡터 시퀀스로 변환한다.  
(한 벡터가 한 프레임의 음성표현)

### Spectrogram -> CNN -> Transformer (Whisper)

Whisper는 waveform을 바로 사용하지 않고 스펙트로그램을 만든다.

**Whisper 입력 생성:**

- waveform -> STFT -> Mel filterbank -> Log -> (80 bins × 3000 frames) mel spectrogram

Whisper는 항상 30초 단위로 오디오를 처리하며 다음 형태를 사용한다.

```python
mel: (80, 3000)
```

멜 스펙토그램을 입력으로 변환하면 다음과 같이 진행된다.

- mel spectrogram -> 2D CNN -> embedding -> Transformer Encoder/Decoder

**장점**

- waveform보다 훨씬 더 압축된다.
- 시퀀스 길이가 짧아 Transformer 계산량이 줄어든다.
- mel-scale은 인간 청각 구조를 반영해 성능이 향상된다.

**단점**

- 위상(phase) 정보가 사라지므로 vocoder가 필요하다(STFT의 반대 과정(ISTFT))

## 어텐션(Attention), Transformer의 핵심

어텐션은 Transformer의 “두뇌”이다.
시퀀스 전체를 동시에 보며 중요한 부분에 높은 가중치를 주고 불필요한 부분은 무시한다.

### Attention이 필요한 이유

음성은 다음과 같이 다양한 시간 규모의 패턴이 존재한다.

- 자음의 burst(수 ms)
- 모음 formant(수백 ms)
- 단어, 억양(pattern)이 수 초 단위로 유지
- 잡음을 간헐적으로 포함

즉, 짧은 패턴과 긴 패턴이 동시에 존재한다.
RNN이나 CNN은 이 조합을 완벽히 처리하기 어렵다.
하지만 Attention은 프레임 간 거리에 관계없이 연결하기 때문에 유리하다.

### Attention 작동 방식

예를 들어 다음과 같은 입력 시퀀스가 있다.

```python
x1, x2, x3, ..., xT
```

각 xi는 임베딩 벡터이다.

**Step 1. Q, K, V 생성**

각 프레임을 다음 3개의 벡터로 변환한다:

- **Query(Q):** 내가 찾고 싶은 정보
- **Key(K):** 내가 어떤 정보를 가지고 있는가
- **Value(V):** 실제 정보 값

**Step 2. Attention Score 계산**

i번째 프레임이 j번째 프레임을 얼마나 “참조할지” 점수 계산:

score(i, j) = (Q_i · K_j) / sqrt(d)

- 여기서 d는 임베딩 차원 수

**Step 3. Softmax로 가중치 만들기**

α(i, j) = Softmax(score)

**Step 4. Value를 가중합**

output_i = Σ_j α(i, j) * V_j

- i: 현재 “관심을 주고 있는” 토큰
- j: 다른 모든 토큰들
- V_j: j번째 토큰의 Value 벡터
- α(i, j): i가 j를 “얼마나 참고할지”를 나타내는 attention weight (softmax 결과)

결과적으로 모든 프레임이 다른 모든 프레임과 연결되어 정보를 공유한다.

### 오디오에서 Attention이 특히 중요한 이유

1. burst(자음)고 formant(모음)를 동시에 고려할 수 있다.
2. 멀리 떨어진 패턴도 즉시 연결할 수 있다. (RNN의 한계를 극복)
3. 노이즈 구간을 자동으로 무시한다.
4. 다양한 언어와 억양(prosody)에 강하다.
5. TTS에서 억양(prosody)을 조절할 때도 필수이다.
6. Whisper의 억양 및 발화 스타일 인식 능력이 이 구조에서 나온다.

## Transformer 출력 방식

Transformer는 항상 “은닉 상태(hidden state) 시퀀스”를 출력한다.
이 벡터들을 후처리하여 원하는 형태로 만들어야 한다.

### ASR (텍스트 출력)

- hidden states -> Linear -> Softmax -> 문자/토큰 확률

Whisper의 decoder가 이 방식을 사용한다.

### TTS (스펙트로그램 출력)

- Transformer -> Linear -> Mel-spectrogram -> Post-Net -> Vocoder -> Waveform
- Post-Net: 스펙트로그램 품질 보정
- Vocoder: mel-spectrogram을 실제 waveform으로 변환

### 오디오 분류

- hidden states -> Pooling -> Linear -> Class probabilities

## Vocoder가 필요한 이유

STFT 기반 스펙트로그램에는 다음 두 정보가 필요하다:

- Amplitude(진폭)
- Phase(위상)

하지만 대부분의 음성 생성 모델(TTS)은 위상 정보를 예측하지 않는다.

- 위상은 복잡하고 불안정
- 예측하기 어려움

따라서 vocoder가 위상을 추정하고 waveform을 생성해야 한다.

대표 vocoder:

- HiFi-GAN
- WaveRNN
- WaveGlow

## 전체 요약

Transformer는 Attention을 기반으로 한 시퀀스 모델이다.
오디오 입력은 waveform 또는 spectrogram을 임베딩 벡터 시퀀스로 변환해야 한다.

오디오 Transformer의 핵심은 다음 3단계다.

1. 입력 임베딩 생성 (waveform -> CNN / spectrogram -> CNN)
2. Transformer에서 Attention 연산 수행
3. hidden states를 텍스트/스펙트로그램 등 원하는 출력으로 변환

Whisper, Wav2Vec2, HuBERT, SpeechT5 등은 모두 Transformer 백본은 동일하며
입력 전처리와 출력 후처리만 다르다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn