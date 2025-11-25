---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 오디오 분류 아키텍처"
date:   2025-11-25 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 분류 아키텍처

이 글에서는 Spectrogram CNN -> AST -> Wav2Vec2 기반 분류를 다룬다.

## 오디오 분류란 무엇인가?

오디오 분류의 목적은 입력 오디오가 어떤 클래스에 속하는지 예측하는 것이다.

오디오 분류는 두 종류로 나뉜다/

### 전체 오디오 단일 레이블 예측 (Clip-level Classification)

예시

- 어떤 새가 우는 소리인지
- 음악 장르 분류
- 비명 소리/경보음 탐지

출력 예시

- 오디오 전체 -> "sparrow"

### 시간 프레임 단위 레이블 예측 (Frame-level Classification)

예시

- Speaker diarization (누가 말하는지 시점별로 분류)
- 음성 활동 감지(VAD)
- 특정 음향 이벤트가 언제 발생했는지 검출

출력:

- 20ms마다 하나의 레이블 시퀀스

이처럼 오디오 분류는 이미지 분류처럼 단일 결과를 뽑을 수도 있고,
자막처럼 시간 축에 따라 시퀀스 출력도 가능하다.

## 스펙트로그램을 사용한 오디오 분류

오디오 분류의 가장 쉬운 접근은
"스펙트로그램을 이미지로 간주하고 CNN/ViT 분류기를 사용한다"는 것이다.

### 스펙트로그램 복습

스펙트로그램은 음성을
**(시간 × 주파수)**로 펼친 2D 이미지이다.

- 가로축: 시간(time)
- 세로축: 주파수(frequency)
- 픽셀값: 에너지(강도)

즉, 오디오를 시간/주파수 이미지로 변형한다.
이 이미지는 CNN이 매우 잘 처리하는 구조다.

### “오디오 스펙트로그램을 이미지처럼 분류”하는 방법

1. 오디오 -> 멜 스펙트로그램 변환
2. 스펙트로그램을 이미지로 간주
3. ResNet/VGG/ViT 같은 이미지 모델에 입력
4. 마지막 FC 레이어만 교체해 fine-tuning

이렇게 하면 예상보다 훨씬 정확한 오디오 분류기가 된다.

### 하지만 스펙트로그램은 “진짜 이미지”와 같지 않다

이미지는 위아래로 움직여도 의미가 크게 바뀌지 않지만,
스펙트로그램은 위아래로 움직이면 주파수 자체가 바뀌므로 소리의 성질이 바뀐다.

예시

- 스펙트로그램을 위로 이동하면 고음이 되고
- 아래로 이동하면 저음이 된다.

즉, 스펙트로그램은 시각적 표현은 이미지 같지만, 의미는 오디오이다.

이미지 증강 기법을 그대로 쓰면 잘못된 특성 학습이 발생하기도 한다.
(예시: vertical flip, random crop-frequency 변형 등)

## Audio Spectrogram Transformer(AST) - ViT 기반 오디오 분류기

오디오 스펙트로그램 트랜스포머(AST)는
ViT를 오디오 스펙트로그램에 그대로 적용한 모델이다.

<div align="center"> 
<img src="https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/ast.png" width="650"/> 
</div>

### AST 작동 방식

1. 멜 스펙트로그램을 이미지처럼 취급
2. 16×16(혹은 비슷한 크기)의 패치로 분할
3. 각 패치를 벡터 임베딩
4. 트랜스포머 인코더에 전달 (Encoder-only 구조)
5. 마지막 hidden state를 분류 헤드로 전달
6. 시그모이드 또는 소프트맥스로 클래스 확률 출력

### AST의 장점

- CNN보다 글로벌 컨텍스트(전체 시퀀스)를 잘 본다.
- Transformers는 긴 시퀀스의 패턴 및 의미를 잡는 데 강하다.
- 다양한 오디오 태스크에서 SOTA 성능 달성한다.

## “모든 트랜스포머는 분류기가 될 수 있다”

CTC처럼 인코더-only 모델(Wav2Vec2, HuBERT 등)은
원래부터 숨겨진 상태를 확률 분포로 바꿔 출력하는 구조라
분류기로 바꾸는 것이 매우 쉽다.

## Wav2Vec2를 오디오 분류기로 쓰는 방법

HF에는 이미 여러 버전의 Wav2Vec2 모델이 있다.

- Wav2Vec2ForCTC: 문자 예측
- Wav2Vec2ForSequenceClassification: 전체 오디오 단일 레이블
- Wav2Vec2ForAudioFrameClassification: 프레임 레이블 시퀀스 출력

구조는 거의 동일하고, 출력 레이어와 손실 함수만 다르다.

### 전체 시퀀스 분류 (Sequence Classification)

출력 예시

```python
"dog barking"
"engine idling"
"keyword: 'Hey Siri'"
```

사용 모델

```python
Wav2Vec2ForSequenceClassification
```

원리

- hidden_states -> 평균(mean pooling) -> classification head


출력

softmax로 단일 클래스 확률 분포

### 프레임 단위 분류 (Audio Frame Classification)

출력 예시

```python
프레임1: speaker A  
프레임2: speaker A  
프레임3: speaker B  
```

사용 모델

```python
Wav2Vec2ForAudioFrameClassification
```

원리

- hidden_states (T × D) -> linear -> (T × classes)

즉, 각 타임스탭(time step)에 대한 예측을 수행한다.

대표 응용

- 화자 분리(diarization)
- 오디오 이벤트 탐지(AED)
- VAD(음성/무음 분류)

## 어떤 모델을 선택해야 할까?


| 작업 유형              | 추천 모델                                         | 이유                      |
| ------------------ | --------------------------------------------- | ----------------------- |
| 전체 오디오 -> 하나의 라벨    | Wav2Vec2ForSequenceClassification / AST / CNN | 전체 특징을 집약해서 분류          |
| 시간축에 따라 라벨이 필요함    | Wav2Vec2ForAudioFrameClassification           | 프레임 단위 출력 제공            |
| 아주 긴 오디오 (수 초~수 분) | Wav2Vec2 기반                                   | Raw waveform 인코딩 강함     |
| 소규모 데이터셋           | 스펙트로그램 + CNN/ViT                              | 일반적으로 더 안정적             |
| 다국어·잡음 robust      | Wav2Vec2 XLSR                                 | 대규모 학습된 multilingual 모델 |


## 실전에서 조심할 점

여기부터는 프로젝트에서 직접 적용할 때 반드시 알아야 할 내용이다.

1. 스펙트로그램을 이미지처럼 쓴다고 해서 “그대로 이미지 증강”을 하면 안 된다

- Vertical shift: 주파수 변화로 소리 성질 왜곡
- Horizontal shift: 타이밍 변화로 정답이 달라짐

추천 증강

- 시간 마스킹(Time Mask)
- 주파수 마스킹(Frequency Mask)
- SpecAugment
- Noise 추가

2. Raw waveform 기반 모델(Wav2Vec2)은 주파수 변화에 강하다

Wav2Vec2는 CNN Feature Extractor가 있기 때문에

- pitch shift
- 낮은 sampling rate
- noise

등에 비교적 강하다.

3. 분류 헤드만 바꾸면 거의 모든 트랜스포머가 분류기로 사용 가능

예시

```python
model.wav2vec2.encoder → freeze
model.classifier = nn.Linear(hidden_dim, num_labels)
```

이렇게 하면 few-shot 분류도 가능하다.

## 결론: 오디오 분류 모델은 생각보다 유연하다

스펙트로그램 기반 CNN/ViT은 빠르고 안정적이어서
간단한 프로젝트에 최적이다.

AST는 고성능 transformer 기반 오디오 분류 모델이다.

Wav2Vec2는 raw audio 기반 모델로, 다국어에 대응해야 하거나 노이즈를 처리해야 하는 상황에 강하다.

어떤 트랜스포머든 hidden states 위에 projection layer를 얹으면 분류가 가능히디.

즉, 오디오 분류는 “오디오 -> 이미지 -> 분류” 또는
“오디오 -> 트랜스포머 -> 분류”의 두 가지 큰 축으로 모두 가능하다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn