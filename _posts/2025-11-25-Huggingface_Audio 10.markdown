---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - Seq2Seq 아키텍처 정리"
date:   2025-11-25 00:10:22 +0900
categories: Huggingface_Audio
---

# Seq2Seq 아키텍처 정리

Whisper, TTS, 번역, 요약까지 가능한 Transformer 구조를 살펴본다.

## Seq2Seq란 무엇인가?

CTC 모델은 Encoder-only Transformer였다.
하지만 트랜스포머 구조는 원래 Encoder + Decoder가 세트이며,
이 두 부분을 모두 쓰면 이를 Sequence-to-Sequence (Seq2Seq) 모델이라고 한다.

### Seq2Seq 모델의 목적

“어떤 시퀀스를 받아 다른 시퀀스로 변환한다.”

즉, 입력 시퀀스와 출력 시퀀스의 길이가 달라도 된다.

### 대표적인 응용

- 번역 (한국어 -> 일본어)
- 요약 (긴 텍스트 -> 짧은 문장)
- ASR (오디오 -> 텍스트)
- TTS (텍스트 -> 오디오 스펙트로그램)
- 대화형 LLM(GPT, Gemini의 디코더는 auto-regressive 구조)

입력과 출력 길이가 달라야 하는 문제에서는
Encoder-only 모델보다 훨씬 자연스럽고 강력하다.

## Encoder-only(CTC)보다 Seq2Seq가 강력한 이유

CTC 모델은 각 프레임마다 하나의 문자를 출력해야 해서
입력/출력 길이는 거의 비례한다.

예시

- 1초 오디오(50 프레임) -> 문자 50개 예측 -> CTC로 중복 제거

Seq2Seq에는 그런 제약이 없다.

- 입력 길이 50 -> 출력 길이 10 (요약)
- 입력 길이 300 -> 출력 길이 20 (ASR)
- 입력 길이 20 → 출력 길이 500 (TTS 스펙트로그램)

이렇게 “입력/출력 길이 비례 강제”가 없기 때문에
모델이 훨씬 언어모델적이고 표현 능력이 뛰어나다.

## Whisper 아키텍처 이해하기

Whisper는 Seq2Seq ASR의 대표적 모델이다.

<div align="center"> 
<img src="https://huggingface.co/blog/assets/111_fine_tune_whisper/whisper_architecture.svg" width="650"> 
</div>

## Whisper Encoder: Mel Spectrogram -> Hidden States

Whisper의 입력은 Mel Spectrogram이다.
여기서 사람들이 자주 헷갈리는 *스펙트로그램과 멜 스펙트로그램*을 정리해 보자.

### 스펙트로그램이란?

- 오디오를 시간 + 주파수로 펼쳐 놓은 이미지.
- 가로축: 시간(time)
- 세로축: 주파수(frequency)
- 픽셀값: 특정 시간 및 주파수에서의 에너지 크기

즉 스펙토그램은 “시간마다 어떤 주파수의 소리가 얼마나 큰지”를 보여주는 2D 그림이다.

오디오를 시간–주파수로 해석할 수 있게 하는 핵심 도구라 할 수 있다.

### 멜 스펙트로그램은 무엇이 다른가?

사람 귀는 저주파에는 민감하고 고주파에는 둔감하다.

이 청각적 특성을 반영한 주파수 변환 스케일이 바로 멜 스케일(Mel Scale)이다.

멜 스케일로 변환하면 다음과 같이 된다.

- 고주파수의 불필요한 해상도가 감소
- 음성 인식 및 TTS에서 더 인간지향적 표현
- 모델 입력 크기가 줄어 학습이 안정됨

Whisper는 “로그 멜 스펙트로그램(log-mel)”을 입력으로 사용한다.
로그를 취하면 소리의 크기를 더 균형 있게 반영할 수 있다.

## Encoder의 역할

Encoder는 멜 스펙트로그램을 입력받아 다음과 같이 변형한다.

```python
[time_steps, mel_dim] → [time_steps_encoded, hidden_dim]
```

예시

```python
[300, 80] → [150, 1024]
```

즉, 인코더는 오디오 전체 의미를 담은 숨겨진 상태 벡터 시퀀스를 만든다.

이 벡터들은 “음성 의미의 압축 표현”이라고 보면 된다.

## 크로스 어텐션(Cross Attention)

Seq2Seq의 핵심은 Cross Attention이다.

### Self-Attention (디코더 내부)

“내가 만든 이전 토큰들끼리 서로 참고하는 주의 메커니즘”

### Cross-Attention (Encoder -> Decoder)

“디코더가 새로운 토큰을 만들 때,
인코더의 전체 오디오 표현을 참고한다.”

예시

“45번째 출력 토큰을 만들 때
인코더의 13번째 프레임은 중요해.
48번째 프레임은 별로 중요하지 않아.”

이런 가중치를 Attention Score로 계산한다.

즉, Cross Attention은 
“오디오의 어떤 시점이 지금 생성할 텍스트와 
가장 관련이 있는지”를 계산하는 메커니즘이다.

Whisper가 정확도가 높은 이유가 바로 이 Cross Attention 덕분이다.

## Decoder의 역할

디코더는 아래를 반복한다.

1. 이전에 생성한 토큰 시퀀스를 입력으로 변환
2. Cross Attention으로 인코더 정보 참조
3. 다음 토큰 1개를 생성
4. 이를 다시 입력으로 변환
5. End Token을 만날 때까지 반복

이 방식을 Auto-Regressive(자가회귀) 방식이라고 한다.

## 디코더는 왜 “미래를 못 보나?” (Causal Mask)

디코더의 Self-Attention에는 Causal Mask가 걸린다.

예를 들어 3번째 토큰 생성할 때 4번째, 5번째 토큰을 볼 수 없으며,
항상 “이전까지만” 참조 가능하다.

이는 GPT의 작동 방식과 완전히 동일하다.

## Seq2Seq ASR의 장점

- CTC보다 훨씬 강력한 “언어모델” 능력 포함
- 단어/문장 단위 토큰을 생성 (Whisper는 50k 토큰)
- 출력 시퀀스가 매우 짧아 속도·정확도 좋음
- 철자/문맥/언어적 규칙을 자연스럽게 반영

즉, Whisper 같은 모델이 CTC 모델보다 뛰어난 이유는
모델 구조가 언어를 이해하고 생성하는 데 최적화되어 있기 때문이다.

## Seq2Seq와 TTS(Text-to-Speech)

TTS 모델도 입력/출력 길이가 달라질 수 있다.
TTS에서의 Seq2Seq는 다음처럼 작동한다.

### Encoder: 텍스트 -> Hidden States

입력

```python
"こんにちは、私はホシです"
```

토크나이저로 자르고 임베딩하여 인코딩

```
[15 tokens] -> [15, hidden_dim]
```

### Decoder: Hidden States -> Mel Spectrogram

디코더는 아래를 반복한다.

- 이전 스펙트로그램 조각 + 인코더 정보
- 다음 스펙트로그램 조각 생성

즉, Whisper와 구조는 동일하지만 입출력이 반대이다.

## TTS가 어려운 이유

ASR은 “입력 오디오 -> 정답 텍스트”가 하나뿐이지만,
TTS는 “입력 텍스트 -> 가능한 음성”이 여러 개 있을 수 있다.

예시

- 억양/강세
- 감정/목소리 스타일
- 발음 속도
- 화자 특성

그래서 L1/MSE 같은 손실값만으로 품질을 평가하기 어렵다.
대신 사람이 듣고 평가하는 **MOS (Mean Opinion Score)**를 사용한다.

## Vocoder란?

TTS 모델의 출력은 스펙트로그램이다.
이걸 직접 들을 수는 없다.

그래서 스펙트로그램을 오디오 파형으로 변환하는
별도의 모델인 Vocoder가 필요하다.

대표

- MelGAN
- HiFi-GAN
- WaveGlow
- WaveRNN

TTS 프로젝트를 한다면
Seq2Seq 모델 + Vocoder를 조합해야 한다.

## Seq2Seq의 단점

- Auto-regressive 방식이라 속도가 느림
- 반복/루프/단어 건너뛰기 등 문제가 발생 가능
- 빔 검색(beam search)을 쓰면 더 느려짐

Whisper가 빠른 이유는
전체 구조는 seq2seq지만 내부 최적화가 매우 우수해서다.

## 결론

Seq2Seq는 CTC보다 훨씬 강력한 범용 아키텍처다.

- ASR에도 사용되고
- TTS에도 사용되고
- 번역/요약/대화에도 쓰이는
Transformer 세상의 핵심 구조이다.

Whisper와 SpeechT5가 강력한 이유는 다음과 같다.

- 스펙트로그램 기반 입력
- 강력한 언어 모델 기반 디코더
- Cross Attention
- Auto-regressive 생성 능력

이 모든 요소가 결합되어
오디오 -> 텍스트 / 텍스트 -> 오디오
모두 해결하는 강력한 구조이기다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn