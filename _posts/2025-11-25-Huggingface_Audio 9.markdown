---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - CTC 아키텍처 정리"
date:   2025-11-25 00:10:22 +0900
categories: Huggingface_Audio
---

# CTC 아키텍처 정리

연결주의 시간 분류(Connectionist Temporal Classification)는 오디오 트랜스포머에서 정렬(Alignment) 문제를 푸는 핵심 기법이다.

## CTC 아키텍처란 무엇인가?

**CTC(Connectionist Temporal Classification)**는 자동 음성 인식(ASR) 모델에서 정렬 정보를 몰라도 텍스트를 예측할 수 있게 하는 알고리즘이다.

CTC는 특히 인코더만 사용하는 트랜스포머 모델에 사용된다. 

예시
- Wav2Vec2
- HuBERT
- M-CTC-T
- (그 외 여러 ASR 모델)

이 모델들은 공통적으로 다음 구조를 가진다.

오디오 -> CNN Feature Encoder -> Transformer Encoder -> CTC Linear Head -> 문자 로그 확률

즉, 이 모델은 Transformer Encoder만 있어서 입력(오디오)을 벡터 시퀀스로 변환하고,
그 위에 얹힌 **선형 레이어(CTC Head)**가 각 타임스텝마다 문자(label)를 예측하는 구조다.

## 왜 ASR에는 CTC가 필요한가?

텍스트는 시간에 따라 흐르지만, 오디오와 텍스트 사이의 정확한 대응 관계(alignment)를 모르기 때문이다.

- 오디오에는 말의 길이와 속도가 포함된다.
- 텍스트에는 “정답 문장”만 있다.
- 데이터셋에는 대부분 타이밍 정보가 없다(“여기서 A라는 소리가 난다” 같은 정보 없음).

그래서 모델은 이렇게 묻는다.

“오디오의 이 부분이 텍스트의 어떤 문자에 대응되는지 모르는데, 그럼 어떻게 학습하지?”

CTC는 이 문제를 아주 우아하게 해결한다.

## 오디오에서 Transformer Encoder까지의 과정

### Wav2Vec2 예시

1. 1초짜리 오디오
2. CNN Feature Encoder

- 오디오를 약 20ms 단위로 다운샘플링
- 약 50개의 hidden state가 생성됨

3. Transformer Encoder

- 입력: (sequence_length=50, hidden_dim=768)
- 출력도 동일 길이의 시퀀스

원래 허깅페이스 문서에는 (768, 50)이라고 잘못 표기되어 있다.
실제 텐서는 (50, 768) 형태이다.
(PyTorch의 Transformer는 (batch, seq, dim)이 기본)

각 타임스텝은 약 25ms 분량의 음성 정보를 담고 있다.

## 문자 기반 분류(head)가 필요한 이유

Transformer output 768차원을 **어휘(문자 집합)**에 맵핑하기 위해 선형층을 사용한다.

예시
- 알파벳 26자 + 특수문자 + 공백 + 단어 구분용 문자
-> 총 vocab size ≈ 30~40개

따라서 최종 로짓(logits) 모양은 다음과 같다.

```python
(50, vocab_size)
```

즉, 모델은 오디오 1초에서 50개의 문자 예측을 한다.
문제는 여기서부터 시작된다.

## Alignment 문제 = 중복 문제

예측 시퀀스가 이런 식으로 나온다고 해보자.

BRIIONSAWWSOMEETHINGCLOSETOPANICON...

왜 이렇게 될까?

- 같은 소리가 여러 프레임에 걸쳐 존재힌디.
- 모델은 매 프레임마다 문자를 하나 출력해야 한다.
- 단어의 “정확한 시작/끝”을 모르기 때문에 과도하게 반복된다.
- 겹치는 오디오 구간도 존재한다.

이 문제를 해결하는 것이 CTC의 본질이다.

## CTC 알고리즘의 핵심: Blank Token

CTC는 특수 공백(blank) 토큰을 도입하여 문제를 푼다.
이 토큰을 _로 표시하자.

CTC 출력 예시시는 다음과 같다.

B_R_II_O_N_||_S_AWW_|||||_S_OMEE_TH_ING...

여기서 하는 일은 매우 간단하지만 강력하다.

### CTC 디코딩 규칙

- 연속된 같은 문자 반복 -> 1개로 줄인다
- blank(_) 토큰은 제거한다

예시는 다음과 같다.

- 원본 예측

_ER_RRR_ORR

- 중복 축소

_ER_R_OR

- blank 제거

ERROR

이 간단한 규칙으로 정렬 문제를 해결한다.

## CTC가 잘 작동하는 이유

음성은 본질적으로 단조로운(monotonic) 구조다.

- 텍스트 순서는 항상 시간 순서와 동일
- 하지만 정확히 언제 어떤 글자가 나오는지는 모름

CTC는 “가능한 모든 alignment”를 고려해 확률을 계산한다.

즉, alignment를 직접 맞추지 않아도 모델이 알아서 문자 시퀀스 확률을 최대로 만들도록 학습된다.

## CTC의 한계

전체 단어가 아니라 문자 단위만 보기 때문에
맞춤법은 약할 수 있다.

예를 들어, "there"와 "their" 같은 발음 동일 단어는 헷갈릴 수 있다.

이를 보완하기 위해 보통 **언어 모델(LM)**을 디코딩에 추가한다.

- KenLM, n-gram LM
- Transformer LM (Shallow Fusion 등)

Whisper가 높은 품질을 가지는 이유도
Encoder-Decoder 구조 + 강한 LM 효과가 결합되어 있기 때문이다.

## Wav2Vec2, HuBERT, M-CTC-T 차이

### 공통점

- 모두 Transformer Encoder
- 모두 CTC Head 사용
- 모두 오디오 분류/ASR에서 거의 동일한 구조

### 차이점

1. Wav2Vec2

- 입력: raw waveform
- pretraining: masked reconstruction(음성 unit 예측)
- 가장 널리 사용됨

2. HuBERT

- 입력: raw waveform
- pretraining 방식이 다름
(군집 기반 pseudo-label → BERT 스타일 예측)

3. M-CTC-T

- 입력: Mel Spectrogram
- 다국어 지원 목적
- vocab이 큼(한자 포함)

즉, 아키텍처는 거의 같고, 목표/데이터/사전학습 방식이 다를 뿐이다.

## CTC의 중요성

오디오 AI(ASR, TTS) 공부할 때 CTC는 크게 도움이 된다.

- Whisper, Wav2Vec2 구조를 이해하는 핵심
- Alignment-free 학습 개념 이해
- Tacotron 같은 TTS 모델과 비교 이해 가능
- TTS/ASR 파이프라인 설계 시
“character-level vs phoneme-level” 선택 기준을 이해하게 됨

또한, CLIP이나 ViT 같은 다른 Transformer 모델도 결국
"embedding -> projection -> classification" 구조라
CTC 헤드 개념이 자연스럽게 이어진다.

## 전체 시각화

Audio (waveform)
      ↓
CNN Feature Extractor (50 frames per second)
      ↓   
Transformer Encoder
      ↓
CTC Linear Head (vocab size: 30)
      ↓
(50, vocab) logits
      ↓
CTC Decoding (merge repeats + remove blank)
      ↓
Final Text

## 정리

CTC는 ASR에서 Alignment 문제를 해결하는 가장 단순하고 강력한 알고리즘이다.

- 프레임마다 문자 예측 -> 중복 불가피
- CTC blank와 merge 규칙으로 정렬을 해결
- Transformer Encoder + Linear Projection -> ASR 모델의 전형적 구조
- 사전학습 방식(Wav2Vec2/HuBERT)만 다를 뿐 구조는 유사함
- 필요시 외부 언어 모델로 맞춤법 보완

Whisper 같은 encoder-decoder 모델을 이해할 때도
CTC 기반 encoder-only 모델 이해가 큰 기반이 된다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn