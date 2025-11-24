---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 파이프라인을 이용한 자동 음성 인식"
date:   2025-11-24 00:10:22 +0900
categories: Huggingface_Audio
---

# 파이프라인을 이용한 자동 음성 인식

자동 음성 인식(ASR, Automatic Speech Recognition)은 오디오 신호를 텍스트로 변환하는 작업이다.
예시는 다음과 같이 매우 다양하다.

- 유튜브/교육 영상 자막 생성
- 콜센터 상담 텍스트화
- Siri, Alexa 같은 음성 비서
- 회의록 자동 작성
- 오디오 기반 검색 시스템

이 글에서는 Transformers의 pipeline("automatic-speech-recognition")을 사용하여
단 몇 줄만으로 음성을 텍스트로 바꾸는 방법을 알아본다.

## 데이터 준비

이전 단원에서 사용한 MINDS-14 데이터셋을 그대로 사용한다.
오디오 분류와 마찬가지로, ASR 모델도 16kHz 샘플링 속도를 요구하므로 리샘플링이 필요하다.

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

## ASR 파이프라인 불러오기

Transformers는 자동 음성 인식을 위한 automatic-speech-recognition 파이프라인을 제공한다.

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition")
```

기본적으로 영어에 최적화된 Whisper 기반의 모델을 사용한다(환경에 따라 달라질 수 있음).

## 오디오 -> 텍스트 변환

데이터셋에서 하나를 선택해보자.

```python
example = minds[0]
asr(example["audio"]["array"])
```

출력

```python
{"text": "I WOULD LIKE TO PAY MY ELECTRICITY BILL USING MY COD CAN YOU PLEASE ASSIST"}
```

## 실제 정답과 비교하기

```python
example["english_transcription"]
```

출력

```python
"I would like to pay my electricity bill using my card can you please assist"
```

결과를 보면 거의 완벽히 맞췄다.
다만 "card"를 "cod" 로 오인식했는데, 이는 호주식 억양(Australian accent) 특성 때문에 충분히 발생할 수 있는 사례이다.

## 다른 언어에도 적용해 보기

기본 파이프라인은 영어 모델을 사용하므로
독일어, 프랑스어, 한국어 등 다른 언어 처리가 필요하면
허깅페이스 허브에서 원하는 언어의 ASR 모델을 선택해서 사용해야 한다.

예시: MINDS-14의 독일어(de-DE) 부분을 사용해보자.

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

예제 문장을 확인한다.

```python
example = minds[0]
example["transcription"]
```

출력

```python
"ich möchte gerne Geld auf mein Konto einzahlen"
```

이번에는 독일어 ASR 모델을 불러오자.

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="maxidl/wav2vec2-large-xlsr-german"
)
asr(example["audio"]["array"])
```

출력

```python
{"text": "ich möchte gerne geld auf mein konto einzallen"}
```

완벽하진 않지만, 발음 특성이나 녹음 품질을 고려하면 충분히 좋은 결과다.

## 왜 파이프라인으로 시작해야 할까?

오디오 ASR 시스템을 개발할 때 pipeline()으로 시작하는 것은 큰 장점이 있다.

1. 사전학습된 모델을 바로 활용

이미 해당 태스크(task)와 언어에 맞춰 파인튜닝된 모델이 있을 가능성이 높다.
시간을 크게 절약할 수 있다.

2. 전처리 및 후처리 자동 처리

- 리샘플링
- 정규화
- 로그-멜 스펙트로그램 생성
- 디코딩(beam search 등)

이런 복잡한 과정을 파이프라인이 모두 처리한다.

3. 빠르게 baseline 확보 가능

모델을 처음부터 파인튜닝하기 전에
파이프라인으로 baseline 성능을 확인하면 큰 도움이 된다.

4. 다른 사람도 쉽게 사용할 수 있는 모델 제공

만약 커스텀 모델을 Hugging Face Hub에 업로드하면
다른 사람들은 아래 한 줄로 해당 모델을 바로 사용할 수 있다.

pipeline("automatic-speech-recognition", model="your-username/your-model")

## 추가 팁: ASR 모델 선택 기준

언어마다 적합한 모델이 다르다.
아래 기준을 참고하면 좋은 모델을 빠르게 고를 수 있다.


| 언어  | 추천 모델                      | 특징          |
| --- | -------------------------- | ----------- |
| 영어  | Whisper, wav2vec2          | 가장 높은 성능    |
| 한국어 | whisper-large-v3, Kosp2e   | 억양/받침 처리 우수 |
| 일본어 | Whisper + CTC 기반 모델        | 억양·피치 정보 중요 |
| 독일어 | wav2vec2-large-xlsr-german | 발음 다양성 대응   |
| 중국어 | Whisper, Chinese Wav2vec2  | 성조 처리 가능    |


참고자료
Huggingface, Audio Course, https://huggingface.co/learn