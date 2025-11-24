---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 파이프라인을 이용한 오디오 분류"
date:   2025-11-24 00:10:22 +0900
categories: Huggingface_Audio
---

# 파이프라인을 이용한 오디오 분류

오디오 분류(Audio Classification)는 녹음된 오디오를 분석하여 해당 오디오가 어떤 카테고리에 속하는지 레이블을 부여하는 작업이다.
예를 들어 다음과 같은 종류를 예측할 수 있다.

- 음악, 음성, 노이즈
- 동물 소리(새소리, 개 짖는 소리 등)
- 기계 소리(엔진, 경보음 등)
- 사용자 의도(Intent) 분류(은행 상담, 주문, 문의 등)

이 글에서는 Transformers의 고수준 API인 pipeline()만 사용해
사전학습된 오디오 분류 모델을 몇 줄 코드로 활용하는 법을 알아본다.

## 데이터 준비: MINDS-14 오디오 데이터셋

MINDS-14는 사람들이 인터넷뱅킹 시스템에 전화로 문의한 내용을 여러 언어 및 방언으로 녹음한 데이터셋이다.
각 오디오에는 intent_class(사용자의 의도)가 붙어 있어 분류 실습에 적합하다.

먼저 데이터와 샘플링 속도를 모델에 맞춰 조정한다.

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

대부분의 음성 분류 모델은 16kHz 입력을 기대하므로
데이터셋도 이에 맞춰 리샘플링한다.

## 오디오 분류 파이프라인 불러오기

Transformers는 오디오 분류에 특화된 파이프라인 "audio-classification"을 제공한다.
그래서 사전학습 + 파인튜닝된 모델을 불러와 바로 사용할 수 있다.

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
```

이 모델은 MINDS-14에 맞춰 이미 파인튜닝되어 있어
바로 의도 분류(intent classification)에 사용할 수 있다.

## 파이프라인에 오디오 넘기기

데이터셋에서 하나의 예시를 가져온다.

```python
example = minds[0]
```

원시 오디오 배열은 다음 위치에 있다.

```python
example["audio"]["array"]
```

이 배열을 그대로 파이프라인에 넣을 수 있다.

```python
classifier(example["audio"]["array"])
```

출력

```python
[
    {"score": 0.96315, "label": "pay_bill"},
    {"score": 0.02819, "label": "freeze"},
    {"score": 0.00327, "label": "card_issues"},
    {"score": 0.00194, "label": "abroad"},
    {"score": 0.00083, "label": "high_value_payment"},
]
```

모델은 해당 발화가 pay_bill일 가능성이 매우 높다고 판단했다.

## 실제 정답 비교

데이터셋의 레이블은 숫자지만
int2str() 메서드를 사용하면 문자열로 바꿀 수 있다.

```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
```

출력

```python
"pay_bill"
```

예측값과 실제 레이블이 일치한다.

## 추가팀

1. 파이프라인은 자동 전처리 + 모델 추론 + 후처리를 수행한다

- 리샘플링
- 정규화
- 패딩
- STFT 또는 feature extraction

이 모든 과정을 자동으로 처리한다.

2. 파이프라인은 NumPy 배열만 넘기면 된다

오디오 파일을 직접 읽어 처리할 필요 없이
데이터셋의 audio["array"]를 그대로 넣어도 된다.

3. 클래스가 원하는 문제와 맞지 않을 경우 "보정(calibration)"이 필요

사전학습된 오디오 분류 모델의 클래스가
내가 원하는 레이블과 다를 때는 파인튜닝(fine-tuning)을 하거나
기존 모델의 마지막 레이어를 교체해 다시 학습해야 한다.

4) pipeline()은 편리하지만 속도/대규모 작업에는 적합하지 않을 수 있다

- 내부에서 CPU 기반 feature extraction
- 반복 호출 시 느려질 수 있음

대규모 inference에서는 model + processor 조합을 직접 사용하는 것이 더 적합하다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn