---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 2일차"
date:   2025-11-22 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 데이터셋 불러오기 및 탐색하기


이 글에서는 허깅페이스 Datasets 라이브러리를 사용하여 오디오 데이터셋을 불러오고 탐색하는 방법을 알아본다.

Datasets은 텍스트, 이미지, 오디오 등 다양한 데이터셋을 손쉽게 다운로드하고 준비할 수 있는 오픈소스 라이브러리이며, 허깅페이스 허브(Hugging Face Hub)에 있는 데이터셋을 파이썬 코드 한 줄로 불러올 수 있다는 장점이 있다.

오디오 데이터셋을 다룰 때 필요한 기능도 기본적으로 제공하므로 음성 인식, 오디오 분류, TTS 등 오디오 기반 머신러닝 작업을 하는 사람에게 매우 유용하다.

## Datasets 설치

오디오 지원 기능이 포함된 Datasets를 설치한다.

```python
pip install datasets[audio]
```

## load_dataset()으로 오디오 데이터 불러오기

load_dataset()은 데이터셋 식별자만 넘기면 자동으로 다운로드하고 사용할 준비까지 끝내준다.

예제로 사용할 데이터셋은 MINDS-14이다.
여러 언어·방언으로 사람들이 인터넷뱅킹에 관해 질문한 내용을 녹음한 음성 데이터셋이다.

아래는 MINDS-14의 호주 영어(en-AU) 버전의 train split만 불러오는 코드다.

```python
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds

# 출력
Dataset(
    {
        features: [
            "path",
            "audio",
            "transcription",
            "english_transcription",
            "intent_class",
            "lang_id",
        ],
        num_rows: 654,
    }
)
```

데이터셋에는 총 654개의 음성 파일이 있으며, 각 파일에 다음 정보가 함께 제공된다.

- transcription: 원문 자막
- english_transcription: 영어 번역
- intent_class: 사용자 발화 목표(intent) 레이블
- audio: 원본 오디오 데이터

## 데이터 예시 살펴보기

첫 번째 예제를 확인해 보자.

```python
example = minds[0]
example

# 출력
{
    "path": ".../response_4.wav",
    "audio": {
        "path": ".../response_4.wav",
        "array": array([...], dtype=float32),
        "sampling_rate": 8000,
    },
    "transcription": "I would like to pay my electricity bill using my card can you please assist",
    "english_transcription": "I would like to pay my electricity bill using my card can you please assist",
    "intent_class": 13,
    "lang_id": 2,
}
```

오디오 feature 내부 항목은 다음과 같습니다.

- path: 오디오 파일 경로
- array: 디코딩된 waveform (부동소수점 float32)
- sampling_rate: 샘플링 속도 (여기서는 8kHz)

intent_class 숫자는 int2str 메서드를 사용하여 실제 라벨로 변환할 수 있다.

```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])

# 출력
"pay_bill"
```

## 필요 없는 컬럼 제거하기

데이터 전처리 시, 모델 입력에 필요없는 컬럼은 제거하는 것이 좋다.

제거하면 좋을 컬럼은 다음과 같다.

- lang_id: 같은 방언만 사용하므로 의미 없음
- english_transcription: 이 split에서는 transcription과 동일하므로 의미 없음

컬럼 삭제 코드는 다음과 같이 할 수 있다.

```python
columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
minds

# 출력
Dataset({features: ["path", "audio", "transcription", "intent_class"], num_rows: 654})
```

## Gradio로 오디오 샘플 듣기

오디오를 직접 들어보면 데이터셋의 품질과 특징을 훨씬 빠르게 파악할 수 있다.
아래 코드는 랜덤 오디오 4개를 재생하는 Gradio 앱을 만든다.

```python
import gradio as gr

def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label(example["intent_class"])

with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)
```

## 파형 시각화하기

오디오의 파형(waveform)을 시각화하면 다음을 확인할 수 있다.

- 발화 길이
- 무음 비율
- 노이즈 존재 여부
- 클리핑(clipping) 여부
- 전처리 상태

```python
import librosa
import matplotlib.pyplot as plt
import librosa.display

array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)
```

## 다른 언어 버전도 탐색해 보기

MINDS-14는 다양한 언어와 방언을 제공한다.
여러 언어 버전을 로드하여 비교해 보면 데이터 구조를 더 잘 이해할 수 있다.

언어 목록: [링크](https://huggingface.co/datasets/PolyAI/minds14)

## 추가 팁

1. sampling_rate는 모델 학습에 결정적인 요소

ASR/TTS 모델은 입력 파형(waveform) 또는 스펙트로그램의 크기에 민감하므로,
데이터셋 전체의 sampling_rate가 통일되어 있는지 반드시 확인해야 한다.

2. audio["array"]는 항상 float32

ML 모델은 float32 기반으로 동작하므로
일반적인 16bit PCM 오디오는 로드 시 float32로 변환된다.

3. remove_columns()는 필수적인 전처리

불필요한 컬럼이 많을수록 데이터 콜레이터(DataCollator)나 모델 입력 과정이 복잡해지므로
필요한 컬럼만 남기는 것이 가장 안정적인 파이프라인을 만든다.

4. shuffle()을 호출하면 확률적으로 랜덤 샘플링

Datasets의 shuffle()은 전체 데이터를 섞은 새 버전을 반환한다.
즉, 매번 새로운 shuffle 객체가 생성된다.

5. Gradio는 오디오 확인용으로 최적

데이터 품질을 빠르게 점검할 때 매우 강력하다.
특히 ASR/TTS 데이터셋의 유용성을 초기에 판단하는 데 도움이 된다.

## 마무리

이 글에서는 Datasets 라이브러리를 사용해 오디오 데이터셋을 불러오고,
구성 요소를 탐색하며, 불필요한 컬럼을 제거하고, 오디오를 직접 듣고 시각화하는 방법까지 다루었다.

오디오 기반 모델링(ASR, 오디오 분류, 음성 질의 응답 등)을 진행하기 위해서는
데이터셋 구조를 정확하게 이해하는 것이 출발점이며,
Datasets 라이브러리는 이 작업을 간단하고 효율적으로 만들어준다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn