---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 3일차"
date:   2025-11-23 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 데이터셋 전처리하기


오디오 데이터셋을 불러오는 것만으로는 부족하다.
모델을 학습시키거나 추론(inference)에 사용하기 위해서는 반드시 전처리(preprocessing) 과정이 필요하다.

일반적인 오디오 전처리 단계는 다음과 같다.

- 오디오 리샘플링(Resampling)
- 불필요한 데이터 필터링
- 모델 입력 형식에 맞춘 변환(Feature Extraction)

아래는 이 과정을 모두 실제 코드와 함께 설명한다.

## 오디오 데이터 리샘플링하기

load_dataset()으로 불러온 오디오는 원래 데이터셋이 가진 샘플링 속도를 그대로 따른다.
하지만 이는 학습하려는 모델의 샘플링 속도와 다를 수 있다.

예시:
- MINDS-14 데이터셋: 8 kHz
- 대부분의 사전학습된 음성 모델(Whisper, Wav2Vec2 등): 16 kHz

샘플링 속도가 맞지 않으면 모델 입력 차원 자체가 달라져 제대로 학습할 수 없다.
따라서 리샘플링(resampling) 이 필요하다.

### cast_column()을 사용한 리샘플링

```python
from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

이 코드는 오디오 파일을 직접 수정하지 않는다.
대신 불러올 때 즉시(on-the-fly) 16 kHz로 리샘플링한다.

- 리샘플링 적용 확인
```python
minds[0]

{
    "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
    "audio": {
        "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
        "array": array(
            [
                2.0634243e-05,
                1.9437837e-04,
                2.2419340e-04,
                ...,
                9.3852862e-04,
                1.1302452e-03,
                7.1531429e-04,
            ],
            dtype=float32,
        ),
        "sampling_rate": 16000,
    },
    "transcription": "I would like to pay my electricity bill using my card can you please assist",
    "intent_class": 13,
}
```

출력에서 sampling_rate: 16000으로 바뀐 것을 확인할 수 있다.

또한 배열의 길이가 기존보다 두 배 길어진 것도 확인할 수 있는데,
8kHz -> 16kHz 업샘플링이므로 총 샘플 수가 2배가 되기 때문이다.

### 팁: 리샘플링의 원리와 주의사항

리샘플링은 단순히 "샘플을 더 찍어 넣기"가 아니다.

- 업샘플링(up-sampling)
기존 샘플 사이 값을 보간(interpolation)

- 다운샘플링(down-sampling)
나이퀴스트 한계를 초과하는 주파수 성분을 제거한 뒤 샘플링
그렇지 않으면 aliasing(왜곡) 발생

따라서 리샘플링은 직접 구현하기보다
librosa, torchaudio, datasets.Audio 같은 검증된 라이브러리를 사용하는 것이 좋다.

## 데이터셋 필터링하기

데이터셋을 학습에 사용하기 전, 여러 기준으로 필터링해야 할 때가 많다.

예시:
- 너무 긴 오디오(메모리 초과 방지)
- 너무 짧은 오디오
- 노이즈가 심한 파일
- 텍스트 라벨이 비어있는 파일

아래는 20초 이상인 오디오를 제거하는 예시다.

1. 오디오 길이 컬럼 추가하기

```python
# use librosa to get example's duration from the audio file
new_column = [librosa.get_duration(filename=x) for x in minds["path"]]
minds = minds.add_column("duration", new_column)
```

2. 필터 함수 정의

```python
MAX_DURATION_IN_SECONDS = 20.0

def is_audio_length_in_range(duration):
    return duration < MAX_DURATION_IN_SECONDS
```

3. 데이터셋 필터링

```python
minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])
```

4. 임시 컬럼 삭제

```python
minds = minds.remove_columns(["duration"])
```

필터링 후 데이터 개수는 줄어든다.
(위의 예시: 654 -> 624)

## 오디오 데이터 모델 입력 형식으로 전처리하기

오디오 데이터셋 전처리에서 가장 중요한 것은
원시 파형(waveform)을 모델이 요구하는 입력 형태로 변환하는 것이다.

이 변환 과정은 모델마다 다르다:

- Whisper: 30초 패딩 + log-mel spectrogram
- Wav2Vec2: raw audio 그대로 사용
- Hubert: raw audio
- SpeechT5: mel-spectrogram 등

다행히 transformers에서는 모델별 feature extractor를 제공하므로
원시 오디오를 자동으로 올바른 입력 형식으로 변환할 수 있다.

## Whisper의 feature extractor 예시

Whisper는 다음 두 작업을 수행한다.

1. 길이 맞추기 (padding/truncation)

- Whisper는 항상 30초 길이의 입력을 받는다.
- 더 짧으면 0-padding
- 더 길면 30초로 잘라냄
- 모든 입력의 길이가 동일하므로 attention mask 불필요

이는 Whisper가 자체적으로 “무시할 부분”을 추론하도록 학습되었기 때문이다.

2. log-mel spectrogram 생성

- raw audio -> mel-spectrogram
- mel-spectrogram -> log scale 변환
- shape: (80, time_frames) 
* Whisper 모델은 n_mels(mel 필터)이 80개다.
* 각 필터는 특정 주파수 범위를 커버하고 각 필터마다 한 값을 가진다.

이 변환은 Whisper 논문에서도 핵심적 구조이다.

### Whisper feature extractor 불러오기

```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
```

### 전처리 함수 작성

```python
def prepare_dataset(example):
    audio = example["audio"]
    features = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        padding=True,
    )
    return features
```

### 데이터셋 전체에 적용

```python
minds = minds.map(prepare_dataset)
```

이제 데이터셋에는 input_features 컬럼이 추가된다.

## 전처리된 Whisper 입력 시각화

```
import numpy as np

example = minds[0]
input_features = example["input_features"]

plt.figure().set_figwidth(12)
librosa.display.specshow(
    np.asarray(input_features[0]),
    x_axis="time",
    y_axis="mel",
    sr=feature_extractor.sampling_rate,
    hop_length=feature_extractor.hop_length,
)
plt.colorbar()
```

이 시각화는 Whisper가 실제로 입력받는 log-mel spectrogram이다.

## Processor를 사용해 feature extractor + tokenizer 동시 로딩

Whisper처럼 오디오와 텍스트를 모두 사용하는 모델은 두 가지 전처리가 필요하다.

- audio -> feature extractor
- text -> tokenizer

이 두 요소를 하나로 묶은 것이 AutoProcessor이다.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small")
```

## 추가 팁

1. 모델 입력 형식은 모델마다 다르다

Whisper는 mel-spectrogram을 사용하지만
Wav2Vec2·HuBERT·wavLM 등은 raw audio를 그대로 입력으로 사용한다.

2. 전처리가 잘못되면 학습 품질이 급격히 떨어진다

- 샘플링 속도 불일치
- 잘못된 spectrogram 파라미터
- padding 오류

이런 문제는 학습 자체가 성립하지 않게 만들 수 있다.

3. map()은 병렬 처리 가능

Datasets의 map은 매우 빠르며,
멀티프로세싱도 쉽게 설정할 수 있다.

4. 커스텀 데이터셋에도 동일한 방식 적용 가능

중요한 것은 prepare_dataset() 함수만 잘 만들면 무엇이든 전처리 가능하다는 점이다.

## 마무리

이 글에서는 오디오 데이터 전처리의 핵심 과정을 모두 다루었다.

- 리샘플링
- 길이 기준 필터링
- feature extractor 기반 mel-spectrogram 변환
- Whisper 입력 포맷 이해
- Processor 사용법

오디오 모델링을 제대로 하려면 전처리 단계가 가장 중요하다.
datasets와 transformers를 함께 사용하면 이 전체 파이프라인을 매우 효율적으로 구축할 수 있다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn