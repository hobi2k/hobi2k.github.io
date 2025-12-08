---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - Meeting Transcription 정리"
date:   2025-12-08 00:10:22 +0900
categories: Huggingface_Audio
---

# Meeting Transcription 정리

현대 회의 기록 서비스(Otter.ai, Notion AI 등)는 단순한 음성->텍스트 변환이 아니라 “누가 언제 무엇을 말했는지”까지 포함한 풍부한 회의록을 제공한다.

이 문서에서는 다음 3단계를 조합하여
완전 자동 회의록 생성 시스템을 구축한다.

1. Speaker Diarization - 화자 구분 (“누가 말했는가?”)
2. Speech Transcription - Whisper로 텍스트 변환
3. Timestamp Alignment - Speechbox로 두 정보를 결합하여 화자별 발화 기록 생성

최종 결과물은 다음과 같다:

```python
SPEAKER_01 (0.0–15.5) …내용…
SPEAKER_00 (15.5–21.3) …내용…
```

## Speaker Diarization - Who Spoke When?

Speaker diarization은 이름 그대로

“이 오디오에서 누구(Speaker)가 언제(Segment) 말했는가?”
를 예측하는 작업이다.

Whisper에는 diarization 기능이 없기 때문에, pyannote.audio의 전문 diarization 모델을 사용한다.

### pyannote.audio 설치

```python
pip install --upgrade pyannote.audio
```

모델을 사용하려면 Hugging Face Hub에서
Terms of Use(사용 약관) 동의가 필요하다:

### Pretrained Diarization Pipeline 로드

```python
from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=True
)
```

이 pipeline은 내부적으로

- segmentation 모델로 음성 영역 탐지
- embedding 모델로 화자 특징 추출
- clustering 알고리즘으로 화자를 그룹화

를 수행한다.

### 테스트 예제 로드

```python
from datasets import load_dataset

concatenated_librispeech = load_dataset(
    "sanchit-gandhi/concatenated_librispeech", split="train", streaming=True
)
sample = next(iter(concatenated_librispeech))
```

오디오 확인

```python
from IPython.display import Audio
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

### 화자 구간 예측

pyannote는 입력을 (channels, length) 형태의 tensor로 받는다.

```python
import torch

input_tensor = torch.from_numpy(sample["audio"]["array"][None, :]).float()
outputs = diarization_pipeline(
    {"waveform": input_tensor, "sample_rate": sample["audio"]["sampling_rate"]}
)

outputs.for_json()["content"]
```

예상 결과

```python
[
 {'segment': {'start': 0.49, 'end': 14.52}, 'label': 'SPEAKER_01'},
 {'segment': {'start': 15.36, 'end': 21.37}, 'label': 'SPEAKER_00'}
]
```

- 첫 화자는 약 0~14.5초,
- 두 번째 화자는 약 15.4초 이후 발화한 것으로 인식한다.

## Whisper Transcription — What Was Said?

이제 동일한 오디오에 대해 Whisper로 텍스트 전사 + 세그먼트 단위 타임스탬프를 얻는다.

### Whisper 로드

```python
from transformers import pipeline

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)
```

return_timestamps=True 를 활성화해야 세그먼트 정보가 나온다.

```python
asr_pipeline(
    sample["audio"].copy(),
    generate_kwargs={"max_new_tokens": 256},
    return_timestamps=True
)
```

출력 예시

```python
"text": "... full transcript ...",
"chunks": [
  {"timestamp": (0.0, 3.56), "text": "..."},
  {"timestamp": (3.56, 7.84), "text": "..."},
  ...
  {"timestamp": (15.48, 19.44), "text": "..."},
]
```

특징

- Whisper는 본문을 여러 조각(chunk)으로 나눠 전사
- 각 chunk는 (start, end) 타임스탬프 포함
- 그러나 diarization timestamp와 정확히 일치하지 않음
-> 이후 alignment 과정 필요

## Aligning Diarization + Transcription

화자 구간 + Whisper 구간 -> 완전한 회의록

문제

- diarization 구간: 14.52초에서 화자 변경
- Whisper 구간: 13.88, 15.48초 등 미세하게 다름
- 따라서 두 정보를 맞춰서 어떤 텍스트가 어떤 화자의 발화인지 정해야 한다.

이를 자동으로 해결하는 패키지가 Speechbox.

### Speechbox 설치

```python
pip install git+https://github.com/huggingface/speechbox
```

ASR과 diarization pipeline을 결합한 ASRDiarizationPipeline 사용

```python
from speechbox import ASRDiarizationPipeline

pipeline = ASRDiarizationPipeline(
    asr_pipeline=asr_pipeline,
    diarization_pipeline=diarization_pipeline
)
```

사용

```python
pipeline(sample["audio"].copy())
```

예시 결과

```python
[
 {'speaker': 'SPEAKER_01',
  'text': ' ...first speaker text...',
  'timestamp': (0.0, 15.48)},
 {'speaker': 'SPEAKER_00',
  'text': ' ...second speaker text...',
  'timestamp': (15.48, 21.28)}
]
```

-> 두 정보를 완벽하게 병합된 형태로 반환한다.

## Formatting the Transcript

보기 좋게 포맷을 만들어보자.

```python
def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits),
                round(start_end_tuple[1], ndigits)))

def format_as_transcription(raw_segments):
    return "\n\n".join([
        chunk["speaker"] + " " +
        tuple_to_string(chunk["timestamp"]) +
        " " + chunk["text"]
        for chunk in raw_segments
    ])
```

사용

```python
outputs = pipeline(sample["audio"].copy())
format_as_transcription(outputs)
```

결과

```python
SPEAKER_01 (0.0, 15.5) The second and importance is as follows...
In France…

SPEAKER_00 (15.5, 21.3) He was in a favored state of mind…
```

## How It Works

이 파이프라인은 실제 회의록 서비스가 사용하는 구조와 매우 유사하다.

1. Diarization: Speaker Timeline 생성

- pyannote 모델이 화자 변경 지점을 감지
- 각 화자에 고유 ID 할당 (SPEAKER_00, SPEAKER_01 등)

2. Whisper: Segment Timeline 생성

- Whisper가 텍스트 chunk + timestamp 생성

3. Alignment: Timeline 결합

- 두 개의 시간선(time axis)을 정렬
- 가장 가까운 timestamp 기준으로 chunk를 speaker에게 연결

4. Final Transcript

- 화자 + 발화 시간 + 내용 이 포함된 회의록 출력

이는 실제 SaaS 서비스가 사용하는 구조의 “경량 버전”이지만
이미 높은 수준의 자동 회의 요약 파이프라인의 핵심 구성요소를 모두 갖추고 있다.

## 더 높은 품질을 위한 확장 아이디어

1. Whisper Large-V3 사용

- 정확도 향상 (단, GPU 필요)

2. pyannote speaker embedding fine-tuning

- 개인 사용자 화자를 지속적으로 학습하여 정확도 증가

3. 회의 요약 (Abstractive Summarization) 결합

- LLM을 통해 회의 후 요약 자동 생성

4. 음성 activity 기반 자동 stop

- 현재 예제는 전체 길이를 그대로 사용하지만, VAD 모델을 추가하면 “듣는 타이밍”을 더 자연스럽게 조절 가능

참고자료
Huggingface, Audio Course, https://huggingface.co/learn