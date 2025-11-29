---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 오디오 인식을 위한 사전학습 모델"
date:   2025-11-28 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 인식(ASR)을 위한 사전학습 모델


이 글에서는 허깅페이스의 pipeline()을 이용해 사전학습된 오디오 인식 모델을 사용하는 방법을 설명한다.

추가로 CTC 모델과 Seq2Seq 모델이 정확히 무엇이 다른지, Whisper가 왜 뛰어난지, 오디오 인식을 할 때 어떤 점을 알고 있어야 하는지를 정리합니다.

## 오디오 인식 모델의 두 계열

오디오 인식 모델은 크게 두 종류로 나뉜다.

| 종류                                              | 구조                                  | 특징                                                              |
| ----------------------------------------------- | ----------------------------------- | --------------------------------------------------------------- |
| **CTC (Connectionist Temporal Classification)** | Encoder-only + Linear head          | 빠름, 가벼움, 적은 데이터로 fine-tuning 가능. 하지만 **언어적 맥락이 부족**해 철자 오류가 많음  |
| **Seq2Seq (Encoder–Decoder)**                   | Encoder + Decoder + Cross-Attention | **언어 모델 기능을 가진 Decoder** 덕에 철자, 문맥, 문장 구조를 잘 맞춤. 단, 느리고 데이터 많이 필요 |


왜 이 구분이 중요할까?

1. CTC는 “소리 기반” 모델 -> 들리는 대로 적는 경향이 강함 (phonetic spelling).
2. Seq2Seq는 “소리+언어” 모델 -> 문맥, 철자, 문장 구조까지 이용.

따라서 정확한 음성 인식, 문장 단위의 자연스러운 출력, 구두점/대소문자까지 포함된 결과를 원한다면
CTC에서 Seq2Seq(Whisper)로 이동하는 구조라고 보면 된다.

## CTC 모델(Wav2Vec2) 탐구

### LibriSpeech 샘플 불러오기

```python
from datasets import load_dataset

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = dataset[2]
```

이 데이터는 텍스트와 오디오를 제공하며, 샘플을 들어보고 정답 텍스트도 확인할 수 있다.

### Wav2Vec2 모델로 음성 인식

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-100h")
pipe(sample["audio"].copy())
```

### 결과 비교

- 정답

CHRISTMAS, ROAST BEEF, SIMILES

- 예측(CTC)

CHRISTMAUS, ROSE BEEF, SIMALYIS

- 왜 이런 오류가 생길까?

CTC 모델 구조는 단순하다.
Audio -> Encoder -> Linear Layer -> Characters

- 오디오의 소리(phoneme)만 보고 글자를 출력한다.
- 앞뒤 문맥을 활용하지 못한다.
- 언어 모델 기능이 없다.

그래서 들리는 대로 쓰는 경향이 생긴다.

## Seq2Seq 모델로의 발전 - Whisper

Whisper는 Seq2Seq 구조로 되어 있으며, 이것이 CTC와의 핵심 차이다.

Audio
-> Encoder: 음성 특징 추출
-> Decoder: 언어 모델 (문맥, 철자, 단어 선택)
-> Output text


| 항목     | Whisper                   | Wav2Vec2/CTC      |
| ------ | ------------------------- | ----------------- |
| 학습 데이터 | **680,000시간 (대부분 라벨 있음)** | ~60,000시간 (라벨 없음) |
| 언어 지원  | **96개 언어**                | 주로 영어             |
| 출력 특성  | **대소문자, 구두점, 맞춤법 정확**     | 구두점 없음, 철자 오류 많음  |
| 긴 오디오  | **Chunking으로 안정적 처리**     | 길어지면 모델 입력을 넘어감   |
| 성능     | **State-of-the-art**      | 좋은 편이지만 한계 있음     |


Whisper가 음성 인식의 기본 모델로 자리 잡은 이유가 여기에 있다.

## Whisper 사용하기

```python
import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device
)
```

### 예측

```python
pipe(sample["audio"], max_new_tokens=256)
```

Whisper는 자동으로 구두점, 문장 구조를 생성한다.

CTC와 Seq2Seq 비교


| 모델       | 예측 결과                 |
| -------- | --------------------- |
| Wav2Vec2 | CHRISTMAUS, ROSE BEEF |
| Whisper  | Christmas, roast beef |


Whisper는 철자, 문맥, 단어 선택이 더 정확하다.

## Whisper로 다국어 음성 인식 & 번역

Whisper는 단순히 '다국어를 인식'하는 모델이 아니다.

- Speech-to-Text (Transcribe)
- Speech-to-English (Translate)

두 작업 모두 수행한다.

예시

- 스페인어 -> 스페인어(Transcribe)

```python
pipe(sample["audio"].copy(),
     max_new_tokens=256,
     generate_kwargs={"task": "transcribe"})
```

- 스페인어 -> 영어(Translate)

```python
pipe(sample["audio"],
     max_new_tokens=256,
     generate_kwargs={"task": "translate"})
```

Whisper가 단일 모델로
ASR + Translation을 동시에 할 수 있다는 점은 엄청난 장점이다.

## 긴 오디오(long-form) 처리 - Chunking

Whisper는 원래 30초 오디오만 처리하도록 설계되았다.
하지만 영화, 강의, 대본, 인터뷰처럼 오디오가 긴 경우에도 대처할 수 있다.

문제점

- 30초 이상은 잘려서 처리됨 (손실)
- Transformer 메모리 사용량은 길이²에 비례 -> OOM 발생

해결책: Chunking(분할처리)

- 긴 오디오를 30초 정도 크기로 나누고
- 약간의 overlap을 둬서 연결 부분을 자연스럽게 이어 붙인다
- 각 chunk는 독립적으로 처리 가능 -> 병렬 처리(batch) 가능

```python
pipe(
    long_audio,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=8,
    generate_kwargs={"task": "transcribe"}
)
```

Chunking은 Whisper 실전 사용에서 거의 필수이다.

## 타임스탬프 예측

자막 생성에 매우 유용하다.

```python
pipe(
    long_audio,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=8,
    return_timestamps=True,
)["chunks"]
```

출력 예시는 다음과 같다:

```python
[ { "timestamp": (0.0, 26.4), "text": "..." },
  { "timestamp": (26.4, 32.48), "text": "..." } ... ]
```

이를 이용하면
유튜브 자막(srt), 방송용 자막(vtt), 회의록 등을 쉽게 구현할 수 있다.

## Whisper가 실전에서 강력한 이유

Whisper는 일반 사용자 기준으로 완전한 실전형 모델이다.

장점

- 맞춤법, 문장구조, 구두점 자동 생성
- 96개 언어 지원
- 긴 오디오 강력 처리
- 번역 기능 탑재
- 잡음에 강함
- 도메인 일반화가 뛰어남 (audiobook, 회의, 대화, 유튜브 등)

단점

- 느림 (Decoder 때문에 step-by-step generation)
- 큰 모델은 VRAM 요구량 높음
- 저자원 언어/특정 억양/방언은 정확도 낮아짐

## Fine-tuning 필요성

Whisper는 매우 강력하지만 모든 언어, 모든 억양, 모든 상황을 완벽히 처리하지는 못한다.

- 저자원 언어
- 특정 억양
- 도메인 특화(의료, 법률, 방송, 콜센터 등)
- 방언, 지역적 표현

여기서 fine-tuning을 하면 큰 효과가 있다.

Whisper 논문에서도 “10시간 정도 추가 데이터로 성능이 100% 이상 개선”되는 경우가 있었다고 말한다.

다음 섹션에서 다룰 내용이 바로 이 fine-tuning이다.

## 요약

ASR 모델은 CTC vs Seq2Seq 두 가지 구조로 나뉜다.
Wav2Vec2와 같은 CTC 모델은 빠르고 효율적이지만 철자 오류가 많다.
Whisper는 680,000시간 라벨링 데이터로 학습된 SOTA Seq2Seq 모델이다.

Whisper는

- 다국어
- 구두점
- 번역
- 긴 오디오(chunking)
- 타임스탬프

를 모두 지원한다.

Whisper는 실제 서비스에 가장 적합한 모델 중 하나다.

부족한 분야(억양/저자원 언어)는 fine-tuning으로 개선 가능하다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn