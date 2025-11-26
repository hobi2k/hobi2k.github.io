---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 오디오 분류를 위한 사전학습 모델"
date:   2025-11-26 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 분류를 위한 사전학습 모델과 데이터셋 정리

오디오 분류(audio classification)는 오디오 입력 전체를 보고 단일 클래스 라벨을 예측하는 작업이다. 자연어 처리(NLP)의 문장 분류와 유사한 구조를 사용하지만, 입력이 텍스트가 아닌 오디오 파형(audio array)이라는 점만 다르다.

Hugging Face Hub에는 500개 이상의 오디오 분류용 사전학습 모델이 올라와 있으며, pipeline() API를 이용하면 어떤 모델을 사용하더라도 거의 같은 인터페이스로 처리할 수 있다. 즉, 모델을 바꿔도 코드 구조는 동일하다.

이 글에서는 다음을 정리한다.

- 오디오 분류에 적합한 Transformer 구조(Encoder-only 중심)
- 대표 데이터셋: MINDS-14, Speech Commands, FLEURS
- Keyword Spotting(KWS)
- Language Identification(LID)
- Zero-shot Audio Classification(CLAP 기반)
- Hugging Face Hub에서 적절한 모델 찾는 방법

## 오디오 분류에 적합한 Transformer 구조

오디오 분류는 입력 시퀀스 -> 단일 라벨이라는 특징상 Encoder-only Transformer 구조가 가장 잘 맞는다.

### Encoder-only 구조가 적합한 이유

- 입력 오디오 시퀀스 -> hidden state(프레임 단위 표현)
- hidden states 전체를 평균(mean pooling)
- 하나의 벡터를 Linear Layer에 넣어 클래스 예측

즉, 전체 시퀀스를 요약해(label-level representation) 하나의 라벨을 뽑기 때문에 Encoder-only 구조가 효율적이다.

### Decoder-only(GPT 계열) 모델이 적합하지 않은 이유

Decoder-only 모델은 출력을 시퀀스로 생성해야 한다는 가정을 갖고 있다. 즉, 토큰을 하나씩 자가회귀(autoregressive)하게 생성하는 구조다.

오디오 분류 태스크에는 다음의 특징이 있다.

- 단 하나의 라벨만 필요
- 시퀀스 출력 불필요
- latency(추론 속도)가 매우 중요

따라서 Decoder-only는 구조적 복잡성과 속도 측면에서 비효율적이다.

### Encoder-Decoder(T5, Whisper) 모델이 잘 쓰이지 않는 이유

Encoder-Decoder Transformer는 입력을 인코딩하고 디코더가 출력을 생성하는 구조로, 역시 “시퀀스 생성” 중심이다. 오디오 분류처럼 단일 라벨만 예측하면 되는 경우에는 과한 구조이다.

## Transformers 설치

일부 오디오 기능은 PyPI 버전에 아직 반영되지 않았기 때문에, 최신 기능이 필요한 경우 GitHub main 브랜치를 설치한다.

```python
pip install git+https://github.com/huggingface/transformers
```

## Keyword Spotting (KWS)

Keyword Spotting은 음성 내 특정 단어(명령어, 의도, wake word)를 탐지하는 작업이다.

### MINDS-14

이전에 다룬 바 있는 은행 상담(intent) 분류 데이터셋.
다양한 언어로 고객의 문의 의도를 녹음한 음성 데이터가 포함되어 있다.

```python
from datasets import load_dataset
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
```

추천 모델

- anton-l/xtreme_s_xlsr_300m_minds14

XLS-R 기반 모델이며 MINDS-14 전 언어 기준 약 90% 정확도를 보인다.

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
classifier(minds[0]["audio"])
```

콜센터 보이스봇, 자동 라우팅 시스템 등에서 실무 활용도가 매우 높은 분야다.

### Speech Commands

구글이 만든 기초 명령어 인식 데이터셋.
15개의 짧은 단어(“up”, “down”, “stop” 등)를 포함한다.

스마트폰의 “Hey Siri”, “Hey Google” 기능에 쓰이는 wake word detector와 구조적으로 같다.

```python
speech_commands = load_dataset(
    "speech_commands", "v0.02", split="validation", streaming=True
)
sample = next(iter(speech_commands))
```

추천 모델

- MIT/ast-finetuned-speech-commands-v2

```python
classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-speech-commands-v2"
)
classifier(sample["audio"].copy())
```

오디오가 어떤 단어를 말하고 있는지 가장 높은 확률로 예측한다.

## Language Identification (LID)

LID는 오디오가 어떤 언어인지 판별하는 작업이다.
음성 인식 시스템에서는 “어떤 ASR 모델을 적용할지” 결정하기 위해 필수적인 단계다.

### FLEURS 데이터셋

102개의 언어가 포함된 대규모 다국어 음성 데이터셋.

```python
fleurs = load_dataset("google/fleurs", "all", split="validation", streaming=True)
sample = next(iter(fleurs))
```

Whisper 기반 LID 모델을 사용할 수 있다.

```python
classifier = pipeline(
    "audio-classification", 
    model="sanchit-gandhi/whisper-medium-fleurs-lang-id"
)
classifier(sample["audio"])
```

Whisper의 Encoder는 음성 표현을 매우 잘 파악하기 때문에 LID에서도 좋은 성능을 보인다.

## Zero-shot Audio Classification (CLAP 기반)

Zero-shot 오디오 분류는 미리 정의된 클래스 라벨이 없이 텍스트 설명을 직접 입력하여 분류하는 방식이다.

핵심 모델: CLAP (Contrastive Language-Audio Pre-training)

- Audio Encoder + Text Encoder
- 오디오와 텍스트를 동일한 임베딩 공간에 매핑
- similarity를 계산하여 가장 유사한 텍스트 라벨을 선택

### 예시: 개 짖는 소리 vs 청소기 소리

```python
classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused"
)

candidate_labels = ["Sound of a dog", "Sound of vacuum cleaner"]
classifier(audio_sample, candidate_labels=candidate_labels)
```

Zero-shot 모델은 환경음(ESC50)처럼 일반적이고 다양한 소리에 강하다.
하지만 언어 식별처럼 매우 미세한 차이를 구분해야 하는 경우에는 Whisper 기반 LID 모델이 훨씬 정확하다.

즉, Zero-shot은 범용성이 있으나 정밀도가 높은 태스크에는 한계가 있다.

## Hugging Face Hub에서 오디오 모델 찾기

1. https://huggingface.co/models로 이동
2. 왼쪽의 Task 메뉴에서 Audio Classification 선택
3. Dataset 필터 사용
- speech_commands
- minds14
- fleurs
4. Model Architecture 필터
- wav2vec2
- XLS-R
- AST
- Whisper

실제 업계에서도 이렇게 태스크와 데이터셋 기준으로 베이스라인을 선정한다.

## 다음 단계: 음악 분류(Music Classification)

오디오에는 음성(speech)만 있는 것이 아니다.
음악 데이터를 분류하거나 태깅하는 작업으로 확장할 수 있다.

- 음악 장르 분류
- 악기 분류
- 분위기/감정 분석
- 음악 태깅(multi-label classification)

Whisper, AST 등 기존 음성 기반 모델도 음악 스펙트로그램을 입력으로 활용하면 fine-tuning이 가능하다.

## 추가로 알아두면 좋은 핵심 개념

1. 스펙트로그램(Spectrogram)이 핵심 입력

Transformer는 원래 시계열보다는 2D 형태의 구조에 강하므로,
오디오 파형을 바로 넣기보다 Mel Spectrogram 혹은 STFT를 사용한다.

2. 오디오 분류는 결국 시계열 분류

텍스트의 token -> 오디오의 frame
CLS 토큰 -> mean pooling

구조적 유사성이 매우 높다.

3. 지연 시간(latency) 중요성

Wake word detector는 계속 실행된다.
따라서 모델은 반드시 가볍고 빠른 구조여야 한다.

4. Zero-shot 모델은 범용, 전문 모델은 정밀

CLAP은 다양한 소리에는 강하지만,
LID, Emotion Classification 등은 전문 모델이 훨씬 정확하다.

5. Whisper Encoder의 강점

Whisper Encoder는 음성 표현학습에 매우 뛰어나,
LID, 음성 감정 분석 등에서 좋은 성능을 보인다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn