---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - Voice Assistant 정리"
date:   2025-12-08 00:10:22 +0900
categories: Huggingface_Audio
---

# Voice Assistant 정리

이 글에서는 Amazon Alexa, Apple Siri처럼 음성으로 깨우고 음성으로 질문하고 음성으로 대답하는 AI 비서 시스템(voice assistant)을 직접 만들어본다.

음성 기반 비서는 아래 4단계 파이프라인으로 구성된다.

## Wake Word Detection

음성 비서는 항상 마이크를 듣고 있지만,
아무 말이나 반응하면 배터리를 소모하고 매우 불편하다.
따라서 반드시 특정 단어(= wake word)를 들었을 때만 활성화된다.

핵심 특징

- 항상 실행되어야 하므로 경량 모델이 필요함
- Audio Classification 모델을 사용
- 흔히 Speech Commands 데이터셋으로 훈련됨
- 여기서는 “marvin”이라는 단어를 wake word로 사용

### 사전 훈련된 Audio Classification 모델 찾기

Hugging Face Hub -> Audio Classification -> Speech Commands 필터 적용
-> MIT/ast-finetuned-speech-commands-v2 모델 선택.

이 모델은 35개 단어 중 "marvin"도 포함하고 있다.

```python
from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-speech-commands-v2",
    device=device
)
```

라벨 확인

```python
classifier.model.config.id2label[27]
# 'marvin'
```

### Wake Word 실시간 감지

ffmpeg_microphone_live() 함수를 사용하면
마이크 입력을 일정한 길이로 잘라 실시간 스트리밍 형태로 모델에 전달할 수 있다.

이 함수의 내부 동작 요약

- 마이크 소리를 chunk_length_s 길이로 잘라냄
- 그 안에서도 stream_chunk_s 단위로 작은 조각을 먼저 보내 즉각 추론 시작
- sliding window 형태로 연속 오디오를 부드럽게 처리
- generator 형태로 반환 -> pipeline에 직접 전달 가능

```python
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

def launch_fn(
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False
):
    if wake_word not in classifier.model.config.label2id:
        raise ValueError("Invalid wake word.")
    
    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Listening for wake word...")
    for prediction in classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)

        if prediction["label"] == wake_word and prediction["score"] > prob_threshold:
            return True
```

## Speech Transcription (ASR)

Wake word 감지 후, 사용자가 무엇을 말했는지를 텍스트로 바꿔야 한다.
Whisper Base English 모델을 사용한다.

```python
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    device=device
)
```

### 실시간 ASR을 위한 chunk 기반 스트리밍

Speech recognition은 오디오 전체를 기다리면 느리다.
따라서 ffmpeg_microphone_live를 통해 조각난 오디오를 연속 추론한다.

```python
import sys

def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate
    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")

        if not item["partial"][0]:  # 마지막 chunk
            break

        return item["text"]
```

이 방식은 반응 속도(실시간성) 과 정확도 사이에서 조절 가능하다.

- stream_chunk_s ↓: 더 빠른 반응
- stream_chunk_s ↑: 더 높은 정확도

## Query the LLM

사용자가 말한 텍스트를 LLM에게 보내 적절한 답변을 생성한다.

왜 LLM은 클라우드에서 실행하나?

- 파라미터 규모가 매우 크기 때문 (7B~70B)
- 단일 질문은 매우 짧아서 통신 비용은 낮음
- Hugging Face Inference API로 매우 간단하게 호출 가능

### Inference API로 LLM 질의

예: Falcon-7B-Instruct 모델 사용

```python
from huggingface_hub import HfFolder
import requests

def query(text, model_id="tiiuae/falcon-7b-instruct"):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HfFolder().get_token()}"}
    payload = {"inputs": text}

    print(f"Querying...: {text}")
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()[0]["generated_text"][len(text) + 1:]
```

이 방식의 장점

- 로컬은 마이크/ASR/TTS만 담당 -> 가벼움
- 인퍼런스는 GPU 서버에서 초고속 수행

## Text-to-Speech (TTS)

이제 LLM이 만든 텍스트를 음성으로 바꿔야 한다.
SpeechT5 TTS + HiFiGAN vocoder 조합을 사용한다.

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)


Speaker embedding:

from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
```

### TTS 함수

```python
def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device),
        speaker_embeddings.to(device),
        vocoder=vocoder
    )
    return speech.cpu()
```

## Full Pipeline

이제 4단계를 모두 연결하면 아래와 같다.

```python
launch_fn()  # wake word 감지
transcription = transcribe()  # ASR
response = query(transcription)  # LLM 답변
audio = synthesise(response)  # TTS

from IPython.display import Audio
Audio(audio, rate=16000, autoplay=True)
```

## Extensions

- 더 작은 Wake Word 모델

    - 35개 클래스 대신 “wake word vs not-wake-word” 이진 분류 모델 사용 가능

- 지연 시간을 줄이기

    - 필요할 때만 모델 로드: 전력 절약

- Voice Activity Detection(VAD) 추가

    - 사용자가 말이 끝났음을 자동으로 판단
    - 너무 길거나 너무 짧게 끊기는 문제 해결

## Using Transformers Agents - 텍스트, 오디오, 이미지까지 확장

Transformers Agents는 LLM이 “무엇을 해야 하는지” 스스로 판단하여
적절한 모델을 선택해 실행하는 멀티모달 AI 에이전트 시스템이다.

예: 이미지 생성

```python
from transformers import HfAgent

agent = HfAgent(
    url_endpoint="https://api-inference.huggingface.co/models/bigcode/starcoder"
)

agent.run("Generate an image of a cat")
```

응용: 음성 비서와 결합

```python
launch_fn()
transcription = transcribe()
agent.run(transcription)
```

-> “고양이 이미지 만들어줘” 라고 말하면 Stable Diffusion으로 이미지 생성
-> “내용을 설명해줘” 요청도 가능

참고자료
Huggingface, Audio Course, https://huggingface.co/learn