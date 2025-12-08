---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - Speech-to-Speech Translation (STST) 정리"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_Audio
---

# Speech-to-Speech Translation (STST) 정리

Speech-to-Speech Translation(STST, 음성-음성 번역)은 한 언어의 말소리를 입력받아, 다른 언어의 말소리로 대답하는 모델을 말한다.
기존의 텍스트 번역(MT), 음성 인식(ASR), 음성 합성(TTS) 기술들이 결합되어 만들어지는 복합 AI 시스템이라고 볼 수 있다.

STST는 단순히 입력 음성을 텍스트로 바꾸는 것을 넘어서, 바로 다른 언어로 말해주는 기능을 제공한다는 점에서 차별화된다.
즉, 실제 사람 간 대화처럼 “듣고 -> 이해하고 -> 말하는” 전체 과정을 모델이 자동으로 수행한다.

## STST는 왜 중요한가?

### 기존 텍스트 기반 번역의 한계

일반적으로 사람들은 번역할 내용을 직접 타이핑해야 했다.
하지만 많은 상황에서는 말로 전달하는 편이 자연스럽고 빠르다.

STST는 이런 현실적인 요구를 해결한다.

- 외국인과 직접 음성 대화 가능
- 이동 중에도 실시간 번역 가능
- 문자 입력이 어렵거나 불편한 상황에서도 사용 가능

## Cascaded Pipeline - 실용적인 STST 구성 방법

STST 모델을 만드는 가장 간단한 방법은 다음 두 모델을 연결하는 것이다.

1. Speech Translation(ST): 음성을 받아 번역된 텍스트 생성
2. Text-to-Speech(TTS): 번역된 텍스트를 음성으로 합성

이 방법은 구현이 단순하며, 이미 성능 검증된 ASR/MT/TTS 모델을 조합할 수 있기 때문에 데이터와 연산량이 적게 든다는 장점이 있다.

## Speech Translation 단계 (Whisper 활용)

Whisper는 다국어 음성 인식 + 번역 모델로,
96개 이상의 언어를 영어로 번역할 수 있는 ST 모델이다.

Whisper Base 모델 로드

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

코드 설명

- pipeline()은 Hugging Face가 제공하는 고수준 API로, 토크나이저/모델/후처리를 자동으로 묶어준다.
- Whisper는 inference 시 내부적으로 "task" 설정에 따라 번역 모드(translate) 또는 **음성 인식 모드(transcribe)**로 동작한다.

## 데이터 불러오기 - VoxPopuli

이탈리아어(it) 음성 샘플을 스트리밍 모드로 불러온다.

```python
from datasets import load_dataset

dataset = load_dataset("facebook/voxpopuli", "it", split="validation", streaming=True)
sample = next(iter(dataset))
```

왜 스트리밍인가?

- 거대한 오디오 데이터셋을 모두 다운로드하면 시간이 오래 걸림
- streaming=True는 필요할 때만 데이터를 읽어온다

## Whisper로 번역 함수 정의

```python
def translate(audio):
    outputs = pipe(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})
    return outputs["text"]
```

내부에서 일어나는 과정

- Whisper processor가 오디오를 멜스펙트로그램으로 변환
- 모델이 auto-regressive decoding으로 영어 문장 생성
- "task": "translate" -> 번역 모드

## 번역 결과 비교

Whisper 출력

```python
"psychological and social. I think that it is a very important step..."
```

실제 이탈리아어 원문과 자연스럽게 대응됨을 확인.
Whisper가 문맥을 완전히 이해하는 것은 아니지만
전반적인 의미 중심 번역에는 강하다.

## Text-to-Speech 단계 (SpeechT5 활용)

Whisper로 얻은 영어 텍스트를 영어 음성으로 합성해야 한다.

Transformers에는 아직 TTS 전용 pipeline이 없기 때문에
직접 모델을 호출하는 형태로 구현한다.

## SpeechT5 로드

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```

구성 설명

- processor: 텍스트 -> input_ids 변환
- 모델: SpeechT5 Transformer 본체
- vocoder: 생성된 멜스펙트로그램 -> 실제 오디오 파형 생성

## Speaker Embedding 적용

SpeechT5는 입력 텍스트뿐 아니라
화자의 음색을 컨트롤하는 speaker embedding을 받는다.

```python
from datasets import load_dataset
import torch

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
```

9. 음성 합성 함수 정의
def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device),
        speaker_embeddings.to(device),
        vocoder=vocoder
    )
    return speech.cpu()

내부 동작 설명

1. 텍스트는 tokenizer를 통해 input_ids로 변환
2. SpeechT5 encoder-decoder가 mel-spectrogram 생성
3. HiFi-GAN vocoder가 실제 음성 파형을 합성

## ST + TTS 결합: Speech-to-Speech 함수 만들기

이제 두 단계를 연결해 하나의 end-to-end 함수로 만든다.

```python
import numpy as np

target_dtype = np.int16
max_range = np.iinfo(target_dtype).max

def speech_to_speech_translation(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech.numpy() * max_range).astype(np.int16)
    return 16000, synthesised_speech
```

왜 int16로 변환하는가?

- WAV 파일 기본 표준은 16-bit PCM
- Gradio의 Audio 출력도 int16을 기대함

## Gradio 데모 만들기

마이크 입력과 파일 입력을 모두 지원하는 UI 구성.

```python
import gradio as gr

demo = gr.Blocks()

mic_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
)

with demo:
    gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])

demo.launch(debug=True)
```

## Cascaded STST의 한계와 최신 흐름

1. 장점

- 구현 쉬움
- 기존 모델 재사용 가능
- 데이터 요구량 적음

2. 단점

- Error propagation
    - ST가 오역 -> TTS도 그대로 오역
- Latency 증가
    - 여러 모델을 연달아 실행

3. 최신 기술: Direct S2ST

- 텍스트를 거치지 않고 음성 -> 음성 직접 변환
- 화자 음색 보존 가능
- 예: Translatotron, SeamlessM4T 등

참고자료
Huggingface, Audio Course, https://huggingface.co/learn