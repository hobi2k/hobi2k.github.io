---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - Gradio로 음악 분류 데모 구축하기"
date:   2025-11-26 00:10:22 +0900
categories: Huggingface_Audio
---

# Gradio로 음악 분류 데모 구축하기

이 글에서는 GTZAN 음악 장르 분류 모델(DistilHuBERT 기반)을 이용해
Gradio 인터페이스로 바로 실행 가능한 데모를 만드는 과정을 설명한다.

Gradio는 머신러닝 모델을 웹 UI로 매우 쉽게 보여줄 수 있는 프레임워크이며,
Hugging Face Spaces와도 바로 연동된다.

## 파인튜닝된 모델 불러오기

우리는 이미 DistilHuBERT를 GTZAN으로 파인튜닝한 모델을 업로드했다고 가정한다.
이 모델을 불러오는 가장 간단한 방법은 Hugging Face pipeline()을 이용하는 것이다.

```python
from transformers import pipeline

model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"  # 자신의 모델로 교체 가능
pipe = pipeline("audio-classification", model=model_id)
```

### pipeline이 하는 일

pipeline("audio-classification")은 아래 작업을 자동으로 처리한다.

- 파일 로딩 (wav, mp3 등)
- sampling rate 확인 및 자동 resampling
- feature extractor 적용 (normalization, padding)
- 모델 forward
- softmax 후 label + score 형태 반환

따라서 입력이 오디오 파일 경로만 되면 전체 처리 과정이 자동으로 진행된다.

## 오디오 분류 함수 작성

Gradio는 UI 입력을 함수에 전달하기 때문에,
사용자의 입력 오디오 파일 경로를 받아 pipeline에 전달하는 간단한 함수를 작성한다.

```python
def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs
```

### 이 함수에서 중요한 점

1. filepath는 Gradio에서 전달하는 로컬 임시 파일 경로
2. pipe(filepath)를 하면

- 오디오 파일 로드
- 모델의 sampling rate에 맞춰 자동 변환
- model inference

3. top-N 클래스와 점수를 리스트로 반환

Gradio `Label` 컴포넌트는 dict 형태를 입력으로 받기 때문에
{label: score} 형태로 변환한 뒤 반환해야 한다.

## Gradio 인터페이스 생성

```python
import gradio as gr

demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label()
)
demo.launch(debug=True)
```

### 구성 설명

1. inputs = gr.Audio(type="filepath")

- type="filepath"이 매우 중요하다.
- 기본값 "numpy"는 오디오 신호 배열만 전달한다.
- pipeline은 “파일 경로”를 기대하므로 filepath가 맞다

2. outputs = gr.Label()

- dict 형태의 {라벨: 점수}를 깔끔하게 시각화해주는 컴포넌트다.
- 자동으로 가장 높은 점수를 하이라이트한다.

3. launch(debug=True)

- debug 모드에서는 콘솔에 자세한 로그를 출력한다.
- 개발 단계에서 매우 유용하다.
- 배포 시에는 보통 debug=False 또는 생략한다.

## 실행 화면

이 코드를 실행하면 다음과 같은 형태의 Gradio 웹 UI가 로컬에서 열린다.

- 브라우저에서 파일 업로드 또는 녹음
- 모델이 음악 파일을 분석하고
- 각 장르에 대한 confidence score를 시각적으로 출력

Gradio의 장점은
“코드 몇 줄로 인터랙티브 오디오 데모를 만들 수 있다는 것”이다.

## Hugging Face Space로 배포하기

완성된 데모는 Hugging Face Space에서 아래처럼 실행될 수 있다.

https://course-demos-song-classifier.hf.space


Space에서 실행하면 좋은 이유

- GPU 지원 가능 (T4/A10)
- URL만 공유해도 누구나 실행 가능
- 모델과 데모를 한 곳에서 관리

Space 배포 시 필요한 파일

- app.py
- requirements.txt


requirements 예시

```python
transformers
datasets
gradio
torch
soundfile
```

## 추가 정보

1. mp3 파일 인식

pipeline()은 음성 파일을 soundfile 또는 ffmpeg로 읽는다.
Hugging Face 환경에서는 mp3 -> wav 자동 변환이 수행된다.

2. 긴 오디오 파일 처리

GTZAN 모델은 30초 기준으로 학습되었기 때문에
5분짜리 음악을 넣으면 앞부분 일부만 분석할 수 있다.
필요하다면 아래를 커스텀해야 한다.

- 자동 segmentation
- sliding window 방식
- 평균 score 산출
- 가장 가능성 높은 segment 추출

3. 모델 latency

DistilHuBERT는 Encoder-only 모델이므로
GPU가 없더라도 빠른 편이다.
배포 시 CPU-only 환경에서도 실용적이다.

4. 실시간 마이크 입력

inputs=gr.Audio(source="microphone", type="filepath")로 바꾸면
마이크 녹음 후 바로 모델에 전달할 수 있다.

## 마무리

이 Gradio 데모는 다음 흐름으로 구성된다.

- Hugging Face Hub에서 fine-tuned 모델 로드
- pipeline()으로 inference 준비
- classify_audio() 함수로 라벨/점수 반환
- Gradio Interface로 웹 UI 구성
- 로컬 or Hugging Face Space에서 실행 및 배포

오디오 분류 모델을 빠르게 시연하거나 공유하기 위한
가장 직관적이고 실용적인 방법이 바로 Gradio이다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn