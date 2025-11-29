---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - Gradio로 Whisper ASR 데모 만들기"
date:   2025-11-29 00:10:22 +0900
categories: Huggingface_Audio
---

# Gradio로 Whisper ASR 데모 만들기

Whisper 모델을 파인튜닝했다면, 이를 바로 웹에서 사용할 수 있는 데모로 만들 수 있다.
이 데모는 브라우저에서 바로 음성을 녹음하거나 파일을 업로드하여 실시간으로 음성을 텍스트로 변환해 준다.
이 포스트에서는 Hugging Face pipeline() + Gradio로 실제 서비스 가능한 데모를 만드는 과정을 정리한다.

## 파인튜닝된 Whisper 모델 로드하기

Whisper 모델을 불러오는 방법은 Hugging Face 모델 사용 방식과 동일하다.
model_id에 Hub에 업로드한 모델 또는 기본 Whisper 모델을 입력하면 된다.

```python
from transformers import pipeline

model_id = "sanchit-gandhi/whisper-small-dv"  # 자신이 fine-tuning한 모델 ID로 변경
pipe = pipeline("automatic-speech-recognition", model=model_id)
```

pipeline을 사용하는 이유

pipeline("automatic-speech-recognition")은 기본적으로 다음 작업을 모두 처리한다.

- 오디오 파일 자동 로딩
- 샘플링레이트 변환(Whisper는 16kHz)
- feature extraction
- 모델 inference
- 토큰 디코딩

즉, 전처리 -> 모델 -> 후처리 전부 자동이다.
따라서 Gradio에서 처리해야 할 작업이 거의 없다.

## 음성 파일을 받아 텍스트로 변환하는 함수 정의

Gradio는 파일 경로 또는 raw 오디오 배열을 넘겨주는데, pipeline은 파일 경로도 처리할 수 있다.

```python
def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",   # 번역이 아니라 원 언어 그대로 텍스트로 변환
            "language": "sinhalese" # Whisper tokenizer의 언어 설정 (파인튜닝 언어 기반으로 수정)
        },
        chunk_length_s=30, # 긴 오디오 chunking 활성화
        batch_size=8, # 병렬 배치 처리
    )
    return output["text"]
```

chunking을 사용하는 이유

Whisper는 최대 30초 오디오를 입력으로 받는다.
30초보다 더 긴 오디오는 그냥 잘려서 인식이 안 된다.
따라서 다음이 필요하다.

- 오디오를 30초씩 분할
- 각 chunk를 개별적으로 인식
- 겹치는 구간으로 이어붙여 정확도 손실 방지

즉, chunking은 긴 오디오를 안정적으로 처리하기 위한 필수 기능이다.

## Gradio Blocks로 UI 구성하기

Gradio에는 여러 방식이 있지만, 여기서는 Blocks 기반 탭 UI를 사용한다.

```python
import gradio as gr

demo = gr.Blocks()

# 1) 마이크 입력
mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.components.Textbox(),
)

# 2) 파일 업로드 입력
file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.components.Textbox(),
)
```

마이크 입력 vs 파일 업로드 차이


| 입력 방식                  | 기능                            |
| ---------------------- | ----------------------------- |
| `sources="microphone"` | 웹 브라우저에서 음성 녹음 후 바로 처리        |
| `sources="upload"`     | 음성 파일(.wav, .mp3 등)을 업로드하여 처리 |


둘 다 Whisper pipeline이 자동 전처리하므로 특별한 처리 필요 없음.

## Gradio 앱 실행하기

```python
with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(debug=True)
```

debug=True 의 의미

- 오류 발생 시 traceback을 UI에 표시
- 개발 단계에서 문제 해결에 매우 유용
- 배포 후에는 보안상 False 권장

## Hugging Face Spaces로 배포하기

로컬에서만 사용하는 것이 아니라, Hugging Face Spaces에 배포하면 다음을 할 수 있다.

- 웹에서 누구나 접속 가능
- API 형태로도 사용 가능
- 모델과 데모를 한곳에 통합 관리
- 템플릿 Space 복사 방법

아래 템플릿을 복제하면 바로 Whisper 데모가 만들어진다.

https://huggingface.co/spaces/course-demos/whisper-small?duplicate=true

복제 후 해야 할 작업

- 좌측 메뉴: Files and versions 클릭
- app.py 열기
- 모델 경로를 자신의 모델로 수정
- Commit changes

Space가 재시작되면 완성.

## 실전 운영 팁: Gradio + Whisper 운영시 알아두면 좋은 점

1. Whisper 모델 크기별 속도 차이


| 모델     | 처리 속도 | 정확도   | VRAM  |
| ------ | ----- | ----- | ----- |
| tiny   | 매우 빠름 | 낮음    | 1GB   |
| base   | 빠름    | 중간    | 1.5GB |
| small  | 보통    | 좋음    | 2.3GB |
| medium | 느림    | 매우 좋음 | 4GB   |
| large  | 매우 느림 | 최고    | 7.5GB |


웹 데모에서는 대개 tiny 또는 small 추천.

2. CPU에서도 실행 가능하지만 느림

음성 인식은 GPU 사용 시 10배 이상 빠름.
Spaces에서 GPU 선택도 가능.

3. 실서비스에서는 오디오 파일 길이 제한을 반드시 둘 것

긴 오디오(chunking 사용)의 경우 다음 문제가 발생할 수 있다.

- 메모리 증가
- 처리 시간 증가
- 여러 사용자가 동시에 접속하면 서버 과부하
- 따라서 max 30~60초 권장.

4. Whisper multilingual에서 language 설정은 매우 중요

예시

- "task": "transcribe" -> 해당 언어 그대로 출력
- "task": "translate" -> 영어로 번역 출력
- "language": "korean" 같은 설정이 tokenizer에 직접 영향을 줌

언어 설정이 제대로 안 되면 인식률이 급격히 떨어진다.

## 정리

이 글에서는 Whisper ASR 모델로 Web 데모를 만드는 전체 과정을 정리했다.

- Whisper 모델을 pipeline으로 로드
- 오디오 입력을 처리하는 함수 구현
- Gradio로 마이크 및 파일 업로드 UI 제공
- Hugging Face Spaces로 배포까지 가능

이 구조만 이해하면 어떤 음성 인식 모델이든 즉시 웹 데모로 만들 수 있다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn