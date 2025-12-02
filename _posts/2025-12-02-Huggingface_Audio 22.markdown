---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - Text-to-Speech(TTS) 데이터셋"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_Audio
---

# Text-to-Speech(TTS) 데이터셋

Text-to-Speech(TTS), 즉 음성 합성은 텍스트를 사람 같은 음성으로 바꾸는 작업이다.
하지만 이 작업은 단순히 텍스트를 음성으로 변환하는 것이 아니며, 다양한 기술적 난제를 포함한다.

이 글에서는 TTS 데이터셋의 개념, 어려움, 핵심 요구 조건, 대표 데이터셋을 정리한다.

## TTS 데이터셋의 난제

### Alignment 문제(정렬 문제)

TTS 모델은 문장을 받으면, 그 안의 **각 음소(phoneme)**가
얼마나 길게, 어떤 발음으로, 어떤 타이밍으로 발화되는지 알아야 한다.

- ASR: 오디오 -> 텍스트 (정답이 하나로 고정)
- TTS: 텍스트 -> 오디오 (가능한 정답이 아주 다양)

즉, TTS는 one-to-many 문제다.

예를 들어 “Hello!”라는 문장은 다음처럼 여러 버전이 모두 정답이다.

- 남성/여성
- 빠르게/느리게
- 밝게/차분하게
- 끊어서/연결해서

그래서 “텍스트와 오디오를 정확히 매칭시키는 것” 자체가 큰 과제다.

### 긴 문맥(Long-distance dependency)

자연스러운 발화를 위해서는 문장 전체의 흐름을 파악해야 한다.

예시

“나는 학교에 갔는데, 생각보다 사람이 많아서 놀랐다.”

이 문장의 억양과 속도는 뒤의 내용까지 고려하여 자연스럽게 이어져야 한다.
즉, 모델은 긴 문장을 기억하며 음성을 생성해야 한다.

### 데이터 품질 문제

TTS는 고품질(clean), 정제된 녹음을 필요로 한다.

ASR 데이터셋은 다음과 같은 특징이 많다.

- 잡음이 섞인 거리 녹음
- 마이크 음질이 낮은 경우
- 말투가 일정하지 않은 음성

ASR에는 도움이 되지만, TTS에 사용하면 다음과 같은 문제가 생긴다.

- 생성된 음성이 잡음과 함께 나옴
- 목소리가 흔들리거나 불안정함
- 말투가 들쑥날쑥한 음성 생성

그래서 ASR 데이터는 TTS에 그대로 쓰기 어렵다.

### 멀티스피커 데이터의 부족

고품질, 다국어, 다화자 음성을 모으는 것은 매우 어렵다.

- 비용이 비쌈
- 언어별 데이터가 극도로 부족함
- 감정, 억양까지 포함하려면 더 많은 녹음이 필요

그래서 많은 사람들이 “TTS 데이터 어디서 구해요?”라는 질문을 하게 된다.
아래에서 바로 그 답을 소개한다.

## 대표적인 TTS 데이터셋

아래는 HuggingFace Hub에서 TTS 연구에 가장 널리 쓰이는 데이터셋이다.

### LJSpeech

링크: https://huggingface.co/datasets/lj_speech

특징

- 13,100개의 영어 문장-음성 쌍
- 1명의 여성 화자가 7권의 비소설 책을 읽은 음성
- 데이터 품질이 매우 좋음
- 대부분의 “Hello World TTS 모델”은 LJSpeech로 데모를 만든다

언제 쓰면 좋을까?

- 단일 화자(single-speaker) TTS
- 기본 구조 테스트
- Tacotron2 / Glow-TTS / FastSpeech2 같은 모델 실험

실제로 HuggingFace의 많은 튜토리얼이 LJSpeech 기반이다.

### Multilingual LibriSpeech

링크: https://huggingface.co/datasets/facebook/multilingual_librispeech

특징

- LibriSpeech(영어)의 다국어 확장판
- 포함 언어: 영어, 독일어, 네덜란드어, 스페인어, 프랑스어, 이탈리아어, 포르투갈어, 폴란드어
- 정제된 오디오 + 텍스트 정렬

장점

- 다국어(multi-lingual) 모델 학습에 좋음
- 언어별 발음 학습이 가능
- cross-lingual TTS 실험에 특히 유용

주의점

- 기본적으로 ASR 기반이라 TTS에 완벽하게 최적화된 품질은 아님
- 잡음 유입 가능성 있음 -> fine-tuning 용으로는 OK

### VCTK (Voice Cloning Toolkit)

링크: https://huggingface.co/datasets/vctk

특징

- 110명의 영어 화자(남녀, 다양한 억양)
- 각 화자당 약 400 문장
- 음성 품질 양호
- 다양한 발음/억양 데이터 제공

언제 쓰면 좋을까?

- 멀티 스피커 TTS
- 보이스 클로닝(voice cloning)
- Accent Transfer 실험
- Speaker embedding 학습

“화자 임베딩 하나 넣으면 어떤 목소리든 생성하는 TTS” 모델을 만들 때 필수다.

### LibriTTS / LibriTTS-R

링크: https://huggingface.co/datasets/cdminix/libritts-r-aligned

특징

- 약 585시간이라는 매우 큰 규모
- 영어 기반 멀티스피커
- 24kHz 고품질 음성
- 잡음 음성 제거
- 문장 단위로 세분화
- 원본 텍스트 + 정규화 텍스트 둘 다 제공

장점

- 실전 TTS 연구에서 가장 많이 쓰이는 멀티스피커 데이터셋
- 대규모 모델 학습 가능
- 문맥(context) 정보까지 포함되어 문장 길이 실험 가능

## 좋은 TTS 데이터셋의 조건

TTS용 데이터는 다음 기준을 충족해야 한다.

### 고품질 음성

- 스튜디오 수준의 잡음 제거
- 일정한 마이크 환경
- 일정한 말투(발음 품질 안정성)
- clipping, echo, pop noise가 없어야 함

TTS는 **오디오 품질이 80%**라고 해도 과언이 아니다.

### 정확한 전사(Transcription)

ASR과 달리, TTS는 텍스트 기준으로 음성을 만든다.
따라서 텍스트와 오디오가 1:1로 매칭되어야 한다.

예시

- 오디오: "Hello, I'm John."
- 텍스트: "Hello, I am John."

이 차이 정도로 모델 품질에 악영향을 준다.

### 다양한 화자/악센트/스타일

- 여러 화자의 목소리
- 다양한 감정
- 여러 속도
- 여러 억양

이 다양성이 높을수록, 멀티스피커 TTS 성능이 강해진다.

### 다양한 언어, 도메인 포함

일상 대화, 뉴스, 소설, 대본 등 다양한 문체가 포함되어야
모델이 다양한 텍스트에 자연스럽게 대응할 수 있다.

## 굳이 TTS를 "처음부터" 학습할 필요는 없다

요즘은 HuggingFace Hub에 이미 최상급 TTS 모델이 많다.

예시

- VITS 계열
- FastSpeech2 계열
- Bark
- XTTS v2
- StyleTTS2
- VALL-E 스타일 모델들

즉, 대부분의 경우,

- fine-tuning만으로도 충분함
- 음성 클로닝은 1~3분의 음성만으로도 가능
- HuggingFace의 tts pipeline이나 Coqui TTS 활용 가능

## 마무리

- TTS는 텍스트 발생 + 음성 생성이라는 두 개의 난제를 모두 포함함
- 데이터 품질이 매우 중요
- 멀티스피커/고품질/정확한 정렬 데이터는 확보가 어렵지만 HuggingFace에 이미 좋은 데이터가 많음
- ASR 데이터는 TTS에는 적합하지 않음(잡음 때문)
- 대규모 TTS 연구에는 LibriTTS 계열 / VCTK가 정석
- 단일 화자 TTS는 LJSpeech가 표준

참고자료
Huggingface, Audio Course, https://huggingface.co/learn