---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 오디오 인식을 위한 데이터셋 선택"
date:   2025-11-29 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 인식을 위한 데이터셋 선택

- Whisper, Wav2Vec2 등 오디오 인식 모델을 학습할 때 꼭 알아야 하는 핵심 개념 총정리

ASR(Automatic Speech Recognition) 모델의 성능은 데이터 품질이 80% 이상을 차지한다.
심지어 Whisper처럼 강력한 pretrained 모델이라도, fine-tuning 시 데이터 선택 기준이 맞지 않으면 성능이 오히려 떨어질 수 있다.

이 글에서는 오디오 인식용 데이터셋을 고를 때 반드시 알아야 하는 조건을 정리하고,
허깅페이스 Hub의 주요 데이터셋을 도메인/스타일/포맷별로 비교한다.

## 음성 데이터셋을 고를 때 고려해야 하는 4가지 핵심 요소

오디오 인식 성능은 다음 4가지 요소에 의해 크게 달라진다.

### 데이터 총 시간(hours)

데이터셋 크기는 **총 오디오 길이(시간)**로 표현된다.
텍스트 데이터셋의 “샘플 수”와 같은 개념이다.

하지만 크면 무조건 좋은 것은 아니다.
중요한 것은 “크기 + 다양성”이다.

예시

- 100시간의 다양한 화자, 다양한 환경 데이터
- 1000시간의 한 화자, 한 환경 데이터

실제 모델 성능은 전자가 훨씬 높다.

**왜 다양성이 중요한가?**

Whisper나 CTC 모델은 “음성 패턴”을 배우기 때문에
특정 화자, 도메인, 녹음 환경이 과대표집되면 과적합이 일어난다.

또한, 실제 환경(노이즈, 억양 등)에 약해진다.

### Domain(도메인)

오디오가 어디에서 수집되었는가?


| Domain               | 특징                 | 예               |
| -------------------- | ------------------ | --------------- |
| **Audiobook**        | 조용함, 고품질, 스크립트 기반  | LibriSpeech     |
| **YouTube**          | 다양한 환경, 잡음, 실생활 화법 | GigaSpeech      |
| **Meetings/Finance** | 전문 용어, 빠른 말속도      | AMI, SPGISpeech |
| **Crowdsourced**     | 다양한 화자/마이크/환경      | Common Voice    |


왜 도메인이 중요한가?

모델은 항상 학습 도메인과 비슷한 환경에서 가장 잘 작동한다.

예시

- 오디오북으로 학습한 모델 -> 실제 대화(잡음, 억양)에서는 성능 급락
- 회의 음성으로 학습한 모델 -> 유튜브 브이로그에서는 오류 증가

따라서 실제 사용 환경과 맞는 도메인을 고르는 것이 핵심이다.

### Speaking Style (화법 스타일)

1. Narrated (낭독/스크립트 기반)

- 정확한 발음
- 문장 구조가 깔끔
- 억양 변화 적음
- 오류 거의 없음
-> LibriSpeech, Common Voice Wikipedia 텍스트

2. Spontaneous (자유 대화/즉흥적 발화)

- 말 더듬기
- 불필요한 반복
- 끼어드는 소리
- 일상 언어(uhh, like, you know...) 자주 등장
-> 미팅 데이터(AMI), 금융 콜, 유튜브

이유

모델이 배우는 텍스트 패턴이 완전히 다르기 때문이다.

- Narrated: 정제된 텍스트에 강함
- Spontaneous: 자연스러운 회화에 강함

### Transcription Style (전사 스타일)

텍스트가 어떻게 적혀있는가?


| 요소              | 의미            |
| --------------- | ------------- |
| **Casing**      | 대소문자          |
| **Punctuation** | 마침표, 쉼표, 따옴표  |
| **Verbatim 여부** | 발화 그대로 적는지 여부 |


Whisper fine-tuning에서는 특히 중요하다.
Whisper는 원래 다음을 모두 포함하는 출력을 생성한다.

- 구두점
- 대소문자
- 문장 구조

따라서 전사 데이터가 이런 요소를 얼마나 포함하느냐에 따라 fine-tuning 방향이 달라진다.

예시

- 데이터셋이 모두 소문자, 구두점 없음 -> Whisper fine-tuning 시 그런 스타일로 변화
- Fully formatted transcription -> Whisper도 문장 단위로 깔끔하게 출력

## 주요 영어 ASR 데이터셋 비교표

아래는 허깅페이스 Hub에서 자주 쓰이는 영어 음성 인식 데이터셋의 도메인, 스타일·사용 목적을 비교한 표다.


| Dataset             | Hours  | Domain             | Style       | Casing | Punct | License        | Use case      |
| ------------------- | ------ | ------------------ | ----------- | ------ | ----- | -------------- | ------------- |
| **LibriSpeech**     | 960    | Audiobooks         | Narrated    | ❌      | ❌     | CC-BY-4.0      | Benchmark, 연구 |
| **Common Voice 11** | 3000   | Wikipedia 발화       | Narrated    | ✅      | ✅     | CC0            | 다양한 화자, 발음    |
| **VoxPopuli**       | 540    | EU Parliament      | Oratory     | ❌      | ✅     | CC0            | 비원어민 억양       |
| **TED-LIUM**        | 450    | TED Talks          | Oratory     | ❌      | ❌     | CC-BY-NC-ND    | 전문 분야         |
| **GigaSpeech**      | 10,000 | YouTube/Podcast    | Mixed       | ❌      | ✅     | Apache 2.0     | 복합 환경         |
| **SPGISpeech**      | 5000   | Financial meetings | Mixed       | ✅      | ✅     | User agreement | 금융 회의록        |
| **Earnings22**      | 119    | Finance            | Mixed       | ✅      | ✅     | CC-BY-SA       | 억양 다양성        |
| **AMI**             | 100    | Meetings           | Spontaneous | ✅      | ✅     | CC-BY          | 노이즈/대화 환경     |


- Common Voice: 다양한 화자 + 구두점 포함
- GigaSpeech: 여러 환경에서 로버스트한 모델 필요할 때
- LibriSpeech: 실전보단 연구용

## 주요 다국어(multi-lingual) 데이터셋

영어 외 언어를 훈련할 때 중요한 선택 기준은 언어 다양성 + 도메인 + 화자 수다.

| Dataset                      | Languages | Domain                   | Style       | Case | Punct | 사용 목적         |
| ---------------------------- | --------- | ------------------------ | ----------- | ---- | ----- | ------------- |
| **Multilingual LibriSpeech** | 6         | Audiobooks               | Narrated    | ❌    | ❌     | 연구용           |
| **Common Voice 13**          | 108       | Wikipedia + Crowdsourced | Narrated    | ✅    | ✅     | 다양한 억양/화자     |
| **VoxPopuli**                | 15        | EU Parliament            | Spontaneous | ❌    | ✅     | 유럽 언어         |
| **FLEURS**                   | 101       | News-like                | Spontaneous | ❌    | ❌     | 평가용 benchmark |


## Common Voice 13을 선택하는 이유

Whisper fine-tuning을 가정하면, Common Voice 13은 매우 좋은 선택이다.

1. 108개 언어 지원

저자원 언어까지 폭넓게 포함

2. 구두점, 대소문자 포함

Whisper의 “formatted text output” 능력과 잘 맞음

3. Crowdsourced

다양한 환경, 다양한 억양

4. 허깅페이스 Hub에서 바로 미리 듣기 가능

데이터 품질을 직접 확인할 수 있음

## Dataset Preview 활용 팁

허깅페이스 Dataset Preview는 실제 오디오를 듣고 판단할 수 있는 최고의 도구다.

1. 녹음 품질

- 잡음
- 볼륨
- 마이크 퀄리티
- 방 안 울림

2. 화자 다양성

- 성별
- 억양
- 나이대
- 발음 스타일

3. 텍스트 포맷

- 구두점 포함 여부
- 대소문자
- 숫자 처리 방식 (“twenty-three” vs “23”)
- 외래어 표기법

Whisper fine-tuning 시 텍스트 포맷과 출력 스타일이 반드시 일치해야 한다.

## “언어를 모르는 상황”에서 모델을 평가하는 법

예를 들어 Dhivehi(디베히)처럼 모르는 언어를 fine-tuning한다면?

모델의 품질을 확인하려면 다음이 필요하다.

1. Validation/Test split 필수
2. ASR 평가 지표

- WER (Word Error Rate)
- CER (Character Error Rate)

3. 기준 모델(Whisper base 등)과 비교
4. 음성 샘플을 직접 듣고 “합리적 발화”인지 감으로 평가

## Summary

- ASR 데이터셋 선택은 도메인, 화법, 포맷, 시간 4요소가 핵심.
- Whisper처럼 formatted text를 출력하는 모델은 구두점, 대소문자가 있는 데이터셋이 유리.
- 도메인 mismatch(오디오북 학습 -> 회의 음성 평가)는 성능을 크게 떨어뜨림.
- Common Voice 13은 Whisper fine-tuning에 가장 실제적이고 유연한 선택.
- Dataset Preview에서 직접 듣고 판단하는 것이 가장 확실.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn