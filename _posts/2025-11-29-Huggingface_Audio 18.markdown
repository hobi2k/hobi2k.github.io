---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 오디오 인식 모델 평가 지표"
date:   2025-11-29 00:10:22 +0900
categories: Huggingface_Audio
---

# 음성 인식 모델 평가 지표

ASR(Automatic Speech Recognition) 시스템을 평가할 때는 **텍스트 정답(reference)**과 **모델 예측(prediction)**을 비교해
얼마나 정확하게 인식했는지를 측정한다.

ASR 평가 지표는 자연어 처리의 Levenshtein Distance(편집 거리) 개념과 동일하게
수정해야 하는 최소 횟수를 기반으로 계산된다.

모든 ASR 지표는 다음 세 가지 오류를 기반으로 한다.


| 구분                   | 의미              | 예             |
| -------------------- | --------------- | ------------- |
| **S (Substitution)** | 잘못된 단어/문자로 바뀜   | sat → sit     |
| **I (Insertion)**    | 불필요한 단어/문자가 추가됨 | sat → sat the |
| **D (Deletion)**     | 단어/문자가 누락됨      | sat on → sat  |


차이는

- 단어 단위로 비교하면 WER
- 문자 단위로 비교하면 CER

이 된다는 점이다.

## Word Error Rate (WER)

WER는 ASR 평가에서 가장 많이 사용되는 사실상 표준(de facto standard) 지표다.

### WER 계산 방식

WER = (S + I + D) / N
N = 정답(reference)의 총 단어 수
WER 값은 0 이상이며, 낮을수록 좋다.

예시는 다음과 같다.

Reference

- the cat sat on the mat

Prediction

- the cat sit on the


오류

- S = 1 ("sat" → "sit")
- I = 0
- D = 1 ("mat" 누락)

따라서 다음과 같은 수치가 나온다.

WER = (1 + 0 + 1) / 6 = 0.333 (33.3%)

### WER의 특성

단 하나의 철자 오류라도 단어 전체가 오답 처리된다.
문맥적 오류(시제, 의미 등)를 엄격하게 잡아낸다.
저해상도/자원 부족 언어에서도 자주 사용된다.

### WER 최대값은 얼마인가?

많은 사람이 WER의 최대를 1(100%)로 생각하지만
정답 단어보다 예측 단어 수가 훨씬 많으면 WER은 100% 넘을 수 있다.

예시: 정답 2단어, 예측 10단어 -> WER = 10/2 = 5 (500%)

따라서 WER가 100%보다 큰 경우도 자연스러을 때가 있다.
물론 모델이 매우 잘못된 결과를 냈다는 뜻이기도 하다.

## Word Accuracy (WAcc)

WER을 뒤집은 것.

WAcc = 1 - WER

높을수록 좋은 지표로 만들고 싶을 때 사용하지만,
ASR 연구에서 WAcc는 거의 사용하지 않는다.
WER 중심으로 평가하는 것이 일반적이다.

## Character Error Rate (CER)

WER가 단어 단위라면, CER는 문자 단위다.

### CER의 장점

철자 오류를 덜 가혹하게 평가한다.
단어 구분이 없는 언어(중국어, 일본어 등)에서 사실상 필수 지표다.

긴 합성어가 많은 언어에도 적합하다.

### CER 계산 예시

Reference

- the cat sat on the mat

Prediction

- the cat sit on the


문자 단위로 비교하면

- S = 1 (“a” → “i”)
- D = 4 ("mat"의 m,a,t 삭제)
- 총 문자 수 = 22

CER = (1 + 0 + 4) / 22 = 0.227 (22.7%)

WER보다 낮게 나온다.
이는 “sit”의 “s”와 “t”는 맞았기 때문이다.

## 어떤 지표를 언제 쓰는가?


| 언어/목적                  | 추천 지표   | 이유                |
| ---------------------- | ------- | ----------------- |
| 띄어쓰기 존재 (영어/한국어/독일어 등) | **WER** | 단어 의미/문맥까지 평가 가능  |
| 띄어쓰기 없음 (중국어/일본어)      | **CER** | 단어 개념이 모호함        |
| 오탈자 중심 평가              | **CER** | 문자 단위 정밀 비교       |
| 실제 서비스 품질 판단           | **WER** | 의미·문맥 오류를 강하게 잡아냄 |


Whisper나 Wav2Vec2 같은 모델의 학술 평가에서는 WER가 핵심 지표다.

## Normalisation (정규화)의 역할

평가 방식에 따라 WER이 크게 달라질 수 있다.
ASR 모델의 출력은 다음 중 하나일 수 있다.

- Orthographic (대소문자 + 구두점 포함)
- Non-orthographic (소문자 + 구두점 없음)

Whisper는 자연스럽게 구두점과 대소문자를 예측하지만,
CTC 모델(Wav2Vec2)은 대부분 소문자/무구두점 형태로 제공된다.

따라서 정규화 여부는 WER에 큰 영향을 준다.

### 정규화의 효과

- 소문자로 변환
- 구두점 제거
- 숫자 표기 통일 (twenty-one vs 21)

정규화를 적용하면 오류 판단 기준이 느슨해지므로 WER가 자연스럽게 낮아진다.

Whisper 논문에서도 정규화 후 WER가 크게 개선된다고 명시되어 있다.

### 정규화 전략


| 전략                                  | 장점                          | 단점                |
| ----------------------------------- | --------------------------- | ----------------- |
| **정규화 안 함** (orthographic 평가)       | 실제 사용 품질과 가장 가까움            | WER이 높게 나옴        |
| **정규화 후 평가**                        | WER이 낮아지고 모델 비교가 쉬움         | 실제 사용 품질과 다를 수 있음 |
| **Orthographic 학습 + Normalised 평가** | 모델은 깔끔한 텍스트 생성, 평가는 공정하게 진행 | 가장 균형적            |


Whisper fine-tuning에서는 보통
정규화된 WER(normalized WER)을 기준 지표로 사용한다.

## Whisper Normalizer 예시

Whisper에는 기본 텍스트 정규화 기능이 포함되어 있다.

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()
normalized = normalizer("Hello, World!")
```

결과

- hello world

이렇게 정규화한 뒤 WER을 계산하면 정규화된 비교가 가능하다.

## Baseline 평가 예제: Whisper vs Common Voice Dhivehi

다음은 Whisper-small 모델을 Dhivehi(디베히) 테스트셋에 적용하여
정규화 전/후 WER을 비교한 예시이다.


| 방식                   | 결과(WER %) | 의미                      |
| -------------------- | --------- | ----------------------- |
| **Orthographic WER** | 167%      | 모델이 거의 파악하지 못함          |
| **Normalised WER**   | 126%      | 정규화 때문에 감소했으나 여전히 성능 낮음 |


이 수치는 fine-tuning 전의 baseline 성능이다.
fine-tuning 후 이 값보다 낮아져야 성공한 것이다.

## End-to-End Whisper 평가 파이프라인 요약

- Whisper 모델 로드 (GPU 사용 시 float16 권장)
- Common Voice 데이터 로드
- KeyDataset을 통해 inference 전용 컬럼 선택
- 배치 단위로 inference 수행
- prediction list 생성
- orthographic WER 계산
- normalizer 적용
- normalised WER 계산
- baseline 기록 -> fine-tuning과 비교

이 프로세스는 ASR 연구, 서비스, 튜닝 모두에서 표준 절차이다.

## 정리

- ASR 평가 지표는 S, I, D 오류에 기반한다.
- WER는 단어 단위, CER는 문자 단위로 평가한다.
- Whisper 평가에서는 정규화된 WER을 기준으로 삼는 것이 일반적이다.
- Orthographic vs Normalised 평가 방식은 결과에 큰 영향을 준다.
- Fine-tuning 목표는 baseline WER을 지속적으로 낮추는 것이다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn