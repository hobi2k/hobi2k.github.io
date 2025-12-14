---
layout: post
title:  "Style-Bert-VITS2 학습기 1"
date:   2025-12-10 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 아키텍처 해부

- 목표

“Style-Bert-VITS2가 어떤 입력을 받아 내부에서 무슨 텐서 변환을 거쳐 최종적으로 waveform을 뽑는지”를 코드 읽을 수 있는 수준으로 정리한다.

- 초점

BERT 문맥 특징 + Style Vector + (tone/language) + VITS2 백본이 어떻게 결합되는가.

## 왜 Style-Bert-VITS2가 ‘캐릭터 일본어 음성’에 강한가

일본어 TTS의 난점은 단순 발음(phoneme)보다 아래가 더 크게 작동한다.

- 억양/악센트(고저)
- 문맥 기반 발화 리듬(끊김/늘임/주저)
- 감정/스타일(차분/불안/기쁨/속삭임 등)

Style-Bert-VITS2는 이걸 정면으로 공략한다.

- 텍스트를 “토큰 임베딩”으로만 쓰지 않고, BERT 문맥 임베딩을 같이 넣는다.
- “화자(speaker)”만이 아니라 **스타일(style)**도 별도의 벡터로 주입한다.
- 백본은 VITS2 계열이라 inference가 빠르고 end-to-end(별도 vocoder 모델 분리 없이)로 음성을 생성한다.

즉, LLM이 만든 대사를 ‘말투’로 읽게 만들기에 구조적으로 잘 맞는다.

## 전체 파이프라인 한 장 요약

아키텍처를 크게 나누면 4파트다.

- TextEncoder (조건 입력을 전부 합쳐 Transformer로 인코딩)
- Alignment/Duration (텍스트 길이 -> 프레임 길이 정렬)
- Latent Modeling (PosteriorEncoder + Flow로 분포 정합)
- Generator (latent -> waveform 생성)

### 데이터 흐름 다이어그램

```
[Text tokens] ─┐
[Tone IDs]   ──┼──> (sum) --> [Transformer TextEncoder] --> prior stats (m, logs)
[Lang IDs]   ──┤
[BERT feats] ─Conv┤
[Style vec]  ─Linear┘

             + [Duration Predictor / SDP]
             + [Monotonic Alignment Search (MAS)]

Training only:
[GT spectrogram/audio features] --> [PosteriorEncoder] --> z_q
z_q <--> [Flow] <--> z_p (from prior)

Inference:
z_p ~ N(m, exp(logs)) --> Flow(reverse) --> z --> Generator --> waveform
```

## 입력(Condition) 설계가 핵심이다

Style-Bert-VITS2에서 “스타일 제어”는 후단에서 갑자기 생기는 게 아니라 TextEncoder 들어가기 전에 이미 결정된다.

TextEncoder에 들어가는 조건은 보통 다음 범주다.

- x: text tokens (심볼 ID)
- tone: tone/accent IDs (일본어 억양용)
- language: 언어 ID (단일 언어여도 코드 구조상 남아있을 수 있음)
- bert: BERT 문맥 특징 (보통 1024차원 계열)
- style_vec: 스타일 벡터 (256차원)
- g: (옵션) speaker embedding / global conditioning (gin_channels)

### “다 더해서(sum) Transformer로 보낸다”의 의미

이 구조는 중요한 성질을 만든다.

- 조건들이 “분기(branch)”로 따로 처리되는 게 아니라,
- **같은 시퀀스 표현 공간(hidden_channels)**에서 결합됨
- 결과적으로 Transformer가 “문맥 + 억양 + 스타일”을 동시에 참고하며 텍스트 표현을 만든다.

실무적으로는:

- LLM이 만든 문장에 포함된 말버릇/문장부호/반복이
-> BERT 문맥 특징과 결합되어
-> “주저/끊김” 같은 리듬으로 반영될 가능성이 커진다.

## TextEncoder 내부: 가장 중요한 모듈

TextEncoder는 크게 다음을 수행한다.

1. 텍스트 토큰 임베딩
2. tone/language 임베딩
3. BERT 특징을 Conv1d로 투영(projection)
4. Style vector를 Linear로 투영
5. 합(sum)
6. Transformer Encoder 통과
7. prior 분포 파라미터(m, logs) 생성

### 왜 BERT에 Conv1d를 쓰나?

BERT 특징은 보통 다음 형태로 들어온다(개념적으로):

- bert: (B, T_text, D_bert) 예: D_bert=1024

VITS 계열 TextEncoder의 내부 채널은 대개 hidden_channels (예: 192/256/…).
그래서 차원 맞추기가 필요하다.

- Conv1d는 보통 “시간축(T_text)을 유지하며 채널을 변환”하기 좋다.
- (B, D_bert, T_text)로 transpose한 뒤 Conv1d(D_bert -> hidden) 같은 식으로 투영한다.

즉, 목적은 간단하다.

“BERT 문맥 특징을 VITS TextEncoder의 히든 차원으로 사상(mapping)”

### Style vector가 Linear인 이유

Style vector는 보통:

- style_vec: (B, D_style) 예: D_style=256

이건 시퀀스가 아니라 “전역 조건”이다.
따라서 다음처럼 쓰기 쉽다.

- Linear(D_style -> hidden)으로 바꾼 뒤
- 시퀀스 길이 T_text에 broadcast해서 더한다.

## Alignment / Duration: VITS가 “잘 읽게” 만드는 핵심

TTS에서 텍스트 길이와 오디오 프레임 길이는 다르다.

- 텍스트: 토큰 수 T_text
- 오디오(혹은 mel frame): 프레임 수 T_mel (훨씬 김)

그래서 “이 토큰이 몇 프레임만큼 발화되나”를 알아야 한다.

Style-Bert-VITS2/VITS2 계열은 보통 아래 중 하나 또는 혼합을 쓴다.

- Duration Predictor: 토큰별 길이 예측
- Stochastic Duration Predictor(SDP): 확률적으로 길이를 모델링(표현력↑)
- MAS(Monotonic Alignment Search): 학습 초기에 강제로 단조 정렬 경로를 찾아 안정화

### MAS가 왜 필요한가?

일반 attention 기반 TTS는 alignment가 흔들리면

- 반복 발화
- 건너뛰기
- 무한 루프(오토리그레시브에서)

같은 문제가 생긴다.

VITS 계열은 오토리그레시브가 아니라도 초기 정렬이 불안정하면 학습이 깨질 수 있다.

MAS는 간단히 말해:

“텍스트는 순서대로 읽힌다(단조). 그 제약을 걸고 최적의 정렬 경로를 찾자.”

이 덕분에 “초기 발화 안정성”이 올라간다.

## Latent Modeling: PosteriorEncoder + Flow가 하는 일

여기부터는 VITS의 핵심 철학이다.

*오디오는 엄청 복잡하니, latent z로 표현하고,*
*text 조건부로 z의 분포를 맞추자.*

이를 위해 학습 시 두 경로가 존재한다.

## 학습(Training) 경로: “정답 오디오에서 z를 만든다”

- enc_q(PosteriorEncoder)가 정답 오디오(멜/특징)를 받아 latent z_q를 만든다.

즉,

- 오디오 -> z_q

이 z_q는 “정답 음성이 가진 정보”를 담는다.

### 텍스트 조건(prior) 경로: “텍스트에서 z의 분포를 예측한다”

TextEncoder 출력에서 prior 분포 파라미터를 만든다.

- m: 평균
- logs: 로그 스케일(표준편차의 로그 등)

즉,

- 텍스트 -> (m, logs) -> z_p ~ N(m, exp(logs))

### Flow의 역할: “z의 분포를 정합(matching)”

Flow는 가역변환(invertible transform)이다.

- z_q를 flow로 변환해 z_p와 분포를 맞추거나,
- 추론 시에는 z_p를 flow 역변환해 생성에 적합한 z로 만든다.

정리하면:

- Training

z_q(정답에서 추출) <-> Flow <-> z_p(텍스트 prior)
둘이 잘 맞도록 학습한다.

- Inference

z_p를 샘플링 -> Flow(reverse) -> z -> Generator -> waveform

여기서 중요 포인트는 추론 때는 enc_q(PosteriorEncoder)를 쓰지 않는다는 것.
(정답 오디오가 없으니까)

## Generator: latent z로 waveform을 뽑는 곳

Generator는 보통 HiFi-GAN 류의 업샘플링 + ResBlock 구조로 구성된다.

- 입력: z (프레임 단위 latent)
- 출력: waveform (샘플 단위)

구현 세부는 프로젝트마다 다르지만 개념은 동일하다.

*“저해상도 시간축 표현(z)을 업샘플링하며 실제 음파로 변환”*

## Style Vector는 어디서 오나?

Style-Bert-VITS2가 캐릭터 TTS에 강한 이유는 style vector를 운영자가 설계할 수 있기 때문이다.

일반적인 운영 방식은:

- 어떤 음성 샘플(또는 샘플 묶음)에서 style embedding을 추출하고
- 그 평균(혹은 대표 벡터)을 style vector로 저장
- 추론 시 원하는 style vector를 선택해 주입

### 캐릭터 TTS에서의 추천 설계

캐릭터는 감정이 일정하지 않고, 특정 상태가 반복된다면 다음처럼 스타일을 분리하는 것이 좋다.

- neutral/ : 기본 톤
- shy/ : 주저, 소심
- unsteady/ : 불안, 흔들림
- whisper/ : 속삭임
- angry/ : 폭발

이 구조로 style vec을 만들면, LLM에서 다음 같은 프로토콜로 제어할 수 있다.

```python
{
  "text": "……ねえ。今日さ、ちょっとだけ話してもいい？",
  "style_id": "unsteady",
  "style_weight": 0.8,
  "speed": 1.0
}
```

즉 LLM이 ‘대본’뿐 아니라 ‘연출 파라미터’까지 출력하도록 설계 가능해진다.

## 학습 vs 추론: 텐서 레벨로 다시 정리

### Training 시그널 흐름(개념)

1. TextEncoder:
입력(텍스트/톤/언어/BERT/style) -> 출력(hidden, m, logs)

2. Alignment:
MAS/Duration으로 텍스트 토큰을 오디오 프레임으로 확장

3. PosteriorEncoder:
정답 오디오 -> z_q

4. Flow:
z_q <-> z_p 정합

5. Generator:
z -> waveform 생성

6. Loss:
재구성/적대적(있는 경우)/정렬/kl 등

### Inference 흐름(개념)

1. TextEncoder(조건 포함) -> (m, logs)
2. z_p 샘플링
3. Flow(reverse)로 z 생성
4. Generator -> waveform

추론에선 정답 오디오 경로(enc_q)가 없다는 점이 핵심이다.

## “파인튜닝”을 설계할 때 어떤 레버를 만질까

### 가장 안전한 전략

- 기존 일본어 checkpoint를 베이스로
- 우선은 다음을 “최소 변경”으로 fine-tune

권장 우선순위:

1. speaker 관련(있다면 g / gin_channels 계열)
2. Generator(마지막 음색을 결정)
3. TextEncoder는 부분 동결(Freeze) 고려
4. Style 관련 projection은 학습 여지 남김

이렇게 하면:

1. 일본어 발화 안정성을 유지하면서
2. 음색/캐릭터성을 덧입히는 방향으로 수렴한다.

### 데이터 측면에서 정말 중요한 것

- 30~60분이라도 “연기 톤”이 일관되면 강하다.
- 문장 길이 다양성(짧은 독백 ~ 긴 문장)
- 숨/웃음/울먹임 등 비언어가 많으면 난이도 올라감(

## LLM 통합을 위한 “제어 설계”

캐릭터 AI는 “텍스트만 잘 쓰면 된다”가 아니라, 텍스트와 음성 제어를 분리하면 성능과 유지보수성이 크게 올라간다.

### LLM이 출력해야 하는 것(권장 스키마)

- text: 실제 발화 대사
- style_id: 스타일 선택
- style_weight: 스타일 강도
- speed: 말 빠르기
- (가능하면) pause_policy: 쉼표/말줄임 처리 규칙

LLM 프롬프트에서 “출력 포맷”을 강제하면 된다.

## Glossary

- VITS/VITS2: 변분(Variational) + Flow + GAN 계열을 합친 end-to-end TTS 계열
- Posterior Encoder(enc_q): 정답 오디오에서 latent를 뽑는 인코더(학습에만 사용)
- Prior(enc_p): 텍스트에서 latent 분포를 예측하는 경로(TextEncoder가 핵심)
- Flow: 가역변환으로 분포를 맞추는 모듈(추론에서 reverse 사용)
- MAS: 단조 정렬 경로를 찾는 알고리즘(학습 안정화)
- Style Vector: 발화 스타일을 결정하는 전역 조건 벡터(256-d 등)
- BERT feature conditioning: 문맥 임베딩을 TTS 조건으로 넣어 발화 자연성을 올리는 기법