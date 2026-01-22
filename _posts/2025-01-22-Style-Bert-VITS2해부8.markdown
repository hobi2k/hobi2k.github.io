---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – Inference 완전 해부 (텍스트 -> 음성 생성의 실제 경로)"
date:   2026-01-22 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 – Inference 완전 해부 (텍스트 -> 음성 생성의 실제 경로)

## 0. 목표

이제 남은 질문은 하나다.

“학습이 끝난 Style-Bert-VITS2 모델은
텍스트를 받아 실제 음성을 어떻게 만들어내는가?”

이번 편에서는 훈련 코드와 추론 코드의 차이,
사라지는 모듈,
실제로 조절 가능한 파라미터를 전부 정리한다.

## 1. Inference 파이프라인 한 장 요약

훈련 파이프라인과 비교해서 보면 추론은 매우 단순하다.

### 훈련

```
text → TextEncoder → z_prior
mel  → PosteriorEncoder → z_post
z_post → Flow → z_flow
Decoder(z_post)
(loss 계산)
```

### 추론 (Inference)

```
text → TextEncoder → z_prior
sample(z_prior)
Flow^{-1}
Decoder(z)
→ waveform
```

PosteriorEncoder, mel, loss 전부 사라진다.

## 2. 추론의 진짜 시작점: 어디서 호출되나?
핵심 엔트리

- infer.py
- 또는 models/*.py 내부의 infer() 메서드
- WebUI를 쓰면 app.py

이 중 무엇을 쓰든 내부 흐름은 동일하다.

### 3. Step 1 — 텍스트 처리 (훈련과 100% 동일)

이 단계에서 변하면 안 되는 것

- text cleaner
- G2P
- phoneme vocab
- tone 처리
- BERT 처리 방식

즉, 이전에 설명한 파이프라인을 그대로 재사용한다.

입력:

```
"なんとなく、今日は静かな朝だと思った。"
```

출력 텐서:

```
phones, tones, lang_ids, bert_features, lengths, x_mask
```

여기서 하나라도 달라지면:

- 발음 붕괴
- 억양 이상
- 학습과 추론 불일치

### 4. Step 2 — TextEncoder -> z_prior

TextEncoder는 다음을 만든다.

```
z_prior ~ p(z | text)
```

구체적으로는:

- 평균 μ_prior
- 분산 σ_prior

즉, 정규분포 파라미터다.

shape:

```
μ_prior, σ_prior : [B, z_dim, T_z]

```

### 5. Step 3 — Sampling: “무작위성”의 유일한 근원

여기서 추론의 성격이 결정된다.

#### 기본 샘플링

```
z_sample = μ_prior + ε · σ_prior
```

ε ~ N(0, I)

#### temperature 개념

실제로는:

```
z_sample = μ_prior + temperature · ε · σ_prior
```


| temperature  | 효과         |
| ------------ | ---------- |
| 낮음 (0.3~0.6) | 안정적, 반복적   |
| 1.0          | 기본         |
| 높음 (>1.0)    | 다양성 ↑, 불안정 |


Inference에서 사실상 가장 중요한 노브

### 6. Step 4 — Flow 역변환 (Flow⁻¹)

훈련 때는:

```
z_post → Flow → z_flow
```

추론 때는 반대 방향이다.

```
z_sample → Flow^{-1} → z
```

이때:

- Flow는 가역(invertible) 구조
- Jacobian 계산 필요 없음 (추론 시)

이 z가 Decoder의 실제 입력이다.

### 7. Step 5 — Decoder: latent -> waveform

이 단계는 이전 글과 동일하다.

```
z: [B, z_dim, T_z]
 → Upsample (× hop_length)
 → Residual Blocks
 → waveform: [B, 1, T_wave]
```

Decoder는:

- mel을 모른다
- text를 모른다
- z만 보고 음성을 만든다

## 8. Inference 결과의 후처리

### 일반적인 후처리

- clamp / tanh -> [-1, 1]
- PCM 변환
- wav 저장

### 선택적 후처리

- loudness normalization
- silence trimming
- fade-in/out

de-emphasis는 대부분 사용하지 않는다.

## 9. Inference에서 조절 가능한 핵심 파라미터

### (1) temperature (가장 중요)

- 다양성 vs 안정성 조절
- 캐릭터 음성은 낮게 유지하는 게 일반적

### (2) speaking rate / length scale

일부 구현에서는:

```
duration = duration / length_scale
```


| length_scale | 효과  |
| ------------ | --- |
| < 1.0        | 빠르게 |
| 1.0          | 기본  |
| > 1.0        | 느리게 |


### (3) noise scale

- σ_prior에 곱해지는 계수
- temperature와 유사하지만 분리된 경우도 있음

## 10. “추론 음성이 이상할 때” 디버깅 순서

- KL loss 확인
- temperature 너무 높지 않은지
- G2P / text cleaner 훈련과 동일한지
- duration scale 확인
- Flow weight 로딩 정상인지

## 11. 훈련 vs 추론 차이 한 장 정리


| 항목               | 훈련 | 추론      |
| ---------------- | -- | ------- |
| PosteriorEncoder | O  | X       |
| mel              | O  | X       |
| TextEncoder      | O  | O       |
| Flow             | O  | O (역방향) |
| Decoder          | O  | O       |
| loss             | O  | X       |


## 12. Inference의 본질 요약

> Inference란
> **“텍스트 조건으로 정의된 확률 분포에서**
> **음성을 하나 샘플링하는 과정”**이다.

Style-Bert-VITS2는:

- deterministic TTS X
- probabilistic TTS O

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2