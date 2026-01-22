---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – Decoder 완전 해부 (latent z에서 waveform으로)"
date:   2026-01-22 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 – Decoder 완전 해부 (latent z에서 waveform으로)

## 0. 목표

이전 글에서는 사실을 확인했다.

- 추론 시 Decoder의 입력은 오직 latent z
- mel은 직접 입력되지 않는다
- mel은 loss를 통해 Decoder를 교정할 뿐이다

이번 글에서는 Decoder 하나만을 끝까지 추적한다.

```
z (latent)
 → upsampling
 → residual blocks
 → waveform
```

그리고 다음 질문에 명확히 답한다.

> **“이 Decoder는 왜 vocoder처럼 동작하는가?”**

## 1. Decoder의 정체: “Vocoder 역할을 하는 생성기”

Style-Bert-VITS2의 Decoder는 구조적으로 보면:

- 입력: latent z (시간축 포함)
- 출력: raw waveform

즉, 멜 기반 vocoder와 기능적으로 동일하지만,
입력이 mel이 아니라 latent z라는 점이 다르다.

그래서 이 Decoder를 이렇게 이해하면 정확하다.

> **“latent space에서 작동하는 neural vocoder”**

## 2. Decoder 입력/출력 인터페이스

### 입력


| 항목                | shape                   |
| ----------------- | ----------------------- |
| z                 | `[B, z_dim, T_z]`       |
| g (speaker/style) | `[B, g_dim, 1]` (있을 경우) |


- T_z는 mel의 시간 길이와 동일
- z는 시간축을 가진 시퀀스

### 출력


| 항목       | shape            |
| -------- | ---------------- |
| waveform | `[B, 1, T_wave]` |


여기서 핵심 질문:

> 왜 T_wave >> T_z 인가?

## 3. 시간 해상도 차이의 본질

앞에서 본 사실을 다시 쓰면:

- z: 프레임 단위 (hop 단위)
- waveform: 샘플 단위

예시:

- hop = 512
- T_z = 258

그렇다면 최종 파형 길이는 대략:

```
T_wave ≈ T_z × hop ≈ 258 × 512 ≈ 132096 samples
```

즉, Decoder의 가장 중요한 역할은:

> **“프레임 단위 정보를 샘플 단위로 확장”**

## 4. Decoder의 핵심 구조 개요

Decoder는 다음 3단 블록으로 이해하면 된다.

```
[Input Conv]
 → [Upsample Block × N]
 → [Output Conv]
```

각 Upsample Block 내부에는:

- Transposed Convolution (또는 nearest + conv)
- Residual Blocks (HiFi-GAN 스타일)
- Non-linearity (LeakyReLU)

이 구조는 HiFi-GAN 계열 vocoder와 거의 동일하다.

## 5. Upsampling: T_z -> T_wave의 실제 메커니즘

### 업샘플링 비율

Decoder config에는 보통 다음이 정의된다.

- upsample_rates: 예를 들어 [8, 8, 2, 2]
- 총 업샘플 배수 = 8 × 8 × 2 × 2 = 256

이 값은 hop_length와 맞춰 설계된다.

예시:

- hop_length = 256
- 총 upsample = 256

> 프레임 1개 -> 샘플 256개

### shape 변화 예시

초기:

```
z: [1, z_dim, 258]
```

Upsample 1 (×8):

```
[1, C1, 2064]
```

Upsample 2 (×8):

```
[1, C2, 16512]
```

Upsample 3 (×2):

```
[1, C3, 33024]
```

Upsample 4 (×2):

```
[1, C4, 66048]
```

이후 Output Conv에서 채널을 1로 줄이며:

```
waveform: [1, 1, ~132096]
```

(패딩/stride 영향으로 ±몇 샘플 차이는 있을 수 있다)

## 6. Residual Block의 역할 (왜 필요한가)

Upsampling만 하면:

- 계단 현상
- 고주파 손실
- metallic noise

가 발생한다.

그래서 각 Upsample Block마다:

- 여러 개의 residual conv
- 서로 다른 dilation

을 사용한다.

이 구조의 목적은 **“여러 시간 스케일의 패턴을 동시에 학습”**하는 것이다.

- 작은 dilation -> 고주파 디테일
- 큰 dilation -> 장기 구조

## 7. 왜 mel을 Decoder에 직접 넣지 않는가 (재확인)

Decoder는 mel을 직접 보지 않는다.

대신:

- mel은 PosteriorEncoder를 통해 z를 만든다
- Decoder는 z만 보고 waveform을 만든다

이 설계의 장점:

1. 훈련/추론 경로 통일
2. mel 계산 비용 제거(추론)
3. latent 공간에서 스타일/화자 제어 가능

## 8. mel loss는 Decoder를 어떻게 교정하는가

훈련 중에는 다음 비교가 이루어진다.

```
Decoder(z_post) → waveform_pred
GT waveform     → mel_processing → mel_gt
waveform_pred   → mel_processing → mel_pred
```

그리고:

```
L_mel = |mel_pred - mel_gt|
```

즉, Decoder는:

- mel space에서 정답과 같아지도록
- 역전파를 통해 조정된다

Decoder는 mel을 직접 보지 않지만,
mel loss를 통해 “mel을 맞추도록” 학습된다.

## 9. Decoder 출력의 성질

Decoder 출력은:

- tanh 또는 clamp로 제한
- 범위: [-1, 1]
- PCM waveform으로 바로 저장 가능

후처리:

- de-emphasis x (대부분 사용 안 함)
- loudness normalization o (선택)

## 10. 예시 z 1개 완주 요약

```yaml
z: [1, z_dim, 258]
 → Upsample ×256
 → Residual Blocks
 → waveform: [1, 1, ~132k]
```

이 파형이:

- inference 결과 음성
- 훈련 중에는 mel loss 계산 대상

이 된다.

## 11. 요약

- Decoder = latent vocoder
- 입력: z (프레임 단위)
- 출력: waveform (샘플 단위)
- Upsample rate = hop_length와 정확히 대응
- mel은 직접 입력 X, loss로만 사용

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2