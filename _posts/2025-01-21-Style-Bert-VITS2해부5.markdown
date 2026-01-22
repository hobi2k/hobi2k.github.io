---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – PosteriorEncoder · latent z · Flow 완전 해부"
date:   2026-01-20 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 – PosteriorEncoder · latent z · Flow 완전 해부

## 0. 목표

이제까지는 다음을 숫자까지 고정했다.

- mel의 shape

```
mel: [n_mels, T_spec]
```

- T_spec ≈ floor(T / hop_length)
- mel은 훈련 전용 신호이며, 추론에는 사용되지 않는다

이번 편에서는 이 mel 하나가 다음 경로를 거쳐가는 것을 끝까지 추적한다.

```
mel
 → PosteriorEncoder
 → z_posterior
 → Flow
 → z_prior space
 → KL loss
```

그리고 마지막에 훈련 vs 추론 경로가 왜 달라지는지를 코드/개념 양쪽에서 확정한다.

## 1. VITS 구조에서 “Posterior”라는 말의 의미

Style-Bert-VITS2는 이름 그대로 VITS 계열이다.
VITS의 핵심은 다음 한 문장으로 요약된다.

> **“음성은 확률 분포에서 샘플링된다.”**

이를 위해 VITS는 두 개의 분포를 만든다.


| 분포            | 무엇으로부터 만들어지나 |
| ------------- | ------------ |
| **Posterior** | 실제 음성 (mel)  |
| **Prior**     | 텍스트 조건       |


훈련 중에는 둘 다 존재하지만,
추론 시에는 posterior가 사라진다.

이 공백을 메우기 위해 등장하는 것이 Flow다.

## 2. PosteriorEncoder는 무엇을 하는 모듈인가

### 담당 위치

- PosteriorEncoder는 mel -> latent z 변환기다.
- 이 z는 “실제 음성을 가장 잘 설명하는 잠재 표현”이다.

### 입력 / 출력


| 항목        | shape                 |
| --------- | --------------------- |
| 입력 mel    | `[B, n_mels, T_spec]` |
| 출력 z_post | `[B, z_dim, T_z]`     |


여기서 중요한 질문:

왜 T_z = T_spec 인가?
-> PosteriorEncoder는 시간축을 유지한다.
즉, mel 프레임 하나당 latent 하나가 대응된다.

## 3. PosteriorEncoder 내부 개념 (구조 요약)

PosteriorEncoder는 개념적으로 다음을 수행한다.

```
mel(t)
 → Conv / Norm / Activation
 → (μ(t), log σ(t))
 → z_post(t) = μ(t) + ε · σ(t)
```

즉,

- mel을 보고
- 각 시간 프레임마다
- 가우시안 분포의 파라미터(μ, σ) 를 예측하고
- 거기서 샘플링한다

이게 바로 variational inference다.

## 4. 예시 mel 1개로 z_posterior까지 추적

- mel shape

```
mel = [128, 258]
```

배치 차원을 붙이면:

```
mel = [1, 128, 258]
```

PosteriorEncoder 통과 후:

```
z_post = [1, z_dim, 258]
```

여기서 z_dim은 보통:

- 192
- 256

중 하나다 (config에 정의).

중요한 점:

**시간 길이 258은 그대로 유지된다.**

## 5. TextEncoder와 “prior z”의 본질적 한계

TextEncoder는 다음 입력만 가진다.

- phoneme ids
- tone
- BERT features
- language id

즉, **실제 음성의 미세한 변동(잡음, 화자 미세차, 발화 습관)**을
알 방법이 없다.

그래서 TextEncoder는 완전한 z를 만들 수 없다.
대신, 다음을 만든다.

```
z_prior ~ p(z | text)
```

즉, 조건부 확률 분포다.

## 6. Flow는 왜 필요한가

문제 상황을 정리하면 이렇다.

- PosteriorEncoder는

```
z_post ~ q(z | mel)
```

TextEncoder는

```
z_prior ~ p(z | text)
```

이 둘의 분포는 다르다.

그런데 추론 시에는:

- mel이 없다
- z_post를 만들 수 없다
- z_prior에서 샘플링해야 한다

따라서 두 분포를 같은 공간으로 맞춰야 한다.
이 역할을 하는 것이 Normalizing Flow다.

## 7. Flow의 정확한 역할

Flow는 다음 성질을 가진 함수다.

- invertible (역함수 존재)
- Jacobian determinant 계산 가능

훈련 중:

```
z_post
 → Flow
 → z_flow
```

이 z_flow가 prior 분포와 최대한 비슷해지도록 학습된다.

즉,

```
Flow(q(z | mel)) ≈ p(z | text)
```

## 8. KL loss는 정확히 무엇을 맞추는가

KL loss는 다음을 최소화한다.

```
KL( Flow(z_post) || z_prior )
```

의미:

- “실제 음성에서 온 z를
- 텍스트에서 예측한 분포로
- 옮겼을 때
- 얼마나 다른가?”

이걸 시간축 전체(T_z)에 대해 계산한다.

## 9. 훈련 vs 추론 경로 (완전 비교)
### 훈련 시

```
text → TextEncoder → z_prior
mel  → PosteriorEncoder → z_post
z_post → Flow → z_flow
KL(z_flow || z_prior)
Decoder(z_post)
```

### 추론 시

```
text → TextEncoder → z_prior
sample(z_prior)
Flow^{-1}
Decoder(z)
```

PosteriorEncoder는 추론에서 완전히 사라진다.

## 10. 왜 mel을 Decoder에 직접 넣지 않는가

가끔 이런 질문이 나온다.

> “mel을 그냥 Decoder에 넣으면 안 되나?”

답은 안 된다다.

이유:

- mel은 훈련 시에만 존재
- 추론 시에는 텍스트만 있음
- 동일한 입력 인터페이스를 유지해야 함

그래서 mel은:

- 음성을 설명하는 정답 신호
- latent를 학습시키는 교사

역할만 수행하고,
최종 합성에는 직접 관여하지 않는다.

## 11. 요약

- PosteriorEncoder = mel -> z_post
- TextEncoder = text -> z_prior
- Flow = 두 분포를 연결하는 가역 변환
- KL loss = 분포 정렬 비용
- mel은 훈련 전용
- 추론은 prior에서만 시작

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2