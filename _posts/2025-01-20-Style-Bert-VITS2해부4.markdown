---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – 멜 스펙토그램 생성 파이프라인: wav가 모델 입력이 되기까지"
date:   2026-01-20 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 — 멜 스펙토그램 생성 파이프라인: wav가 모델 입력이 되기까지

이번 편의 핵심 질문

> *“Style-Bert-VITS2에서 mel은 언제, 어디서, 어떤 설정으로 만들어지며*
> *모델은 mel을 정확히 어떻게 소비하는가?”*

## 0. 목표

다음 wav 파일 하나가 있다고 해보자.

```
wavs/00001.wav
```

이 글은 이 wav가 다음으로 변하는 전 과정을 본다.

```
wav (float waveform)
 → STFT
 → mel filter bank
 → log-mel
 → normalization
 → mel tensor
 → (PosteriorEncoder 입력)
 → (mel loss 계산)
```

## 1. 멜 생성의 전체 위치

Style-Bert-VITS2에서 멜은 두 군데에서 등장한다.

### 학습 전처리 단계 (offline)

- wav -> mel을 미리 계산해서 저장
- 속도 최적화 목적

### 학습 중/추론 중 (online)

- loss 계산
- 추론 시에는 mel을 만들지 않음

추론은 mel을 입력으로 받지 않는다.
mel은 훈련 전용 신호다.

## 2. 멜 처리 담당 파일

핵심 파일

```
style_bert_vits2/mel_processing.py
```

이 파일이 STFT / mel / log / normalization의 전부를 담당한다.

## 3. 멜 생성의 진입점 (누가 mel_processing을 부르나?)

학습 전처리 엔트리

```
preprocess_all.py
```

역할:

- wav 로드
- 샘플레이트 확인/변환
- mel_processing 호출
- 결과를 .npy 등으로 저장

> 학습 전에 한 번만 실행되는 스크립트

## 4. wav 로딩 단계

### 입력 wav 요구 조건

- PCM wav
- mono
- config에 정의된 샘플레이트와 동일

### 로딩 결과

```python
wav: np.ndarray or torch.Tensor
shape: [T]   # 1차원
range: [-1.0, 1.0]
```

## 5. Step 1 — STFT (시간 -> 주파수)

### mel_processing.py 내부

핵심 파라미터 (config에서 옴):


| 파라미터         | 의미              |
| ------------ | --------------- |
| `n_fft`      | FFT window size |
| `hop_length` | 프레임 간 이동        |
| `win_length` | 실제 윈도우 길이       |
| `window`     | Hann window     |


처리 개념

```
waveform [T]
 → framing
 → FFT
 → magnitude spectrogram
```

결과 텐서

```
spec: [n_fft // 2 + 1, T_spec]
```

## 6. Step 2 — Mel Filter Bank 적용

이유

- 인간 청각에 맞춘 주파수 압축
- 고주파 해상도 ↓, 저주파 해상도 ↑

### mel 파라미터


| 파라미터     | 설명                 |
| -------- | ------------------ |
| `n_mels` | mel bin 개수 (보통 80) |
| `fmin`   | 최소 주파수             |
| `fmax`   | 최대 주파수             |


처리

```
linear spectrogram
 → mel filter bank
 → mel spectrogram
```

결과

```
mel: [n_mels, T_spec]
```

## 7. Step 3 — Log scale

이유

- 에너지 분포를 압축
- 학습 안정성 ↑

처리

```
mel = log(clamp(mel))
```

결과

```
log-mel: [n_mels, T_spec]
```

## 8. Step 4 — Normalization (가장 많이 터지는 지점)

방식

- mel 전체에 대해 평균/표준편차 기반 정규화
- 혹은 고정 상수 기반 스케일링

중요 포인트

- 훈련과 추론에서 반드시 동일한 정규화 정책 사용

정규화가 달라지면:

- loss는 줄어드는데
- 음질은 망가지는 악성 케이스 발생

## 9. 최종 mel 텐서 형태

```python
mel: torch.FloatTensor
shape: [n_mels, T_spec]
```

배치로 묶이면:

```
mel: [B, n_mels, T_spec]
mel_lengths: [B]
```

## 10. mel은 모델 어디로 들어가나?

핵심 개념 정리


| 용도               | mel 사용 여부 |
| ---------------- | --------- |
| TextEncoder      | X         |
| PosteriorEncoder | O         |
| Flow             | O         |
| Decoder          | X         |
| Inference        | X         |


## 11. mel의 실제 소비 위치 (학습)

- PosteriorEncoder

```
mel -> latent z (posterior)
```

이 z는:

- TextEncoder가 만든 prior z와
- KL loss로 맞춰진다

## 12. mel loss는 어디서 계산되나?

계산 대상

- Decoder 출력 wav
- GT wav -> 다시 mel로 변환

loss 종류

- L1 / L2 mel loss
- multi-resolution STFT loss

즉, mel은 “정답 신호”다.

## 13. 예시 wav 1개 전체 요약

```
wavs/00001.wav
 → waveform [T]
 → STFT
 → mel [80, T']
 → log-mel
 → normalized mel
 → PosteriorEncoder 입력
 → mel loss 계산
```

## 14. 자주 터지는 오류 Top 5

- wav 샘플레이트 ≠ config 샘플레이트
- stereo wav
- n_fft < win_length
- fmax > sample_rate / 2
- 정규화 정책 변경 후 기존 체크포인트 재사용

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2