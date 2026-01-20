---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – STFT / FFT / mel shape 완전 해부 (DSP 집중)"
date:   2026-01-20 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 — STFT / FFT / mel shape 완전 해부 (DSP 집중)

## 0. 목표

다음 파이프라인을 수식/shape/코드 기준으로 완전히 고정한다.

```
waveform [T]
→ (reflect pad)
→ STFT → spec [F, T_spec, 2]
→ magnitude → spec_mag [F, T_spec]
→ mel filter bank → mel [n_mels, T_spec]
→ log compression → log_mel [n_mels, T_spec]
```

## 1. 실제 파라미터는 어디서 오나?

Style-Bert-VITS2는 모델별 config.json의 data 섹션에서 DSP 파라미터를 로드한다. 
예시(공개된 데모 모델 설정)에서는 아래처럼 정의되어 있다.

- sampling_rate: 44100
- filter_length (= n_fft): 2048
- hop_length (= hop_size): 512
- win_length (= win_size): 2048
- n_mel_channels (= n_mels): 128
- mel_fmin: 0.0
- mel_fmax: null(= None)

이 값들은 곧바로 mel_processing.py의 STFT/멜 계산에 들어간다.

## 2. mel_processing.py의 핵심 함수들(책임 분리)

mel_processing.py 구현은 크게 3개 축으로 나뉜다.

1. STFT -> magnitude spectrogram

- spectrogram_torch(...)

2. linear spec -> mel spec + log compression

- spec_to_mel_torch(...)

3. waveform → mel(= 1+2를 합친 convenience)

- mel_spectrogram_torch(...)

이 글에서는 **(3)**을 기준으로 “한 번에” 추적하되, 중간에 (1)(2)로 분해해서 shape를 확인한다.

## 3. STFT에서 shape가 왜 [F, T_spec]가 되나?

### 3.1 프레임 분할

- hop_length만큼 이동하면서 win_length 길이의 창을 씌워 FFT를 한다.
- 결과적으로 시간축은 “샘플 개수 T”가 아니라 “프레임 개수 T_spec”가 된다.

### 3.2 주파수 bin 개수 F

- n_fft로 FFT를 하면 원래는 n_fft개 복소수 bin이 생기지만,
- 입력이 실수(real)라서 스펙트럼이 켤레 대칭이므로 “한쪽”만 쓰면 된다.
- 따라서 onesided일 때 주파수 bin 개수는:

```
F = n_fft // 2 + 1
```

이 구현도 onesided=True로 고정되어 있다.

## 4. STFT 구현을 그대로 따라가며 shape 추적
### 4.1 Hann window 캐싱

코드는 dtype/device/win_size 별로 Hann window를 캐싱한다.

> hann_window[wnsize_dtype_device] = torch.hann_window(win_size)...

의미:

- 동일 GPU/동일 dtype에서 반복 호출 시 window 재생성 비용 절약
- 학습 중 mel을 계속 만들기 때문에 이 최적화는 중요

### 4.2 reflect padding의 정체: “center=False인데 패딩은 한다”

이 구현에서 가장 헷갈리는 부분이 여기다.
코드는 STFT 전에 이렇게 패딩한다.

> pad = int((n_fft - hop_size) / 2)

그리고 reflect로 양쪽을 같은 길이만큼 늘린다.

의미:

- center=False(즉, torch.stft가 내부적으로 중앙정렬 패딩을 하지 않음)인데도
- 외부에서 별도의 패딩을 주는 설계다.
- 결과적으로 “프레임 수 계산”이 일반적인 center=True 케이스와 달라진다.

## 4.3 torch.stft 출력 shape

핵심 호출부는 아래다. (핵심 키워드만 인용)

> spec = torch.stft(..., onesided=True, return_complex=False)

return_complex=False이므로, torch는 마지막 축에 (real, imag)를 붙여 반환한다.

따라서 STFT 직후 shape는:

```
spec: [F, T_spec, 2]
```

- F = n_fft//2 + 1
- T_spec = 프레임 개수

### 4.4 magnitude로 변환: 마지막 축(2)을 제거

구현은 다음 한 줄로 magnitude를 만든다.

> spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

의미:

- spec[..., 0]=real, spec[..., 1]=imag
- real^2 + imag^2 후 sqrt -> magnitude
- 이 시점에 shape는

```
spec_mag: [F, T_spec]
```

## 5. Mel filter bank 적용: shape 변화의 핵심
### 5.1 mel basis 행렬의 shape

librosa.filters.mel(...)로 mel filter bank를 만든다.

> mel = librosa_mel_fn(sr=..., n_fft=..., n_mels=..., fmin=..., fmax=...)

librosa의 mel filter bank는 일반적으로 다음 shape다:

```
mel_basis: [n_mels, F]
```

여기서 F = n_fft//2 + 1.
또한 dtype/device/fmax 기준으로 캐싱한다.

### 5.2 matmul로 주파수 축만 압축

적용은 아래와 같다.

> spec = torch.matmul(mel_basis[...], spec)

shape 연산:

```
mel_basis: [n_mels, F]
spec_mag : [F, T_spec]
----------------------
mel      : [n_mels, T_spec]
```

중요 포인트:

- 시간축 T_spec은 그대로 유지
- 주파수축만 F -> n_mels로 줄어든다

## 6. log compression: “로그 멜”이 되는 지점

이 구현은 “log”를 다음 형태로 수행한다.

> return torch.log(torch.clamp(x, min=clip_val) * C)

의미:

- clip_val로 바닥을 깔아 log 폭발 방지
- C는 compression factor (기본 1)
- 결과 shape는 변하지 않는다.

따라서:

```
log_mel: [n_mels, T_spec]
```

## 7. 여기까지의 “shape 변화 표” (한 장으로 정리)

실제 파라미터를 대입하면(예: n_fft=2048, n_mels=128):

- F = 2048//2 + 1 = 1025

| 단계        | 텐서              | shape                           |
| --------- | --------------- | ------------------------------- |
| 입력        | waveform        | `[T]`                           |
| pad 후     | waveform_padded | `[T + (n_fft - hop)]` (이 구현 기준) |
| STFT      | complex spec    | `[1025, T_spec, 2]`             |
| magnitude | spec_mag        | `[1025, T_spec]`                |
| mel       | mel             | `[128, T_spec]`                 |
| log mel   | log_mel         | `[128, T_spec]`                 |


(패딩 길이와 T_spec 계산은 다음 섹션에서 숫자로 확정한다.)

## 8. 예시 wav 1개로 “T_spec 숫자 계산”

이 구현은 center=False지만, 외부에서 reflect pad를 한다는 점이 포인트다.
패딩 길이:

```
pad = (n_fft - hop) / 2  (양쪽)
총 증가량 = n_fft - hop
```

즉, 원래 waveform 길이가 T라면:

```
T_padded = T + (n_fft - hop)
```

그리고 center=False인 STFT의 프레임 수는 일반적으로:

```
T_spec = 1 + floor((T_padded - n_fft) / hop)
```

대입하면:

```
T_spec = 1 + floor((T + (n_fft - hop) - n_fft) / hop)
       = 1 + floor((T - hop) / hop)
```

즉, 매우 깔끔하게 정리된다:

```
T_spec = 1 + floor((T / hop) - 1)
       = floor(T / hop)
```

단, T가 hop의 배수가 아닐 때는 floor 때문에 1 프레임 차이가 날 수 있다.

### 8.1 실제 숫자로 계산 (3초짜리 wav 예시)

앞에서 확인한 실제 파라미터를 사용한다.

- sampling_rate = 44100
- hop = 512
- wav 길이 = 3.0초

그러면 샘플 수:

```
T = 44100 * 3 = 132300
```

프레임 수:

```
T_spec = floor(132300 / 512)
       = floor(258.3984375)
       = 258
```

따라서 이 예시 wav의 최종 log-mel shape는:

```
log_mel: [n_mels, T_spec] = [128, 258]
```

그리고 STFT magnitude는:

```
spec_mag: [F, T_spec] = [1025, 258]
```

## 9. 가장 중요한 해석 3가지

1. hop_length가 “시간 해상도와 메모리”를 동시에 결정

- hop ↓ → T_spec ↑ → mel/time 길이 ↑ → GPU 메모리/연산량 ↑
- hop ↑ → T_spec ↓ → 정보 손실 가능(특히 prosody)

2. n_fft는 “주파수 해상도(F)”를 결정

- F = n_fft//2 + 1
- n_fft ↑ → F ↑ → mel_basis 계산 비용 ↑ (하지만 mel로 바로 압축)

3. n_mels는 “모델이 보는 스펙트럼 해상도” 자체

- 보통 80/100/128
- Style-Bert-VITS2 계열 데모 설정은 128을 쓴다.

10. 결론: 4편에서 축약했던 것, 여기서 완전히 고정

이제 다음 문장을 “정확한 의미로” 쓸 수 있다.

```
mel: [n_mels, T_spec]
```

- n_mels는 config의 n_mel_channels
- T_spec는 이 구현에서는 거의 floor(T/hop)로 떨어진다(reflect pad + center=False 조합 때문)
- 중간 산출물은
    - STFT: [F, T_spec, 2]
    - magnitude: [F, T_spec]
    - mel/log_mel: [n_mels, T_spec]

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2