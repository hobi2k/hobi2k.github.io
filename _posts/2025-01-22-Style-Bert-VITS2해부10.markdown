---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – ONNX 변환과 배포 파이프라인 완전 정리"
date:   2026-01-22 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 – ONNX 변환과 배포 파이프라인 완전 정리

## 1. convert_onnx.py의 기능

이 스크립트의 목적은 단순하다.

> **훈련된 Style-Bert-VITS2 모델을**
> **“학습 기능이 제거된, 실행 전용 음성 생성 엔진”으로 만드는 것**

즉,

- 연구용 PyTorch 모델이 아니다.
- 훈련 재개 가능 모델이 아니다.

대신,

- wav를 뱉는 기계
- 추론 전용 엔진
- Python 없이도 실행 가능한 모델

을 만드는 것이 목적이다.

## 2. 입력으로 요구되는 파일 구조

코드에서 강제하는 입력은 다음 세 개다.

```python
config_path = model_path.parent / "config.json"
style_vec_path = model_path.parent / "style_vectors.npy"

assert model_path.exists()
assert config_path.exists()
assert style_vec_path.exists()
```

즉, ONNX 변환의 최소 단위는:

```
xxx.safetensors
config.json
style_vectors.npy
```

중요한 해석

- style_vectors.npy는 있으면 좋은 부가 자산이 아니다
- ONNX 변환 단계에서부터 필수 구성요소

## 3. TTSModel은 무엇을 로드하는가

```python
tts_model = TTSModel(
    model_path=model_path,
    config_path=config_path,
    style_vec_path=style_vec_path,
    device=device,
)
tts_model.load()
```

여기서 로드되는 것:

- 모델 가중치 (safetensors)
- 하이퍼파라미터 (config.json)
- 스타일 벡터 테이블 (style_vectors.npy)
- 스타일 이름 -> id 매핑 (style2id)

즉, 스타일 벡터는 이미 모델 내부 상태로 로드된다.

## 4. ONNX에 들어가는 모델은 “net_g” 하나뿐이다

```python
assert tts_model.net_g is not None
```

export 대상:

```python
torch.onnx.export(
    model=tts_model.net_g,
    ...
)
```

여기서의 핵심

- Discriminator X
- PosteriorEncoder X
- mel_processing X
- loss X

오직 net_g만 ONNX로 간다.

## 5. 그런데 그냥 net_g.forward를 export하지 않는다

이 스크립트의 가장 중요한 부분이다.

### JP-Extra 모델의 경우

```
def forward_jp_extra(...):
    return tts_model.net_g.infer(...)

tts_model.net_g.forward = forward_jp_extra
```

### Non-JP-Extra 모델의 경우

```
def forward_non_jp_extra(...):
    return tts_model.net_g.infer(...)

tts_model.net_g.forward = forward_non_jp_extra
```

### 이게 의미하는 것

- 훈련용 forward는 버린다
- infer() 경로만 남긴다
- ONNX는 오직 추론 경로만 포함한다

즉, **ONNX 모델 = “infer()만 남은 SynthesizerTrn”**

## 6. infer() 안에는 무엇이 들어 있는가 (개념적 구조)

infer() 내부에는 일반적으로 다음이 포함된다.

1. TextEncoder
2. duration / alignment 로직
3. Flow (latent 변환)
4. Decoder (waveform 생성)
5. noise / length / style conditioning

> 즉, **ONNX 하나에 “텍스트 -> 음성” 전 경로가 들어간다**

## 7. 스타일 벡터는 정확히 어떻게 들어가는가

### 7.1 스타일 선택

```python
if DEFAULT_STYLE in tts_model.style2id:
    style_id = tts_model.style2id[DEFAULT_STYLE]
else:
    style_id = 0
```

### 7.2 스타일 벡터 추출

```python
style_vector = tts_model.get_style_vector(
    style_id,
    DEFAULT_STYLE_WEIGHT
)
```

### 7.3 텐서화

```python
style_vec_tensor = torch.from_numpy(style_vector).to(device).unsqueeze(0)
```

### 7.4 ONNX 입력으로 등록

```python
input_names=[
    ...,
    "style_vec",
    ...
]
```

### 결론

- 스타일 벡터는 ONNX 그래프 내부 상수가 아니다
- ONNX 입력으로 명시적으로 노출
- 즉, 런타임에 교체 가능

스타일 벡터는
“외부 파일 -> 모델 로드 시 내부 상태 -> ONNX 입력으로 주입”
이라는 3단계 구조를 가진다.

## 8. 샘플링은 ONNX 안에 있는가?

있다.

```python
noise_scale = torch.tensor(0.667)
noise_scale_w = torch.tensor(0.8)
length_scale = torch.tensor(1.0)
sdp_ratio = torch.tensor(0.0)
```

그리고 전부 입력이다.

```python
input_names=[
  ...,
  "length_scale",
  "sdp_ratio",
  "noise_scale",
  "noise_scale_w",
]
```

해석

- 난수 생성 로직은 infer() 내부
- 하지만 강도 조절은 외부 입력
- 즉, 확률 경로는 ONNX 안에 있지만 제어는 밖에서

## 9. ONNX 입력 시그니처 (JP-Extra 기준 완전 정리)


| 입력 이름         | 의미             |
| ------------- | -------------- |
| x_tst         | phoneme id 시퀀스 |
| x_tst_lengths | 텍스트 길이         |
| sid           | speaker id     |
| tones         | 억양             |
| language      | 언어 id          |
| bert          | 일본어 BERT 특징    |
| style_vec     | 스타일 벡터         |
| length_scale  | 발화 길이          |
| sdp_ratio     | duration 조절    |
| noise_scale   | 음색 노이즈         |
| noise_scale_w | 억양 노이즈         |


Dynamic axis:

```python
"x_tst": {0:"batch", 1:"length"},
"tones": {0:"batch", 1:"length"},
"language": {0:"batch", 1:"length"},
"bert": {0:"batch", 2:"length"},
"style_vec": {0:"batch"},
```

## 10. 왜 mel / PosteriorEncoder는 완전히 사라지는가

이 ONNX는 wav를 생성하는 기계다.

- mel은 “정답 비교용 신호”
- PosteriorEncoder는 “정답 latent 생성기”
- 둘 다 훈련에서만 필요

따라서:

> ONNX = Generator only

## 11. 이 ONNX 모델의 본질

이 모델은:

- 학습 X
- 파인튜닝 X
- loss 계산 X

대신,

- 텍스트 입력
- 스타일 입력
- 노이즈/길이 입력
-> wav 출력만 한다.

## 12. 이 구조가 실무에서 강력한 이유

- Python 필요 없음
- PyTorch 필요 없음
- onnxruntime 하나로 실행 가능
- 게임 / 앱 / 서버 / 로컬 어디든 동일 동작
- 스타일 교체 가능
- 감정 제어 가능

> 즉, 연구 모델에서 제품 엔진으로 바뀌는 지점

13. 최종 요약

convert_onnx.py가 만드는 ONNX 모델은
“스타일과 샘플링을 외부 입력으로 받는,
infer 전용 완성형 음성 생성 엔진”이다.

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2