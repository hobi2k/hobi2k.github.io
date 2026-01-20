---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – 데이터셋과 파일리스트"
date:   2026-01-17 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 - 데이터셋과 파일리스트: 학습이 성립하는 최소 조건

## 1. 이 편의 목적과 범위

이번 편에서는 다음 질문에 코드 기준으로 답한다.

- Style-Bert-VITS2는 어떤 형태의 데이터를 기대하는가?
- wav 파일과 텍스트는 어떤 규칙으로 연결되는가?
- “데이터는 있는데 학습이 안 되는” 대부분의 문제는 어디서 발생하는가?
- 이후 전처리(텍스트/G2P/멜)로 넘어가기 위한 입력 인터페이스는 무엇인가?

이 편의 범위는 “학습이 시작되기 직전”까지이다.
(멜 스펙토그램, G2P, BERT, 모델 구조는 다음 편부터 다룬다.)

## 2. 데이터셋의 정체: SBV2는 “wav + 메타데이터” 모델이다

Style-Bert-VITS2는 end-to-end TTS처럼 보이지만, 실제로는 다음 3요소가 완전히 분리되어 있다.

- 원본 음성 파일 (.wav)
- 메타데이터 파일(filelist)
- 전처리 결과(phoneme, tone, mel 등)

즉, 학습은 “wav 디렉토리”를 직접 읽지 않는다.
오직 filelist를 통해서만 데이터가 들어온다.

wav 파일이 아무리 많아도, filelist에 없으면 존재하지 않는 데이터다.

## 3. 권장 데이터 디렉토리 구조 (원본 레포 기준)

litagin02 레포는 데이터 구조를 강제하지 않지만,
실제 스크립트들이 기대하는 사실상의 표준 구조는 다음과 같다.

```
dataset_root/
├─ wavs/
│  ├─ 00001.wav
│  ├─ 00002.wav
│  └─ ...
├─ filelists/
│  ├─ train.list
│  ├─ val.list
│  └─ test.list   (선택)
└─ configs/
   └─ config.json (또는 yaml)
```

중요한 점:

- wav 파일명은 자유지만, filelist와 반드시 일치해야 한다.
- wav는 하위 디렉토리 없이 평면 구조가 가장 안전하다.
- 다화자라도 wav 폴더는 하나로 두는 것이 디버깅이 쉽다.

## 4. filelist의 한 줄 스키마

Style-Bert-VITS2의 학습은 filelist 한 줄의 파싱에서 시작된다.

기본 형식

```
<wav_path>|<speaker>|<language>|<text>
```

예시 (일본어, 단일 화자)

```
wavs/00001.wav|saya|jp|なんとなく、今日は静かな朝だと思った。
wavs/00002.wav|saya|jp|少しだけ、胸がざわついている。
```

각 필드의 의미


| 필드         | 의미        | 비고                   |
| ---------- | --------- | -------------------- |
| `wav_path` | wav 파일 경로 | **filelist 기준 상대경로** |
| `speaker`  | 화자 ID     | 단일 화자라도 필수           |
| `language` | 언어 코드     | `jp`, `en`, `zh` 등   |
| `text`     | 정규화 전 텍스트 | 이후 클리너/G2P 대상        |


## 5. 언어 코드(language)의 함정

### 코드에서 기대하는 값

Style-Bert-VITS2 내부에는 언어 enum/매핑이 존재하며,
소문자/대문자 불일치는 그대로 에러로 이어진다.

매핑 기준

- jp — 일본어
- en — 영어
- zh — 중국어

언어 처리는 filelist 값이 100% 기준이다.

## 6. wav 파일 요구 조건 (실제 실패 사례 기준)

### 필수 조건
- PCM wav
- 모노 채널
- 샘플레이트:
  - config에 정의된 값과 완전히 동일해야 함 (보통 22050 / 24000 / 44100 중 하나)

### 강력 권장

- RMS 볼륨 정규화 (너무 작은 음성은 학습 불안정)
- 무음이 앞뒤로 과도하게 길지 않을 것
- 클리핑(peak 0dB 초과) 금지

### 자주 터지는 에러 예시

- 스테레오 wav -> shape mismatch
- 샘플레이트 불일치 -> mel 생성 시 assert 실패
- wav 파일은 있는데 path 오타 -> “파일 없음”이 아니라 NoneType 에러

## 7. train / val 분리 기준

filelist는 보통 다음 두 개로 나눈다.

```
filelists/train.list
filelists/val.list
```

### 권장 비율

- 데이터 1~2천 문장: val 50~100개
- 그 이상: 전체의 5% 내외

val을 비워두면 안 된다.
(loss 로그, best checkpoint 판단이 깨진다)

## 8. “데이터는 있는데 학습이 안 된다” Top 7 원인

실제 Style-Bert-VITS2 사용자 기준, 빈도 순 정리.

1. filelist 경로가 config와 불일치
2. wav_path가 filelist 기준 상대경로가 아님
3. 언어 코드 오타 (JP vs jp)
4. wav 샘플레이트 불일치
5. 텍스트에 제어문자/깨진 인코딩 포함
6. speaker id가 config에 정의되지 않음
7. train/val 중 하나가 비어 있음

9. 이 단계의 “입출력 정의”

**입력**

- wav 파일들
- train.list, val.list

**출력**

- **“전처리가 가능한 상태”**의 데이터

이 시점에서는 모델도, 멜도, phoneme도 없다.
오직 데이터 인터페이스를 통과할 준비만 끝난 상태다.

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2