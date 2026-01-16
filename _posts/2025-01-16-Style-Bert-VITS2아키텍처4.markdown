---
layout: post
title:  "SBV2 Core (Custom) – 실사용 기반 재구성 로드맵"
date:   2026-01-16 00:10:22 +0900
categories: Style-Bert-VITS2
---

# SBV2 Core (Custom) – 실사용 기반 재구성 로드맵

본 문서는 Style-Bert-VITS2를
“범용 TTS 프레임워크”가 아닌
개인 RP 모델 프로젝트의 음성 엔진 코어로 사용하기 위해,

- 실제로 사용한 파일
- 의도적으로 우회·제거한 경로
- 중요하다고 판단해 고정한 지점

만을 기준으로 다시 구성한 로드맵이다.

이 문서의 목적은 “개선”이나 “최적화”가 아니다.
재현 가능한 이해와 재구성이다.

## 0. 로드맵의 전제

**SBV2 사용 방식**

- WebUI는 사용하지 않을 것이므로 원래 리포지토리에서 제거
- 자동 ASR / Dataset 생성 기능은 사용하지 않음
- 입력 데이터는 대상 캐릭터의 발화로 정제
- 감정은 학습 대상이 아니라 주입 조건
- 스타일은 감정 프리셋이 아니라 정신 상태 파라미터

즉, SBV2는 “데이터를 만들어주는 툴”이 아니라
“조건부 음성 합성 엔진”이다.

## 1. 기준 디렉토리

내가 기준으로 삼은 실질적 구조는 다음이다.

```
sbv2_core/
├─ style_bert_vits2/        # 엔진 본체 (절대 수정 최소)
│
├─ wavs/                    # 이미 정제된 Saya 발화
│
├─ preprocess_text.py       # 텍스트 정규화 (LLM 출력 대응)
├─ preprocess_all.py        # 전체 전처리 오케스트레이터
├─ nlp_preprocessing.py     # 일본어 중심 규칙
├─ data_utils.py            # filelist / metadata 실체
├─ resample.py              # 선택적 (보통 최초 1회)
├─ slice.py                 # 긴 wav 분할 (선택적)
│
├─ bert_gen.py              # BERT feature 명시적 생성
├─ gen_yaml.py              # 실험/학습용 설정 자동 생성
│
├─ style-gen.py             # 스타일 벡터 생성 (정신 상태)
├─ make_style_vectors_cli.py
│
├─ train_ms_jp_extra.py     # 실제 사용한 학습 엔트리
│
├─ run_infer.py             # 단발 테스트
├─ sbv2_worker.py           # LLM 파이프라인 연결용 워커 커스터마이즈
│
├─ convert_onnx.py          # 배포용 ONNX 변환
├─ convert_bert_onnx.py
│
├─ config.py
├─ default_config.yml
├─ default_style.py
│
└─ requirements.txt
```

## 2. 학습 단계 원칙

- 학습/분석 단계에서 금지한 것
    - 파일명 변경 안 하기
    - 디렉토리 이동 안 하기
    - config 통합 안 하기
    - 변수명 리팩터링 안 하기
    - 기능 추가 안 하기

- 이유

> “이해 이전의 개선은 반드시 잘못된다.”

특히 SBV2는

- 학습 경로
- 추론 경로
- 전처리 경로

가 의도적으로 중복·비대칭으로 설계돼 있어서
리팩터링하면 원래 의도가 사라진다.

## 3. 실제 코드 읽기 순서 (해당 프로젝트 기준)
### STEP A. “학습은 어디서 시작되는가”
- train_ms_jp_extra.py
    - 이 파일이 사실상의 main() 이다.
    - 여기서 반드시 확인해야 할 것:
        - config 로딩 순서
        - filelist / metadata 참조 방식
        - Dataset / DataLoader 생성 위치
        - style_bert_vits2 내부로 진입하는 최초 지점
        - style / bert feature가 언제 합류하는지

이 파일을 이해 못 하면
SBV2가 이해되지 않는다.

- train_ms.py
    - 비교용
    - “JP Extra가 빠지면 구조가 얼마나 단순해지는가” 확인용

### STEP B. 데이터 전처리 (WebUI 대체 경로)

이건 “학습 보조”가 아니라
RP 모델 파이프라인과 SBV2를 잇는 접합부다.

- preprocess_all.py
    - 전처리의 중심
    - wav / text / meta를 훈련 가능한 단위로 고정

- preprocess_text.py
    - 일본어 정규화
    - LLM 출력과 충돌하지 않게 하기 위한 규칙 집합

- nlp_preprocessing.py
    - 문장 분해 / 기호 처리 / 읽기 안정성

- data_utils.py
    - filelist의 실체
    - “훈련에 들어가는 데이터 1샘플이 정확히 무엇인가”

- resample.py, slice.py
    - 선택적
    - 항상 쓰는 게 아니라, 필요할 때만

*핵심 질문*

“SBV2가 요구하는 최소 입력 형식은 무엇인가?”

### STEP C. BERT / Style는 ‘학습 대상’이 아니다
- bert_gen.py
    - BERT feature를 명시적으로 생성
    - “내부 자동 처리”를 신뢰하지 않기 위한 선택

- style-gen.py
    - 감정이 아님
    - 정신 상태 벡터 생성기

**여기서의 핵심 이해**

- style은 “출력의 성격”이 아니라 모델에 주입되는 조건 벡터다.

### STEP D. 엔진 코어
- style_bert_vits2/models/models_jp_extra.py
- SynthesizerTrn
    - 모든 것이 여기로 수렴
    - 이 단계의 목표는 단 하나다.
        - SynthesizerTrn.forward()를 데이터 흐름 기준으로 설명할 수 있는가
        - 텍스트 -> 어디서 phoneme이 되는가
        - BERT -> 언제 결합되는가
        - style → 어디서 곱/concat 되는가
        - duration / alignment -> 어디서 학습되는가


## 4. 이해 이후 허용 개조

- config 로딩 경로 명시화 (자동 merge 제거)
- preprocess 계열을 Saya 파이프라인 규칙에 맞게 고정
- sbv2_worker.py 중심의 추론 인터페이스 정리
- 최종 모델만 ONNX 변환

## 4. 실제 우리가 따르던 실행 순서
1. resample (필요 시)
2. slice (필요 시)
3. wav 기준 utt 작성 (외부)
4. preprocess_text
5. preprocess_all
6. bert_gen
7. style-gen
8. train_ms_jp_extra


## 6. 핵심

- sbv2_core는 툴이 아니다
- Saya 음성 엔진의 연구·해체·재조립용 코어
- WebUI는 구조적으로 배제된다
- 이해 없이는 리팩터링도 없다