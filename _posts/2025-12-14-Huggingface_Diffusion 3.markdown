---
layout: post
title:  "허깅페이스 디퓨전 코스 - Unit 4"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_diffusion
---

# Diffusion Course Unit 4

DDIM Inversion과 Diffusion for Audio의 상세 코드는 추후에 정리한다.

## Unit 4가 왜 필요한가: “Diffusion = 느리다/제어가 어렵다/도메인이 늘어난다”

Stable Diffusion까지 오면 이미 다음 한계에 부딪힌다.

- 샘플링이 느림: step을 많이 돌려야 품질이 나온다.
- 훈련이 빡셈: 학습 안정화/효율을 위해 수많은 트릭이 필요하다.
- 제어가 어렵다: “프롬프트만으로 원하는 편집/제약을 정확히 걸기”가 힘들다.
- 이미지 밖으로 확장된다: 오디오/비디오 등 다른 모달리티에도 diffusion을 적용하고 싶다.

Unit 4는 이 문제들을 각각 별개의 축으로 나눠 “어떤 연구 라인이 있는지”를 정리한다.

## Faster Sampling via Distillation

### 용어 정의

- **Sampling(샘플링)**: 생성 시점에 `t=T → 0`로 내려오며 반복적으로 노이즈를 제거해 이미지를 만드는 과정(= inference 루프).
- **Distillation(증류)**: 큰/느린 모델(teacher)의 동작을 작은/빠른 모델(student)이 모사하도록 학습해, inference 비용을 줄이는 기법.
- **Progressive distillation(점진적 증류)**: “2 step을 1 step으로” 줄이는 학습을 반복해서, 최종적으로 4~8 step 같은 초저스텝 생성이 가능해지는 계열.

### Progressive Distillation의 핵심 메커니즘
Unit 4는 다음 아이디어를 명확히 설명한다.

- teacher 모델은 (예: DDPM/SD)에서 **두 번의 sampling step**을 수행한다.
- student 모델은 teacher가 “두 step 돌린 결과”를 **한 step**으로 맞추도록 학습한다.
- 이를 반복하면 step 수가 점점 줄어든다(예: 1000 -> 500 -> 250 -> … 8).

### “Guided Distillation” 확장: CFG까지 학생이 한 번에 흉내내게 하기

Unit 4는 “teacher가 classifier-free guidance(CFG)를 사용하면” 학생이 그 효과까지 포함해서 한 step으로 내도록 만들 수 있다고 설명한다.

- teacher: CFG로 유도된 결과(더 고품질/더 prompt-정합)
- student: 추가 입력으로 “원하는 guidance scale” 같은 정보를 받아, 한 번의 평가로 teacher와 동일한 효과를 내도록 학습 

**파이프라인 설계 관점 결론**
- “샘플링 loop를 30~50 step 도는 구조” 자체를 바꿔서, 모바일/실시간 UI에서 쓸 수 있는 latency로 내리는 연구 축.
- 캐릭터 시스템에서 “감정 상태 변화에 따른 이미지 갱신”을 자주 돌릴수록, **distillation 계열의 ROI가 커진다.**

## Training Improvements (훈련 개선 트릭 지도)
Unit 4는 “훈련이 잘 안 되거나 비효율적인 문제”를 개선하는 연구 방향을 여러 갈래로 정리한다.
아래는 각각의 아이디어가 실제로 무엇을 의미하는지(개념 설명) + 어떤 레버인지(엔지니어링 관점)이다.

### Noise schedule / loss weighting / trajectory 튜닝

#### 용어 정의
- **Noise schedule**: timestep에 따라 얼마나 노이즈를 섞고(훈련), 얼마나 크게 업데이트할지(샘플링)를 결정하는 규칙(β/α 계열).
- **Loss weighting**: 모든 timestep의 학습 난이도가 같지 않기 때문에, 특정 구간에 가중치를 주어 학습을 안정화/가속화하는 전략.
- **Sampling trajectories**: 샘플링 시 어떤 경로(예: Euler/LMS/DDIM 등)로 내려올지, step 배치를 어떻게 할지.

**파이프라인 관점**
- “내 모델이 특정 해상도/도메인에서 뭉개진다” 같은 문제가 생기면,
  1) UNet 아키텍처를 뜯기 전에
  2) schedule/weighting/trajectory부터 의심해볼 가치가 있다.

### Diverse aspect ratios로 학습

#### 왜 생기는 문제인가?
많은 모델이 정사각형(512×512 등)에 과적합되면,
- 긴 세로/가로 비율에서 구도가 깨지거나,
- 특정 비율에서 디테일이 급격히 붕괴한다.

**실전 결론**
- 캐릭터 시스템에서 “썸네일/배너/스탠딩 CG”처럼 출력 비율이 다양하면, 이 축이 중요해진다.

### Cascaded diffusion (저해상도 생성 + super-res 확장)

#### 용어 정
- **Cascaded diffusion**: 1단계 모델은 저해상도에서 큰 구조를 만들고,
  2단계(혹은 그 이상) 모델이 super-resolution을 수행해 고해상도를 만든다.

**설계 관점**
- 단일 모델로 1024 이상을 직접 뽑는 것보다,
- “구조->디테일”을 단계별로 분리하면 학습/추론 비용을 제어하기 쉽다.

### Better conditioning: richer text embeddings / multi-conditioning

Unit 4는 conditioning 개선의 예로,
- Imagen이 T5 같은 큰 언어 모델을 사용한다는 점,
- eDiffi 같은 multi-conditioning 방향을 언급한다.

#### 용어 정의(짧게)
- **Conditioning(조건)**: 모델이 “무작위 생성”이 아니라, 텍스트/클래스/마스크/깊이 등 입력 조건을 따르도록 하는 입력.
- **Rich embeddings**: 더 표현력이 큰 텍스트 표현을 써서, 이미지-텍스트 정합을 강화하려는 방향.
- **Multi-conditioning**: 텍스트 + 박스/세그/스케치/포즈 등 여러 조건을 동시에 사용.

**캐릭터 파이프라인 관점**
- “LLM이 만든 감정/상태를 이미지로 반영”하려면, 결국 conditioning 신호 설계 문제로 귀결된다.
- prompt만이 아니라, “상태 벡터->조건 임베딩”으로 확장하는 설계 여지가 생긴다.

### Knowledge Enhancement (지식 강화)
Unit 4는 “captioning/object detection 모델을 훈련 과정에 끼워 넣어 더 좋은 캡션/지식으로 성능을 올리는 방향”을 언급한다. 

#### 용어 정의(짧게)
- **Image captioning**: 이미지->텍스트 설명 생성 모델.
- **Object detection**: 이미지 내 객체 위치/클래스를 예측하는 모델.
- **Knowledge enhancement**: 데이터 라벨(텍스트 캡션)을 더 정보량 있게 만들거나, 학습 신호를 보강하는 전략.

**실무적 의미**
- “데이터 품질이 모델 상한을 결정한다”는 명제를 diffusion에서도 그대로 적용하는 접근.
- 특히 도메인 데이터(캐릭터 CG/일러스트)를 직접 구축할수록 중요해진다.

### Mixture of Denoising Experts (MoDE)
Unit 4는 “노이즈 레벨별로 다른 전문가(experts)를 학습”시키는 MoDE 계열을 소개한다.

#### 용어 정의
- **Mixture of Experts(MoE)**: 여러 “전문가 모델” 중 일부를 선택/조합해 쓰는 구조.
- **MoDE**: denoising(노이즈 제거) 과정을 노이즈 레벨 구간별로 전문가를 나눠 학습해 성능을 올리는 방향.

**설계 관점**
- 전 timestep을 하나의 UNet이 다 책임지게 하면 비효율/충돌이 생길 수 있다.
- “구간별 전문화”는 대형 모델에서 특히 의미가 있다.

## More Control for Generation and Editing (제어/편집을 더 정교하게)

Unit 4는 “학습 개선”이 아니라 “샘플링/추론 단계에서 기능을 추가”하는 연구들을 정리한다.
그리고 편집 방법을 크게 4개 카테고리로 나눈다.

### 카테고리 1: Noise를 추가하고 새 프롬프트로 denoise (img2img 계열)
Unit 4는 img2img의 기본 아이디어를 확장한 연구로 SDEdit, MagicMix 등을 언급한다.

#### 용어 정의
- **img2img**: 입력 이미지를 latent로 바꾼 뒤, 일정 강도(strength)만큼 노이즈를 섞고, 새 프롬프트 조건으로 denoise하여 변환하는 방식.
- **SDEdit**: “노이즈 추가 + denoise”를 편집에 사용한 초기 대표 아이디어.
- **MagicMix**: 더 자연스러운 혼합/편집을 노리는 파생 아이디어.

**중요 포인트**
- 이 범주의 핵심은 “랜덤 노이즈를 얼마나 넣느냐(strength)”가 편집 강도를 결정한다는 것.
- Unit 3에서 배운 구조(encode→add_noise->denoise)가 그대로 재사용된다.

### DDIM Inversion: “랜덤 노이즈를 넣지 말고, 모델의 역과정으로 정확히 올리자”

Unit 4는 DDIM inversion을 “랜덤 노이즈 추가 대신 sampling trajectory를 역으로 되돌리는 방식”으로 설명한다.

#### 용어 정의
- **Inversion(인버전)**: 주어진 실제 이미지(또는 생성 이미지)를 diffusion의 latent/노이즈 상태로 “되돌려”서, 그 상태에서 다시 denoise하며 편집을 수행하는 기술.
- **DDIM inversion**: DDIM의 결정론적(또는 준결정론적) 성질을 활용해, “이미지 -> 특정 latent 경로”를 더 안정적으로 찾는 접근.

**왜 중요한가**
- img2img는 “랜덤 노이즈” 때문에 원본 구조 보존이 불안정할 수 있다.
- inversion은 “원본이 실제로 어디서 왔는지에 가까운 상태”를 찾아 편집 품질/보존성을 올리는 축이다.

### 3.3 Null-text Inversion: CFG의 uncond 임베딩을 step별로 최적화

Unit 4는 Null-text inversion을 “unconditional text embeddings를 step마다 최적화해 편집 품질을 크게 높이는 방식”으로 설명한다.

#### 용어 정의
- **Unconditional embedding (uncond)**: CFG에서 “텍스트 조건이 없는 가지”에 들어가는 텍스트 임베딩(보통 empty prompt).
- **Null-text inversion**: 편집 대상 이미지에 대해, step별 uncond 임베딩을 최적화해서 원본 복원 품질을 극적으로 끌어올리는 접근.

**파이프라인 레벨 의미**
- “CFG의 수식은 그대로인데, uncond 입력을 ‘이미지에 맞게’ 바꾼다.”
- 즉, 모델 가중치를 건드리지 않고도 편집 정합을 끌어올리는 레버다.

### 카테고리 2: (1)을 마스크와 결합 (inpainting 계열)

Unit 4는 Blended Diffusion을 기본 아이디어로, SmartBrush 같은 마스크 기반 정밀 인페인팅을 언급한다.

#### 용어 정의
- **Mask**: 이미지에서 “편집할 영역/보존할 영역”을 지정하는 0/1 지도.
- **Inpainting**: 마스크 영역만 새로 생성하고 나머지는 유지하는 편집.

Unit 3에서 DIY inpainting loop가 “매 step 마스크 합성”이라는 걸 봤다면,
Unit 4의 메시지는 여기서 한 발 더 나간다:
- 마스크 편집 품질을 올리려면, 모델 자체를 그 목적에 맞게 fine-tune 하거나(예: SmartBrush),
- 샘플링 과정에서 더 정교한 blending 규칙이 필요해진다.

### 카테고리 3: Cross-attention Control (Prompt-to-Prompt / paint-with-words)
Unit 4는 “cross-attention을 이용해 편집 위치를 공간적으로 제어”하는 계열을 소개한다.

#### 용어 정의
- **Cross-attention**: 텍스트 토큰과 이미지(또는 latent) 위치 특징을 연결하는 attention.
- **Prompt-to-Prompt**: 프롬프트를 바꿀 때 cross-attention map을 제어/재사용해, 특정 토큰이 특정 공간에 대응되도록 편집을 안정화하는 접근.
- **Paint-with-words**: 단어별로 공간 위치를 지정해, “이 단어는 이 영역에 그려라” 같은 제어를 구현하는 아이디어(예시로 eDiffi 언급).

**캐릭터 시스템에 직결되는 이유**
- “표정만 바꾸고 머리카락은 고정” 같은 요구는 결국 공간 제어 문제다.
- 마스크가 하드한 제어라면, cross-attention control은 “소프트하지만 더 정교한” 제어 축이다.

### 카테고리 4: 단일 이미지에 과적합(fine-tune) 후 편집

Unit 4는 Imagic, UniTune 같은 “한 장에 overfit해서 그 이미지 기반으로 편집”하는 축을 소개한다.

#### 용어 정의
- **Overfit(과적합)**: 특정 데이터(여기서는 1장의 이미지)에 모델을 강하게 맞춰, 그 이미지의 정체성을 매우 잘 보존하도록 만드는 것.
- 이 접근은 “원본 정체성 보존”에 유리하지만, 일반화는 희생한다.

## Video / Audio / Iterative Refinement (모달리티 확장 + 아키텍처 변화)

Unit 4는 이미지 밖으로 확장하는 축과, “diffusion의 정의 자체가 넓어지는 흐름”을 같이 다룬다.

### Video diffusion: 시퀀스를 다루는 아키텍처

Unit 4는 비디오를 “이미지 시퀀스”로 보고 diffusion 아이디어를 적용한다고 설명한다. 그리고 3D UNet 같은 시퀀스 아키텍처가 언급된다.

#### 용어 정의
- **3D UNet**: (H,W)뿐 아니라 시간축(T)까지 포함하는 3차원 컨볼루션/블록으로 시퀀스를 처리.
- 핵심 과제: 프레임 간 일관성(temporal consistency) + 계산 비용.

### Audio diffusion: 오디오를 “스펙트로그램 이미지”로 바꿔서 diffusion 적용
Unit 4는 오디오 diffusion에서 가장 성공적인 접근으로,
원시 파형 대신 **스펙트로그램(spectrogram)**을 2D 이미지처럼 보고 diffusion을 학습하는 방식을 설명한다.

#### 용어 정의
- **Spectrogram(스펙트로그램)**: 시간-주파수 에너지를 2D로 표현한 그림(오디오의 “이미지 표현”).
- 생성 흐름:
  1) 텍스트/조건 -> 스펙트로그램 생성(= diffusion이 하는 일)
  2) 스펙트로그램 -> 오디오로 복원(보코더 등 별도 변환)

Unit 4는 DiffWave, Riffusion, MusicLM, Audio 관련 모델들을 레퍼런스로 제시하며, hands-on으로 “Diffusion for Audio” 노트북을 제공한다.

**캐릭터 파이프라인 관점 메모**
- “이미지와 오디오를 모두 diffusion로” 같은 방향은,
  - 공통 루프(denoise 반복)는 공유하되
  - 표현 공간(latent/스펙트로그램)이 달라진다는 점이 핵심 차이.

### New Architectures and Approaches: diffusion -> iterative refinement로 확장

Unit 4는 “가우시안 노이즈를 더하고 제거하는 좁은 정의를 넘어, 더 일반적인 ‘점진적 정제(iterative refinement)’ 모델군으로 가고 있다”고 정리한다.

#### 용어 정의
- **Iterative refinement(반복 정제)**: 어떤 방식으로든 “대충 망가뜨린(corrupt) 상태”를 여러 번 반복해 되돌리며 샘플을 만드는 생성 패러다임.
- 예: Cold Diffusion(노이즈가 아니라 다른 변환을 역으로 풀기) 같은 아이디어가 여기에 포함된다.

또한 UNet 대신 Transformer 기반(예: DiT)을 사용하는 방향도 언급된다.

#### 용어 정의(짧게)
- **DiT (Diffusion Transformer)**: UNet 대신 Transformer로 denoising 모델을 구성한 계열.
- 의미:
  - “이미지 생성에서도 Transformer가 주류가 될 수 있다”는 아키텍처적 전환 신호.

## Hands-On Notebooks

1) **DDIM Inversion**  
   - inversion을 이용해 기존 diffusion 모델로 이미지 편집을 더 정교하게 수행
2) **Diffusion for Audio**  
   - 스펙트로그램 개념 소개 + 오디오 diffusion 최소 예제 + 특정 장르로 fine-tune

## 추후 예정

1) **DDIM Inversion 노트북을 코드 레벨로 해부**해서,
   - `invert(image) → latents_t`
   - `edit(latents_t, prompt) → edited_image`
   형태의 모듈을 만든다.

2) **Null-text inversion을 캐릭터 일관성 모듈**로 재해석한다.
   - “uncond embedding step별 최적화”를
   - “캐릭터 정체성 고정 레이어”로 시스템에 편입.

3) **Audio diffusion hands-on을 ‘TTS와 경쟁’이 아니라 ‘BGM/효과음 생성’으로 통합**한다.
   - 텍스트->스펙트로그램->오디오 복원 파이프라인으로 설계.

참고자료
Huggingface, Diffusion Course, https://huggingface.co/learn