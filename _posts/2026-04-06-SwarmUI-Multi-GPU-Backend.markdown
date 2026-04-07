---
layout: post
title:  "SwarmUI + ComfyUI 멀티 GPU 서버 병렬 연결 가이드"
date:   2026-04-06 15:00:00 +0900
categories: AI_Tools
---

# SwarmUI + ComfyUI 멀티 GPU 서버 병렬 연결 가이드

SwarmUI는 여러 대의 ComfyUI 서버를 백엔드로 동시에 연결해, 작업자 입장에서는 하나의 UI로 여러 GPU를 함께 쓰는 환경을 만들 수 있다.
이 글은 그 개념, 설정 방법, 주의사항을 처음부터 끝까지 정리한다.

---

## 1. 핵심 개념: SwarmUI의 백엔드 분산 방식

SwarmUI는 **단일 접속점(UI) + 다중 백엔드(생성 엔진)** 구조다.

```
[작업자들]
    ↓ 브라우저로 접속
[SwarmUI] ← 큐 관리, UI 제공
    ├── ComfyUI 백엔드 A (서버 1, GPU ×1)
    ├── ComfyUI 백엔드 B (서버 2, GPU ×1)
    └── ComfyUI 백엔드 C (서버 3, GPU ×2)
```

작업자가 이미지를 요청하면 SwarmUI가 **현재 유휴 상태인 백엔드**를 골라 요청을 넘긴다.
백엔드가 많을수록 동시에 처리할 수 있는 작업 수가 늘어난다.

### 분산의 단위: 작업(Job) 단위

중요한 점은 분산 단위가 **작업 단위**라는 것이다.

- 이미지 1장 생성 = 한 백엔드가 전담
- 이미지 10장 배치 = 유휴 백엔드들이 나눠서 처리
- 한 이미지를 여러 GPU가 나눠서 빠르게 처리하는 방식은 아님

즉, **처리량(throughput)**이 늘어나는 구조이지, 단일 작업의 생성 속도가 빨라지는 구조가 아니다.

---

## 2. 지원하는 멀티 백엔드 패턴

SwarmUI는 아래 세 가지 형태를 지원한다.

### 패턴 1: 같은 서버, 여러 GPU

GPU가 여러 장인 서버에서 각 GPU마다 별도의 ComfyUI 인스턴스를 띄우고, 각각을 백엔드로 연결한다.

```
[서버 1대, GPU 4장]
  ├── ComfyUI :8188  (GPU 0)
  ├── ComfyUI :8189  (GPU 1)
  ├── ComfyUI :8190  (GPU 2)
  └── ComfyUI :8191  (GPU 3)
```

각 ComfyUI 인스턴스를 실행할 때 `--cuda-device 0`, `--cuda-device 1` 등으로 GPU를 지정한다.

### 패턴 2: 여러 서버, 각 서버에 ComfyUI

서버가 여러 대 있을 때 각 서버의 ComfyUI를 백엔드로 붙인다. Tailscale 또는 LAN으로 서로 연결되어 있어야 한다.

```
[서버 A] ComfyUI :8188  → SwarmUI 백엔드로 등록
[서버 B] ComfyUI :8188  → SwarmUI 백엔드로 등록
[서버 C] ComfyUI :8188  → SwarmUI 백엔드로 등록
```

### 패턴 3: 다른 SwarmUI 인스턴스를 백엔드로

원격 서버에 SwarmUI까지 함께 설치된 경우, `Swarm-API-Backend` 타입으로 붙인다. ComfyUI만 있는 경우와 달리 SwarmUI 레이어가 한 번 더 들어가는 구성이다.

---

## 3. 설정 방법

### 3-1. 각 ComfyUI 서버 준비

모든 ComfyUI 인스턴스에 아래 두 가지를 반드시 확인한다.

**① Swarm 전용 커스텀 노드 설치**

```
ComfyUI/custom_nodes/
  ├── SwarmComfyCommon
  └── SwarmComfyExtra
```

이 두 폴더는 SwarmUI 패키지 내 아래 경로에 있다.

```
[SwarmUI 경로]/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/
```

ComfyUI `custom_nodes`에 복사하고 재시작한다.

**② 리슨 주소 설정**

SwarmUI가 외부에서 접근할 수 있도록 `--listen 0.0.0.0`을 실행 인수에 추가한다.

StabilityMatrix 기준:
1. ComfyUI 패키지 옆 `⋮` 클릭
2. `Launch Options` 진입
3. `--listen 0.0.0.0` 추가

**③ 포트 중복 방지 (같은 서버에서 여러 인스턴스 운영 시)**

같은 서버에서 여러 ComfyUI를 돌릴 때는 포트를 다르게 지정한다.

```
ComfyUI 인스턴스 1: --port 8188 --cuda-device 0
ComfyUI 인스턴스 2: --port 8189 --cuda-device 1
ComfyUI 인스턴스 3: --port 8190 --cuda-device 2
```

### 3-2. SwarmUI에서 백엔드 추가

1. `Server` → `Backends` 진입
2. `Add Backend` 클릭
3. 타입: `ComfyUI API By URL` 선택
4. 주소 입력

```
# 같은 서버, 포트가 다른 경우
http://127.0.0.1:8188
http://127.0.0.1:8189

# 다른 서버 (LAN)
http://192.168.1.101:8188
http://192.168.1.102:8188

# 다른 서버 (Tailscale)
http://100.x.x.2:8188
http://100.x.x.3:8188
```

5. 저장 후 상태가 녹색(`Connected`)인지 확인
6. 각 백엔드마다 반복

### 3-3. 백엔드 등록 완료 기준

아래를 모두 통과해야 정상이다.

- 모든 백엔드 상태가 녹색
- 각 백엔드에서 테스트 생성 1건 성공
- `Logs`에 missing node / missing model 오류 없음
- 큐 2건 이상 넣었을 때 두 백엔드가 동시에 처리되는 것이 보임

---

## 4. 반드시 지켜야 하는 운영 규칙

### 4-1. 모델 경로 통일

멀티 백엔드 환경에서 가장 흔한 실패 원인이다.

SwarmUI는 작업 요청 시 모델 이름을 백엔드에 그대로 넘긴다. 백엔드(ComfyUI)가 해당 이름의 파일을 찾지 못하면 생성이 실패한다.

**모든 서버가 아래 조건을 만족해야 한다.**

| 조건 | 설명 |
|------|------|
| 모델 파일명 동일 | `sd_xl_base_1.0.safetensors` ← 모든 서버에서 동일 |
| 폴더 구조 동일 | `Models/checkpoints/`, `Models/loras/` 등 |
| ModelRoot 경로 동일하게 설정 | SwarmUI의 `Server Configuration` 기준 |

좋은 예:

```
[서버 A] D:/AI/Models/checkpoints/sdxl_base.safetensors
[서버 B] D:/AI/Models/checkpoints/sdxl_base.safetensors
→ 파일명, 경로 구조 모두 동일 → 정상 작동
```

나쁜 예:

```
[서버 A] D:/AI/Models/checkpoints/sdxl_base.safetensors
[서버 B] E:/comfy/models/checkpoints/SDXL_latest.safetensors
→ 서버 A에서는 되고 서버 B에서는 실패
→ 어디서 실패했는지 찾기 어려움
```

### 4-2. Swarm 커스텀 노드 버전 통일

모든 ComfyUI 인스턴스의 `SwarmComfyCommon`, `SwarmComfyExtra` 버전이 달라지면 워크플로 호환 문제가 생길 수 있다. 같은 SwarmUI 패키지에서 복사한 것을 쓰는 것이 원칙이다.

### 4-3. 큐 배분 동작 이해

- 큐가 하나씩 들어오면 백엔드 1개만 일하고 나머지는 유휴 상태다.
- 동시 요청이 백엔드 수보다 많아야 실질적인 분산이 이루어진다.
- 백엔드 순서는 `Backends` 목록 순서에 영향을 받는다. 더 빠른 서버를 위에 두는 것이 일반적으로 유리하다.

---

## 5. 같은 서버에서 멀티 GPU 운영 시 추가 설정

### GPU 지정 방법

ComfyUI 실행 시 `--cuda-device` 인수로 사용할 GPU 번호를 지정한다.

```bash
# GPU 0번 사용
python main.py --port 8188 --listen 0.0.0.0 --cuda-device 0

# GPU 1번 사용
python main.py --port 8189 --listen 0.0.0.0 --cuda-device 1
```

StabilityMatrix로 관리하는 경우, 같은 ComfyUI를 여러 개 인스턴스로 관리하기 어렵기 때문에 **직접 실행 스크립트**를 별도로 구성하거나, 서버당 GPU 1개 체계로 분리하는 편이 관리가 쉽다.

### VRAM 충돌 방지

같은 서버의 여러 GPU가 공유 메모리 영역을 잘못 참조하면 충돌이 날 수 있다. `--cuda-device`로 명시하면 대부분 방지된다.

---

## 6. Tailscale 기반 멀티 서버 운영 시 주의사항

서버가 물리적으로 다른 장소에 있을 경우 Tailscale로 연결하는 것이 안전하다.

- SwarmUI가 설치된 메인 서버와 추가 ComfyUI 서버 모두 같은 Tailscale 네트워크(tailnet)에 있어야 한다.
- 메인 SwarmUI 서버에서 추가 ComfyUI 서버의 Tailscale IP로 연결한다.

```
http://100.x.x.2:8188   ← 서버 B의 Tailscale IP
http://100.x.x.3:8188   ← 서버 C의 Tailscale IP
```

- 각 서버의 방화벽에서 Tailscale 대역(`100.64.0.0/10`)으로 `8188` 포트를 허용한다.

```powershell
New-NetFirewallRule -DisplayName "ComfyUI-Tailscale" -Direction Inbound -Protocol TCP -LocalPort 8188 -Profile Private -RemoteAddress 100.64.0.0/10 -Action Allow
```

---

## 7. 한계와 오해하기 쉬운 점

| 항목 | 실제 동작 |
|------|-----------|
| 단일 이미지 생성 속도 | 빨라지지 않음. 한 이미지는 한 백엔드가 전담 |
| 동시 처리 가능 수 | 백엔드 수만큼 늘어남 |
| 모델 자동 동기화 | 없음. 수동으로 모든 서버에 동일하게 맞춰야 함 |
| 백엔드 자동 복구 | 없음. 백엔드가 죽으면 해당 큐 항목은 실패함 |
| 작업 재시도 | 없음. 실패한 작업은 수동 재요청 필요 |
| 부하 기반 스마트 배분 | 없음. 큐 순서 기반으로 유휴 백엔드에 배정 |

---

## 8. 운영 점검표

멀티 백엔드 배포 전에 아래를 확인한다.

- 모든 서버의 모델 파일명과 폴더 구조가 동일한가
- 모든 ComfyUI에 `SwarmComfyCommon`, `SwarmComfyExtra`가 설치되어 있는가
- 모든 ComfyUI가 `--listen 0.0.0.0`으로 실행 중인가
- 모든 백엔드가 SwarmUI에서 녹색인가
- 큐 2건 이상을 동시에 넣어 실제 분산이 확인되는가
- 각 백엔드에서 단독 테스트 생성이 성공하는가
- 방화벽 규칙이 필요한 IP 범위만 허용하는가

---

## 마치며

SwarmUI의 멀티 백엔드 구성은 설정 자체는 단순하지만, **모델 경로 통일**이라는 운영 규칙을 지키지 않으면 바로 실패한다.
초기 구성 시 모델 저장 구조부터 정리해두면 이후 서버를 늘릴 때도 추가만 하면 되는 구조가 된다.

- **적합한 상황**: 동시에 여러 작업자가 요청을 넣는 팀 환경, 배치 작업 처리
- **적합하지 않은 상황**: 단일 이미지의 생성 속도를 높이고 싶은 경우
