---
layout: post
title:  "StabilityMatrix로 ComfyUI + SwarmUI 설치하고 Tailscale로 원격 접속하기"
date:   2026-04-06 12:00:00 +0900
categories: AI_Tools
---

# StabilityMatrix로 ComfyUI + SwarmUI 설치하고 Tailscale로 원격 접속하기

이 글은 **StabilityMatrix**를 사용해 ComfyUI와 SwarmUI를 설치하고,
SwarmUI용 커스텀 노드를 추가한 뒤, **Tailscale**을 이용해 외부에서 원격으로 접속하는
전체 흐름을 처음부터 끝까지 정리한 가이드다.

핵심 목표:
- StabilityMatrix 하나로 ComfyUI + SwarmUI 환경 구성
- SwarmUI에서 ComfyUI 커스텀 노드 사용 가능하게 설정
- Tailscale로 집 밖에서도 로컬 GPU 서버에 접속

---

## 1. Tailscale이란 무엇인가

본론 전에 Tailscale을 먼저 짚고 간다. 이 도구가 왜 필요한지 이해해야
나머지 설정도 의미가 생긴다.

### 1-1. 개요

**Tailscale**은 WireGuard 기반의 **메시 VPN(Mesh VPN)** 서비스다.
쉽게 말하면, 인터넷에 연결된 여러 기기를 마치 **같은 로컬 네트워크에 있는 것처럼** 연결해주는 도구다.

일반 VPN은 모든 트래픽을 중앙 서버를 거쳐 보낸다.
Tailscale은 다르다. 기기끼리 **직접(P2P)** 연결되기 때문에 속도가 빠르고, 중앙 서버에 의존하지 않는다.

### 1-2. 왜 Tailscale인가

AI 이미지·영상 생성 작업에서 자주 생기는 문제가 있다.

- 집에 GPU 서버가 있는데, 카페나 회사에서도 그 서버에 접속하고 싶다
- 포트 포워딩을 열면 보안 위험이 생긴다
- 공유기 설정을 건드리기 싫다

Tailscale은 이 세 가지를 한 번에 해결한다.

- 포트 포워딩 불필요
- 공유기 설정 변경 불필요
- 기기 인증은 Google/GitHub 계정으로 처리
- 무료 플랜에서 최대 100대 기기 지원

### 1-3. 작동 원리

```
[GPU 서버 (서버)]  <──── Tailscale 가상 네트워크 ────>  [노트북 (외부)]
  100.x.x.1                                             100.x.x.2
```

각 기기에 Tailscale을 설치하면 `100.x.x.x` 형태의 **고정 가상 IP**가 부여된다.
이 IP는 인터넷 어디서든 동일하게 유지된다.
즉, GPU 서버의 Tailscale IP로 접속하면 집 밖에서도 로컬처럼 접근이 가능하다.

### 1-4. 주요 기능 요약

| 기능 | 설명 |
|------|------|
| MagicDNS | IP 대신 기기 이름(예: `my-gpu-server`)으로 접속 가능 |
| Subnet Router | Tailscale이 설치되지 않은 기기도 네트워크에 포함 가능 |
| Exit Node | 특정 기기를 VPN 출구로 사용 (IP 우회) |
| Taildrop | 기기 간 파일 전송 |
| Funnel | Tailscale 네트워크 외부에 서비스 공개 (HTTPS 터널) |
| ACL | 기기별 접근 권한 세밀하게 제어 가능 |

---

## 2. 사전 준비

### 필요한 것

- Windows 10/11 PC (GPU 서버)
- 인터넷 연결
- Google 또는 GitHub 계정 (Tailscale 로그인용)

### 다운로드 목록

1. **StabilityMatrix** - [lykos.ai](https://lykos.ai)에서 다운로드
2. **Tailscale** - [tailscale.com/download](https://tailscale.com/download)에서 다운로드

---

## 3. StabilityMatrix 설치

StabilityMatrix는 ComfyUI, SwarmUI, Automatic1111 등 여러 AI 생성 도구를
**하나의 인터페이스**에서 설치·관리·실행할 수 있는 런처다.

모델, 패키지, 환경을 통합 관리할 수 있어서 개별 설치보다 훨씬 편하다.

### 3-1. 설치 절차

1. [lykos.ai](https://lykos.ai) 접속 후 `Download for Windows` 클릭
2. 다운로드된 `StabilityMatrix-win-x64.exe` 실행
3. 설치 경로 선택 (기본값 권장, 단 경로에 한글·공백 없을 것)
4. 초기 실행 시 데이터 폴더 위치 설정

> **주의**: 설치 경로에 한글이나 공백이 포함되면 패키지 설치 오류가 날 수 있다.
> `C:\StabilityMatrix` 또는 `D:\AI\StabilityMatrix` 같은 경로를 권장한다.

### 3-2. 초기 설정

최초 실행 시 **데이터 폴더(Data Folder)** 위치를 묻는다.

- `Portable Mode`: StabilityMatrix 설치 폴더 안에 모든 데이터 저장 (이동 편의성 높음)
- `Default`: `%APPDATA%` 하위에 저장

모델 파일이 수백 GB에 달할 수 있으므로, **용량이 넉넉한 드라이브에 Portable Mode로 설치**하는 것을 권장한다.

---

## 4. ComfyUI 설치

### 4-1. 패키지 추가

1. StabilityMatrix 실행 후 상단 메뉴 `Packages` 클릭
2. `+ Add Package` 버튼 클릭
3. 목록에서 **ComfyUI** 선택
4. `Install` 클릭

설치 중 PyTorch, CUDA 관련 패키지가 함께 설치된다.
처음 설치라면 수 분이 걸릴 수 있다.

### 4-2. ComfyUI 실행 확인

설치 완료 후 `Launch` 버튼 클릭 → 브라우저에서 `http://127.0.0.1:8188` 접속 확인.
ComfyUI 기본 워크플로우 화면이 뜨면 정상이다.

---

## 5. ComfyUI를 SwarmUI 백엔드로 연결하기

이 단계가 이 가이드의 핵심이다.
StabilityMatrix로 설치한 ComfyUI를 SwarmUI의 **외부 백엔드**로 연결해서,
SwarmUI가 그 ComfyUI를 실제로 구동 엔진으로 사용하도록 설정한다.

### 5-1. ComfyUI를 0.0.0.0으로 실행하기

기본적으로 ComfyUI는 `127.0.0.1`(로컬호스트)에만 바인딩된다.
SwarmUI가 외부 프로세스로서 ComfyUI에 접속하려면, ComfyUI가 **모든 인터페이스에서 요청을 받도록** 설정해야 한다.

StabilityMatrix에서:
1. ComfyUI 패키지 옆 `⋮` (더보기) 클릭
2. `Launch Options` 또는 `Extra Launch Arguments` 진입
3. 다음 인수 추가:

```
--listen 0.0.0.0
```

이후 ComfyUI를 재시작한다. `http://0.0.0.0:8188`로 바인딩된 것을 확인할 수 있다.

### 5-2. SwarmUI 설치

1. StabilityMatrix `Packages` 탭으로 이동
2. `+ Add Package` 클릭
3. **SwarmUI** 선택 후 `Install` 클릭

SwarmUI는 .NET 런타임이 필요하다. 없으면 설치 과정에서 자동으로 안내한다.

### 5-3. SwarmUI 초기 설정

첫 실행 시 설정 마법사가 뜬다.

1. **Backend 선택** 단계에서 `None` 또는 `Skip`을 선택한다.
   - Built-In을 선택하면 SwarmUI가 자체 ComfyUI를 새로 띄운다.
   - 우리는 이미 설치된 ComfyUI를 연결할 것이므로 일단 건너뛴다.
2. 모델 폴더 경로는 StabilityMatrix의 `Data/Models` 경로로 잡으면
   ComfyUI와 SwarmUI가 같은 모델 파일을 공유한다.
3. `Finish Setup` 클릭

> **팁**: StabilityMatrix는 `Models` 폴더를 공유할 수 있다.
> SwarmUI 설정에서 모델 경로를 StabilityMatrix의 `Data/Models` 경로로 잡으면
> ComfyUI와 SwarmUI가 같은 모델 파일을 공유한다.

### 5-4. SwarmUI 백엔드에 ComfyUI 추가

SwarmUI 메인 화면이 뜨면:

1. 상단 메뉴 `Server` → `Backends` 클릭
2. `Add Backend` 버튼 클릭
3. 백엔드 타입 목록에서 **`ComfyUI API (External)`** 선택
4. Address 입력란에 ComfyUI 주소 입력:

```
http://127.0.0.1:8188
```

5. `Save` 클릭

저장 후 잠시 기다리면 백엔드 상태가 **녹색(Connected)**으로 바뀐다.
이 상태가 되면 SwarmUI가 ComfyUI를 통해 이미지를 생성할 수 있다.

### 5-5. SwarmUI 실행 확인

`Launch` 클릭 후 브라우저에서 `http://127.0.0.1:7801` 접속.
SwarmUI 메인 화면이 뜨고, 백엔드 상태가 Connected이면 정상이다.

---

## 6. SwarmUI 필수 커스텀 노드 설치

SwarmUI가 ComfyUI API를 통해 제대로 동작하려면
**SwarmUI 전용 커스텀 노드** 두 가지가 ComfyUI에 반드시 설치되어 있어야 한다.

| 노드 | 역할 |
|------|------|
| **SwarmComfyCommon** | SwarmUI ↔ ComfyUI 간 기본 통신에 필요한 공통 노드 |
| **SwarmComfyExtra** | 고급 기능(이미지 저장 방식, 메타데이터 처리 등) 확장 |

이 두 노드가 없으면 SwarmUI에서 이미지 생성 자체가 안 되거나 오류가 발생한다.

### 6-1. 설치 방법

ComfyUI의 `custom_nodes` 폴더 경로:

```
[StabilityMatrix 데이터 폴더]
└── Packages
    └── ComfyUI
        └── custom_nodes   ← 여기
```

터미널(또는 StabilityMatrix의 Console)에서:

```bash
cd [위 경로의 custom_nodes 폴더]
git clone https://github.com/mcmonkeyprojects/SwarmUI.git swarmui-nodes
```

> 또는 SwarmUI 소스 내의 `src/BuiltinExtensions/ComfyUIBackend/ExtraNodes` 폴더에
> **SwarmComfyCommon**과 **SwarmComfyExtra**가 들어 있다.
> 이 폴더를 직접 `custom_nodes`에 복사해도 된다.

실제로 SwarmUI를 설치하면 아래 경로에 이미 해당 노드 소스가 있다:

```
[StabilityMatrix 데이터 폴더]
└── Packages
    └── SwarmUI
        └── src
            └── BuiltinExtensions
                └── ComfyUIBackend
                    └── ExtraNodes
                        ├── SwarmComfyCommon   ← 이걸
                        └── SwarmComfyExtra    ← 이것도
```

이 두 폴더를 ComfyUI의 `custom_nodes` 폴더에 복사(또는 심볼릭 링크)하면 된다.

### 6-2. 노드 적용 확인

ComfyUI 재시작 후 노드 목록에서 `SwarmSaveImageWS` 등 Swarm 관련 노드가 보이면 정상이다.

### 6-3. ComfyUI Manager 설치 (추가 커스텀 노드 관리용)

추가적인 커스텀 노드를 편하게 관리하려면 **ComfyUI Manager**를 설치한다.

```bash
cd [custom_nodes 경로]
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

재시작 후 ComfyUI 화면 상단에 `Manager` 메뉴가 생긴다.

### 6-4. 자주 쓰는 추가 커스텀 노드

| 노드 | 용도 |
|------|------|
| WAS Node Suite | 이미지 처리, 유틸리티 |
| ComfyUI Impact Pack | 얼굴 복원, 디테일 업스케일 |
| ComfyUI_IPAdapter_plus | IP-Adapter (스타일/캐릭터 이식) |
| ComfyUI-VideoHelperSuite | 영상 생성 워크플로우 지원 |
| ComfyUI-KJNodes | 다양한 유틸리티 노드 |
| rgthree-comfy | 효율적인 노드 그룹 관리 |

---

## 7. Tailscale 설치 및 설정

이 단계에서 GPU 서버(집 PC)와 원격 기기(노트북 등) 양쪽에 Tailscale을 설치한다.

### 7-1. GPU 서버(집 PC)에 Tailscale 설치

1. [tailscale.com/download](https://tailscale.com/download) 접속
2. Windows용 다운로드 후 설치
3. 설치 후 시스템 트레이에 Tailscale 아이콘 생성
4. 아이콘 클릭 → `Log in` → Google 또는 GitHub 계정으로 로그인
5. 브라우저에서 기기 인증 완료

로그인 후 이 PC에 `100.x.x.x` 형태의 Tailscale IP가 부여된다.

### 7-2. 원격 기기(노트북 등)에 Tailscale 설치

같은 방법으로 원격 접속에 사용할 기기에도 Tailscale을 설치하고
**동일한 계정**으로 로그인한다.

같은 계정으로 로그인한 기기들은 자동으로 같은 Tailscale 네트워크(tailnet)에 묶인다.

### 7-3. GPU 서버 IP 확인

GPU 서버에서 Tailscale 트레이 아이콘 클릭 → `This device` 항목에서 IP 확인.

또는 PowerShell에서:

```powershell
tailscale ip -4
```

예: `100.64.0.5`

### 7-4. MagicDNS 활성화 (선택, 편의 기능)

Tailscale 관리 콘솔([login.tailscale.com/admin](https://login.tailscale.com/admin))에서 **MagicDNS**를 활성화하면
IP 대신 기기 이름으로 접속할 수 있다.

예: `http://100.64.0.5:7801` 대신 `http://my-gpu-server:7801`

---

## 8. SwarmUI / ComfyUI 원격 접속 설정

기본적으로 SwarmUI와 ComfyUI는 `127.0.0.1`(로컬호스트)에만 바인딩된다.
외부(Tailscale 포함)에서 접속하려면 **바인딩 주소를 변경**해야 한다.

### 8-1. SwarmUI 원격 접속 허용

SwarmUI 실행 인수에 아래 옵션을 추가한다.

StabilityMatrix에서:
1. SwarmUI 패키지 옆 `⋮` (더보기) 클릭
2. `Launch Options` 또는 `Advanced Settings` 진입
3. 실행 인수에 추가:

```
--host 0.0.0.0
```

또는 SwarmUI 설정 파일(`Data/Settings.fds` 또는 `Data/Config.json`)에서
`host` 항목을 `0.0.0.0`으로 변경한다.

이후 SwarmUI를 재시작하면 `http://[Tailscale IP]:7801`로 접속 가능해진다.

### 8-2. ComfyUI 원격 접속 허용

5-1단계에서 이미 `--listen 0.0.0.0`을 설정했다면 추가 작업은 필요 없다.
확인만 한다. ComfyUI가 `0.0.0.0:8188`로 바인딩된 상태여야
Tailscale IP를 통한 외부 접속도 가능하다.

### 8-3. Windows 방화벽 설정

Tailscale 트래픽은 `100.x.x.x` 대역을 사용한다.
Windows 방화벽이 이 포트를 막고 있으면 접속이 안 될 수 있다.

PowerShell (관리자)에서:

```powershell
# SwarmUI 포트 허용
New-NetFirewallRule -DisplayName "SwarmUI" -Direction Inbound -Protocol TCP -LocalPort 7801 -Action Allow

# ComfyUI 포트 허용
New-NetFirewallRule -DisplayName "ComfyUI" -Direction Inbound -Protocol TCP -LocalPort 8188 -Action Allow
```

또는 Windows Defender 방화벽 → 인바운드 규칙 → 새 규칙에서 포트를 직접 추가한다.

---

## 9. 원격 접속 테스트

모든 설정이 완료됐으면 원격 기기에서 접속을 확인한다.

1. 원격 기기에서 Tailscale이 실행 중인지 확인
2. GPU 서버의 Tailscale IP 확인 (예: `100.64.0.5`)
3. 브라우저에서 접속:

```
SwarmUI:  http://100.64.0.5:7801
ComfyUI:  http://100.64.0.5:8188
```

접속이 되면 집에 있는 GPU로 어디서든 이미지/영상 생성이 가능한 상태가 된 것이다.

---

## 10. 전체 구성 요약

```
[집 GPU 서버]
  └── StabilityMatrix
        ├── ComfyUI (포트 8188, --listen 0.0.0.0)
        │     └── custom_nodes/
        │           ├── SwarmComfyCommon  ← SwarmUI 연동 필수
        │           ├── SwarmComfyExtra   ← SwarmUI 연동 필수
        │           ├── ComfyUI-Manager
        │           └── 기타 커스텀 노드
        └── SwarmUI (포트 7801)
              └── Backends 설정
                    └── ComfyUI API (External) → http://127.0.0.1:8188

  └── Tailscale (100.64.0.5)

              ↕ Tailscale 가상 네트워크 (암호화된 P2P)

[원격 기기 (노트북/카페/회사)]
  └── Tailscale
  └── 브라우저 → http://100.64.0.5:7801 접속
```

---

## 11. 자주 생기는 문제

### 접속이 안 될 때 체크리스트

- 양쪽 기기 모두 Tailscale이 실행 중인가?
- 같은 Tailscale 계정으로 로그인했는가?
- SwarmUI/ComfyUI가 `0.0.0.0`으로 바인딩되어 실행 중인가?
- Windows 방화벽에서 해당 포트가 허용되어 있는가?
- GPU 서버가 절전/최대 절전 모드로 들어가지 않았는가?

### GPU 서버가 절전 모드로 꺼지는 문제

원격에서 사용하려면 서버가 항상 켜져 있어야 한다.
Windows 설정 → 전원 → 절전 모드를 `안 함`으로 설정한다.

### Tailscale 연결은 되는데 속도가 느릴 때

Tailscale 관리 콘솔에서 **DERP 릴레이** 서버를 통하는지 확인한다.
P2P 직접 연결이 안 되고 릴레이를 거치면 느려질 수 있다.
이 경우 공유기 UPnP를 활성화하거나, 방화벽 UDP 포트 41641을 열면 P2P 연결이 개선될 수 있다.

---

## 마치며

StabilityMatrix + SwarmUI + Tailscale 조합은 설정 난이도가 낮은 편이면서
실사용 만족도가 높다. 한 번 세팅해두면 집 밖에서도 집 GPU를 마음껏 쓸 수 있다.

정리하면:
- **StabilityMatrix**: 패키지 관리 통합, 모델 공유, 실행 편의성
- **SwarmUI**: 직관적인 UI, 다양한 백엔드 지원
- **ComfyUI 커스텀 노드**: 기능 확장의 핵심
- **Tailscale**: 복잡한 네트워크 설정 없이 어디서든 안전하게 접속
