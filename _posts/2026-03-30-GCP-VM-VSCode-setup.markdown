---
layout: post
title:  "GCP VM을 VS Code에서 바로 쓰기 위한 세팅 절차"
date:   2026-03-30 00:00:00 +0900
categories: ComfyUI
---

# GCP VM을 VS Code에서 바로 쓰기 위한 세팅 절차

이 글에서는 **GCP VM을 만들고 팀원이 접속한 뒤 VS Code에서 바로 작업할 수 있게** 정리했다.

이 글은 다음 상황을 기준으로 썼다.

- 팀장 1명이 먼저 VM과 JupyterHub를 세팅한다.
- 팀원은 VM의 **외부 IP**와 **리눅스 계정**을 전달받는다.
- 팀원은 **자기 로컬 PC의 공개 SSH 키**를 팀장에게 전달해 VM 접속 권한도 받아 둔다.
- 이후 팀원은 **VS Code + Remote SSH**로 접속해서 작업한다.
- GitHub 작업은 별도로 SSH 키 등록까지 끝내서 `push` 할 때마다 비밀번호를 다시 입력하지 않게 만든다.

> 참고
> 
> 원문은 JupyterHub 기준으로 설명되어 있다.
> 이 문서에서는 그 절차를 그대로 따르되, **실제 개발은 VS Code에서 하는 흐름**으로 다시 묶었다.
> 
> **VS Code Remote SSH 연결 부분은 원문의 VM 생성 + SSH 키 등록 절차를 VS Code에 맞게 옮긴 정리**다.

## 0. 준비물

미리 있어야 하는 것

- GCP 프로젝트 이름
- VM 이름
- GitHub 계정
- 프로젝트 GitHub 저장소 주소
- 팀장이 만들어 둔 리눅스 계정 이름
- 팀장이 VM 접속을 허용해 둔 상태

권장 환경

- 로컬 PC에 VS Code 설치
- VS Code 확장 `Remote - SSH` 설치
- GitHub 계정 로그인 완료

## 1. 팀장: VM 만들기

### 1-1. VM 생성

원문 기준 기본 설정은 아래와 같다.

- 위치: **아이오와**
- GPU: **L4 1개**
- 머신 타입: **g2-standard-4**
- 운영체제: **Ubuntu Accelerator 24.04 NVIDIA 580**
- 디스크 크기: **50GB**

Cloud Shell에서 만들면 아래 명령을 그대로 쓰면 된다.

```bash
gcloud compute instances create <VM 이름 입력!> \
  --project=<프로젝트 이름 입력! (ex: sprint-ai-01)> \
  --zone=us-central1-c \
  --machine-type=g2-standard-4 \
  --image-family=ubuntu-accelerator-2404-amd64-with-nvidia-580 \
  --image-project=ubuntu-os-accelerator-images \
  --boot-disk-type=pd-balanced \
  --boot-disk-size=50GB \
  --tags=jupyterhub \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

Cloud Shell이 안 열려도 괜찮다.

- Cloud Shell 사용량 초과로 안 될 수 있다.
- 이 경우 **GCP 콘솔 > Compute Engine > 인스턴스 만들기**에서 같은 조건으로 GUI 생성하면 된다.

### 1-2. 방화벽 열기

팀원들이 웹으로 JupyterHub에 접속할 수 있게 **8000 포트**를 연다.

```bash
gcloud compute firewall-rules create allow-jupyterhub-8000 \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=jupyterhub
```

### 1-3. SSH 접속 후 기본 상태 확인

VM에 SSH로 접속해서 아래를 확인한다.

```bash
python3 --version
nvidia-smi
```

여기서 `nvidia-smi`에 **CUDA Version**이 보여야 한다.

- 안 보이면 이미지가 잘못 들어간 경우라서 VM을 다시 만들거나 CUDA를 직접 설치해야 한다.

## 2. 팀장: JupyterHub 설치


### 2-1. 기본 패키지 설치

```bash
sudo apt update
sudo apt install python3.12-venv python3-pip nodejs npm
```

확인

```bash
pip --version
```

### 2-2. 관리자 가상환경 만들기

```bash
sudo -i
python3 -m venv /opt/jhub-venv
source /opt/jhub-venv/bin/activate
```

### 2-3. JupyterHub 관련 패키지 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install jupyterhub
npm install -g configurable-http-proxy
pip install jupyterlab notebook
```

### 2-4. JupyterHub 설정 파일 만들기

설정 파일 생성

```bash
jupyterhub --generate-config -f ~/jupyterhub_config.py
```

설정 파일 열기

```bash
nano ~/jupyterhub_config.py
```

아래처럼 수정한다.

```python
c = get_config()  #noqa

c.JupyterHub.bind_url = "http://:8000"

# 로그인 → 바로 JupyterLab
c.Spawner.default_url = "/lab"

# 허용 계정 등록 (리눅스 계정)
c.LocalAuthenticator.create_system_users = False
c.Authenticator.allowed_users = {"codeit", "wonchil",}  # 등록된 리눅스 계정 추가 (root 제외)
```

핵심은 이것이다.

- 8000 포트로 JupyterHub 실행
- 로그인하면 바로 JupyterLab 진입
- 실제 사용할 리눅스 계정을 `allowed_users`에 등록

### 2-5. 리눅스 계정 만들기

계정 추가 방법은 2가지다.

- 다른 팀원이 자기 GCP 계정으로 이 VM에 SSH 접속하면 리눅스 계정이 자동 생성될 수 있다.
- 또는 관리자가 직접 계정을 만들어도 된다.

직접 만들 때는 아래를 쓴다.

```bash
useradd -m -s /bin/bash codeit
passwd codeit
ls /home
```

### 2-6. VM 재부팅 시 JupyterHub 자동 실행

서비스 파일 생성

```bash
sudo nano /etc/systemd/system/jupyterhub.service
```

아래 내용 입력

```ini
[Unit]
Description=JupyterHub (venv)
After=network.target

[Service]
Type=simple
Environment="PATH=/opt/jhub-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Environment="PYTHONPATH=/opt/jhub-venv/lib/python3.12/site-packages"
ExecStart=/opt/jhub-venv/bin/jupyterhub \
          -f /root/jupyterhub_config.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

적용

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now jupyterhub
sudo journalctl -u jupyterhub -n 40 --no-pager
```

수정 후 재시작이 필요하면

```bash
sudo systemctl daemon-reload
systemctl restart jupyterhub
```

## 3. 팀원: JupyterHub 접속 확인

먼저 웹에서 접속해 환경이 살아 있는지 확인한다.

### 3-1. 브라우저 접속

팀장이 전달한 **외부 IP**로 접속

```text
http://외부IP:8000
```

예시

```text
http://34.170.107.101:8000
```

### 3-2. 로그인

- 팀장이 전달한 Username 입력
- 팀장이 전달한 Password 입력

### 3-3. GPU 동작 확인

Notebook의 Python 3를 열고 아래 실행

```python
import torch
print(torch.cuda.is_available())
```

- `True`가 나오면 정상
- `False`면 GPU가 VM에 제대로 안 올라온 상태라서 팀장이 다시 확인해야 한다.

## 4. 팀원: JupyterHub 안에서 개인 가상환경 만들기

> 개인이 JupyterHub 안에서 `pip`로 패키지를 설치하면,
> 공용 관리자 가상환경이 아니라 **사용자 계정 쪽 환경**에 설치된다.
> 그래서 사용자도 자기 가상환경을 따로 만들어 두는 편이 안전하다.

### 4-1. 사용자 가상환경 만들기

노트북 셀에서 아래를 순서대로 실행

```bash
!python3 -m venv ~/myenv
!~/myenv/bin/python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

### 4-2. 커널 바꾸기

- `Reconnect to Kernel` 클릭
- 우측 상단 커널 선택 메뉴 클릭
- `Python (myenv)` 선택

이제 이 계정은 자기 가상환경을 기준으로 패키지를 설치하고 실행하면 된다.

## 5. 팀원: GitHub 연결

### 5-1. 저장소 clone

JupyterHub 안에서 먼저 저장소를 가져온다.

```bash
!git clone {프로젝트 레포의 HTTPS 링크}
```

예시

```bash
!git clone https://github.com/hobi2k/LLMoker.git
```

### 5-2. 터미널에서 최신 코드 받기

JupyterLab 메뉴에서 터미널 열기

- `File → New → Terminal`

프로젝트 폴더로 이동 후 최신 코드 반영

```bash
cd {프로젝트 폴더명}
git pull origin main
```

## 6. 팀원: GitHub SSH 키 등록

`push` 할 때마다 아이디와 비밀번호를 치지 않게 만들려면 이 단계까지 해야 한다.

### 6-1. SSH 키 만들기

JupyterLab 터미널에서 아래 실행

```bash
ssh-keygen -t ed25519 -C "{깃허브에 등록된 본인 이메일}"
# 세 번 연속 엔터
ls ~/.ssh
# id_ed25519, id_ed25519.pub 등이 뜨면 SSH 키 발급 성공
cat ~/.ssh/id_ed25519.pub
# 뜨는 문자열이 본인의 SSH 키. 이메일 포함해서 키 전체 복사
```

### 6-2. GitHub에 SSH 키 등록

- `https://github.com/settings/keys` 로 이동
- `New SSH Key` 클릭
- Title은 원하는 이름 입력
- Key에는 `cat ~/.ssh/id_ed25519.pub` 로 출력한 전체 키 붙여넣기
- `Add SSH key` 클릭

### 6-3. 원격 저장소 주소를 HTTPS에서 SSH로 변경

현재 원격 주소 확인

```bash
git remote -v
```

아래처럼 `https://`로 나오면 아직 변경 전이다.

```text
origin  https://github.com/{username}/{프로젝트 레포 이름}.git (fetch)
origin  https://github.com/{username}/{프로젝트 레포 이름}.git (push)
```

이제 SSH 주소로 바꾼다.

```bash
git remote set-url origin git@github.com:{깃허브 username}/{프로젝트 레포지토리 이름}.git
git remote -v
```

바뀐 뒤에는 아래처럼 보여야 한다.

```text
origin  git@github.com:{username}/{프로젝트 레포 이름}.git (fetch)
origin  git@github.com:{username}/{프로젝트 레포 이름}.git (push)
```

GitHub 관련 명령을 처음 실행할 때는 이런 문구가 한 번 나올 수 있다.

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

이 경우 `yes` 를 입력하면 된다.

## 7. VS Code에서 바로 쓰기

여기부터는 **VM 생성 + SSH 계정 + Git SSH 등록**을 VS Code 작업 흐름으로 옮긴 단계다.

중요한 점부터 먼저 정리하면 다음과 같다.

- **VS Code로 VM에 접속할 때 쓰는 키**와
- **VM 안에서 GitHub에 push 할 때 쓰는 키**는
- 같은 키일 수도 있지만, 이 문서에서는 **서로 다른 용도**로 본다.

즉:

- 6번은 **VM 안에서 GitHub 인증용 키**
- 7번은 **내 로컬 PC에서 VM 로그인용 키**

### 7-1. 로컬 PC에서 VS Code 접속용 SSH 키 만들기

로컬 PC에서 아래를 실행한다.

```bash
ssh-keygen -t ed25519 -C "{내 PC에서 쓸 SSH 키}"
cat ~/.ssh/id_ed25519.pub
```

- 여기서 출력된 공개키를 팀장에게 전달한다.
- 이 키는 **VS Code가 VM에 접속할 때 쓰는 키**다.

### 7-2. 팀장이 GCP에 팀원 공개키 등록

이 단계가 안 되어 있으면 VS Code Remote SSH는 붙지 않는다.

공식 문서 기준으로 GCP VM의 SSH 관리는 두 방식이 있다.

#### 방법 A. OS Login을 쓰는 경우

로컬 PC에서 `gcloud`가 준비되어 있다면 아래처럼 공개키를 올릴 수 있다.

```bash
gcloud auth login
gcloud compute os-login ssh-keys add --key-file ~/.ssh/id_ed25519.pub
```

#### 방법 B. 메타데이터 SSH 키를 쓰는 경우

팀장이 GCP 콘솔에서 해당 VM의 SSH 키 항목에 팀원의 공개키를 추가한다.

이때 보통 아래 형식으로 넣는다.

```text
{리눅스계정명}:ssh-ed25519 AAAA...
```

여기서는 **둘 중 하나가 반드시 끝난 상태**여야 한다.

### 7-3. SSH 포트 열려 있는지 확인

원문에는 JupyterHub용 `8000` 포트만 열어 두는 절차가 나온다.
하지만 VS Code Remote SSH를 쓰려면 **SSH 포트 22도 접속 가능해야 한다.**

- 보통 기본 SSH는 22번 포트를 사용한다.
- VS Code 접속이 안 되면 팀장이 방화벽 규칙과 네트워크 접근 범위를 먼저 확인해야 한다.

### 7-4. VS Code에 Remote - SSH 설치

VS Code 확장에서 아래를 설치한다.

- `Remote - SSH`

### 7-5. 로컬 PC의 SSH 설정 파일 만들기

로컬 PC의 SSH 설정 파일에 아래를 추가한다.

macOS / Linux 기준 위치

```bash
~/.ssh/config
```

Windows 기준 위치

```text
C:\Users\{사용자이름}\.ssh\config
```

설정 예시

```sshconfig
Host team-gcp
    HostName {외부IP}
    User {리눅스 계정명}
    IdentityFile ~/.ssh/id_ed25519
```

주의

- 여기서 쓰는 `IdentityFile`은 **로컬 PC의 SSH 키**다.
- 위 6번에서 만든 키는 **JupyterHub/VM 안에서 GitHub 인증용**이다.
- 두 키를 헷갈리면 VS Code 접속 또는 GitHub push 둘 중 하나에서 막힌다.
- GCP Linux VM은 공식 문서 기준으로 **기본적으로 SSH 키 기반 인증**을 쓴다. 비밀번호 로그인 전제는 두지 않는 편이 안전하다.

### 7-6. VS Code에서 VM 접속

- `F1` 또는 `Ctrl+Shift+P`
- `Remote-SSH: Connect to Host`
- `team-gcp` 선택

정상 연결되면 VS Code가 원격 서버 창으로 다시 열린다.

### 7-7. 프로젝트 폴더 열기

원문에서 저장소를 JupyterHub 안에서 clone 했기 때문에, 같은 VM 파일시스템 안에 저장소가 있다.
보통은 사용자 홈 디렉터리 아래에 있다.

예시:

```bash
cd ~
ls
cd {프로젝트 폴더명}
```

즉 VS Code에서도 보통은 아래 경로를 열게 된다.

```text
/home/{리눅스계정명}/{프로젝트 폴더명}
```

필요하면 최신 코드 반영

```bash
git pull origin main
```

이후부터는

- 파일 편집은 VS Code에서 하고
- 실행은 원격 터미널에서 하고
- GitHub 작업은 SSH 방식으로 진행하면 된다.

## 8. 빠른 체크리스트

### 팀장 체크리스트

- VM 생성 완료
- L4 GPU / g2-standard-4 / Ubuntu Accelerator 24.04 NVIDIA 580 확인
- 8000 포트 개방 완료
- `nvidia-smi` 확인 완료
- JupyterHub 설치 완료
- 리눅스 계정 생성 완료
- 외부 IP / 계정 전달 완료
- 팀원 로컬 공개 SSH 키 등록 완료

### 팀원 체크리스트

- `http://외부IP:8000` 접속 성공
- JupyterHub 로그인 성공
- `torch.cuda.is_available()` 결과 `True` 확인
- `~/myenv` 가상환경 생성 완료
- 저장소 clone 완료
- `git pull origin main` 성공
- GitHub SSH 키 등록 완료
- `git remote -v` 결과가 `git@github.com:` 형태로 변경됨
- VS Code Remote SSH로 VM 접속 성공

## 9. 막히는 지점 정리

### Cloud Shell이 안 열릴 때

- 권한 문제일 수도 있지만, 원문 기준으로는 **Cloud Shell 사용량 초과**일 수 있다.
- 이 경우 GCP 콘솔에서 GUI로 같은 VM을 만들면 된다.

### `nvidia-smi` 에 CUDA가 안 보일 때

- 잘못된 이미지로 VM을 만든 경우가 많다.
- 원문 기준 이미지는 **Ubuntu Accelerator 24.04 NVIDIA 580** 이다.

### JupyterHub는 되는데 `pip install` 결과가 이상할 때

- 관리자 가상환경과 사용자 가상환경이 다르기 때문이다.
- 사용자 계정에서 `~/myenv` 같은 **개인 가상환경**을 따로 쓰는 쪽이 안전하다.

### VS Code에서 SSH 접속이 안 될 때

- 로컬 PC의 SSH 키가 VM 로그인 권한에 등록되지 않았을 수 있다.
- 팀장이 GCP 측 SSH 접속 권한을 먼저 열어 줬는지 확인해야 한다.
- GitHub용 SSH 키와 VM 로그인용 SSH 키는 목적이 다를 수 있다.

---

정리하면, 이 흐름으로 가면 된다.

1. 팀장이 VM + JupyterHub를 만든다.
2. 팀원은 웹에서 먼저 접속 확인을 한다.
3. 팀원은 자기 가상환경을 만든다.
4. 저장소를 clone 하고 GitHub SSH 키를 등록한다.
5. 마지막으로 VS Code Remote SSH로 VM에 붙어서 작업한다.
