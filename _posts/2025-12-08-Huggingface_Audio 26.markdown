---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - SpeechT5 한국어 TTS 프로젝트 1"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_Audio
---

# 자모 토크나이저 용어집과 환경 설정 관리

## 전체 개요

첫 번째 프로토타입 SpeechT5 한국어 버전 제작기로서, 그 과정을 정리한다.
이 글은 다음 세 가지를 한 번에 정리한다.

1. 자모 기반 토크나이저(jamo tokenizer)를 만들 때 알아야 할 용어집
2. jamo_vocab_builder.py 코드 라인별 상세 해설
3. 우분투(WSL), Docker, uv 기반 파이썬 가상환경 설정에서 실제로 겪었던 문제와 정리

### 목차

1. 용어집
2. jamo_vocab_builder.py 코드 해설
3. 우분투 / WSL 환경 설정
4. uv 가상환경 & PyTorch 설치 정리
5. Docker + PyTorch 컨테이너 환경 정리

## 용어집

### 한글 / 자모 / 유니코드 관련

1. 완성형 한글 (Hangul Syllables, AC00–D7A3)

- 정의: 우리가 평소에 쓰는 “가, 나, 다, 랑, 봤” 같은 완성된 글자 한 칸 단위의 코드들.
- 유니코드 블록: U+AC00 ~ U+D7A3
- 특징

    - 하나의 완성형 글자 = 초성+중성+(종성) 이 조합된 결과.
    - “한글 1자 = 1유니코드 코드포인트” 라서 처리하기는 편하지만, 음절 단위라서 모든 발음 변형·외래어 등을 세밀하게 컨트롤하기는 애매함.

- 왜 TTS에서 문제될 수 있는가

    - 스펠링/발음 규칙을 정교하게 다루고 싶을 때, 문자 단위가 너무 크다.
    - 희귀 음절, 외래어, 교육용 예제 등에서 OOV(Out-of-Vocabulary) 이슈가 생기기 쉽다.

2. 자모 (Jamo, 1100–11FF, 3130–318F, 등)

- 정의: 한글을 구성하는 최소 단위 자음/모음.

    - 예: ㄱ, ㄴ, ㄷ, ㅏ, ㅑ, ㅇ, ㅂ 등.

- 유니코드 블록

    - 현대 한글 자모(초성): U+1100 ~ U+1112
    - 현대 한글 자모(중성): U+1161 ~ U+1175
    - 현대 한글 자모(종성): U+11A8 ~ U+11C2
    - (여기서는 정확한 끝값보다 “파이썬 range 에서 마지막은 포함 안 된다”는 점이 중요)

- 장점

    - 텍스트를 초성/중성/종성 단위까지 쪼개서 모델에 넣을 수 있음.
    - 발음, 연음, 받침, 어말어미 등을 훨씬 세밀하게 학습시킬 수 있다.
    - 희귀 음절이라도 자모 조합이 같으면 같은 토큰 패턴으로 취급 가능 -> OOV 감소.

- 단점

    - 시퀀스 길이가 길어진다.
    - 예를 들어 “학교” (2 글자) -> 초, 중, 종 기준 4~6 토큰.
    - 토크나이저, 디코딩 로직이 복잡해진다(자모 -> 완성형 재조립 필요).

33. 초성(Choseong), 중성(Jungseong), 종성(Jongseong)

- 초성(Choseong, CHOSEONG)

    - 한글 음절의 첫 자음. 예: ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
    - 코드에서:

```python
CHOSEONG = [chr(c) for c in range(0x1100, 0x1113)]
```

- 중성(Jungseong, JUNGSEONG)

    - 한글 음절의 모음. 예: ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ 등
    - 코드에서:

```python
JUNGSEONG = [chr(c) for c in range(0x1161, 0x1176)]
```

- 종성(Jongseong, JONGSEONG)

    - 받침 자음. 예: ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅇ, etc.
    - 코드에서:

```python
JONGSEONG = [chr(c) for c in range(0x11A8, 0x11C3)]
```

- 중요 포인트

    - 이 세 리스트를 합치면 현대 한국어용 자모 문자들을 포괄적으로 커버한다.
    - 이후 토크나이저는 이 문자들을 기반으로 자모 ID <-> 문자 매핑을 유지하게 된다.

### 토크나이저 / vocab 관련

4. Vocab (어휘 집합, 토큰 집합)

- 정의: 모델이 “알고 있는” 토큰들의 리스트.

    - 예: [ "`<pad>`", "`<unk>`", "`<bos>`", "`<eos>`", "ㄱ", "ㄴ", ..., "ㅏ", "ㅑ", ...]

- 역할

    - 각 토큰에 정수 ID를 부여하여 신경망이 처리할 수 있는 형태로 변환.
    - 예: "ㄱ": 4, "ㅏ": 35 와 같은 식.

- jamo_vocab.txt의 역할

    - 자모 토크나이저를 만들 때 기본 vocab 파일로 사용.
    - Hugging Face PreTrainedTokenizerFast 등에서 vocab_file로 읽어들여, token -> id / id -> token 매핑을 구성한다.

5.  Special Tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`)

- `<pad>`

    - 배치(batch)에서 시퀀스 길이를 맞추기 위해 빈 자리 채우기 용도로 쓰는 토큰.
    - 실제 문장 의미는 없고, 손실 계산에서도 보통 무시하도록 설정한다.

- `<unk>` (unknown)

    - Vocab에 없는 토큰이 들어왔을 때 대신 사용되는 “알 수 없는 토큰” 표시.
    - 자모 기반이면 거의 쓸 일은 줄어들지만, 여전히 안전장치로 두는 것이 일반적이다.

- `<bos>` (begin of sentence)

    - 문장의 시작을 알리는 토큰.
    - 디코더가 “여기서부터 문장을 생성하면 된다”고 인식하는 기준점.

- `<eos>` (end of sentence)

    - 문장의 끝을 알리는 토큰.
    - 디코더가 이 토큰을 생성하면 “여기서 멈추라”는 신호로 사용.

### 시스템 / 환경 / 툴 용어

6. 우분투(Ubuntu) / WSL2

- 우분투(Ubuntu)

    - 리눅스 배포판 중 하나. 개발, 서버, ML 환경에서 가장 많이 사용되는 OS.

- WSL2(Windows Subsystem for Linux 2)

    - 윈도우에서 리눅스 커널 환경을 직접 돌릴 수 있게 해주는 레이어.
    - 특징:
        - 윈도우에 따로 듀얼 부트할 필요 없이 Ubuntu 환경을 사용할 수 있다.
        - GPU 연동도 지원(CUDA/WSL), PyTorch 훈련 가능.

- 프로젝트 흐름

    - “윈도우 + Ubuntu(WSl2) + VSCode + Docker + conda(or uv)” 조합으로 전부 시도.

7. uv (astral-sh/uv)

- 정의: 빠른 패키지/환경 관리 툴.

    - pip + venv + pip-tools를 합쳐놓은 느낌.

- 장점

    - 매우 빠른 설치/해결.
    - uv venv, uv pip로 가상환경 만들고 관리 가능.

- 주의점

    - 파이썬 버전과 호환을 항상 신경 써야 한다.
    - 예: torchaudio 가 아직 cp313 (Python 3.13) wheel이 없는데 Python 3.13 환경에서 uv pip install torchaudio를 시도하면 “no wheels with matching ABI tag cp313” 같은 오류가 난다.

8. Docker / 이미지 / 컨테이너

- 이미지(Image)

    - 실행 가능한 환경 템플릿. OS + 라이브러리 + 설정이 포함된 “스냅샷”.
    - 예: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

- 컨테이너(Container)

    - 이미지를 기반으로 실제로 실행된 개별 인스턴스.
    - 각 컨테이너는 독립된 파일 시스템과 프로세스 공간을 가진다.

- 레지스트리(Registry)

    - Docker 이미지를 저장하고 가져오는 서버.
    - 예: Docker Hub, GitHub Container Registry.

- 메시지 예시

    - Unable to find image 'pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel' locally
    - 로컬에 이 이미지가 없으니, 레지스트리(Docker Hub 등)에서 받겠다는 의미.

## jamo_vocab_builder.py 코드 해설

이제 실제 코드 한 줄씩 뜯어보면서 무엇을, 왜 하는지 정리한다.

```python
from pathlib import Path
```

- **pathlib.Path**를 가져오는 부분.
- 기존의 os.path 대신 Path를 쓰면:
    - /, joinpath, .name, .suffix 등을 객체지향적으로 사용할 수 있다.
    - 운영체제(Windows, Linux)에 따라 적절한 폴더 구분자를 자동으로 사용한다.

```python
# 유니코드 자모 세트 정의
CHOSEONG = [chr(c) for c in range(0x1100, 0x1113)]
JUNGSEONG = [chr(c) for c in range(0x1161, 0x1176)]
JONGSEONG = [chr(c) for c in range(0x11A8, 0x11C3)]
```

### range + chr 로 자모 리스트 만들기

- range(0x1100, 0x1113)

    - 16진수 0x1100 ~ 0x1112 까지를 포함하는 정수 시퀀스.
    - range의 끝 인덱스는 포함되지 않는다는 점이 핵심.

- chr(c)

    - 정수 c를 유니코드 문자로 바꿔주는 파이썬 내장 함수.
    - 예: chr(0x1100) -> 'ᄀ' (초성 ㄱ 자모)

- 따라서 한 줄씩 의미는:

    - CHOSEONG: 유니코드 상의 초성 자모 문자 리스트
    - JUNGSEONG: 중성 자모 리스트
    - JONGSEONG: 종성 자모 리스트

왜 이렇게 구간을 잘랐나?

- 현대 한국어에 필요한 표준 자모 범위를 크게 커버하기 위한 선택이다.
- 자모 전체 블록(1100~11FF)을 무작정 다 쓰기보다는, 실제 현대 한국어에서 쓰는 범위로 제한하여 불필요한 토큰 수를 줄이기 위함.

```python
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
```

- 앞에서 용어집에 설명한 스페셜 토큰 4개를 정의.
- vocab 리스트에서 항상 가장 앞쪽에 배치하는 것이 일반적이다.
- 이유:

    - ID 0을 `<pad>`로 두는 경우가 많아서 마스킹/패딩 구현이 편하다.
    - `<bos>`, `<eos>` 등도 초기에 고정된 ID를 주고, 모델 config에서 재사용하기 쉬워진다.

```python
VOCAB = SPECIAL + CHOSEONG + JUNGSEONG + JONGSEONG
```

- 전체 vocab 리스트를 만드는 부분.
- 리스트 합치기:

    - [special 토큰들] + [초성들] + [중성들] + [종성들]

- 이 시점에서 토큰 순서가 곧 토큰 ID가 된다.
- 예:

```python
VOCAB[0] = "<pad>", VOCAB[1] = "<unk>", ...
```

- VOCAB[len(SPECIAL)] 이후부터는 실제 자모들이 등장.

```python
# svocab.txt 파일 저장
vocab_path = Path("jamo_vocab.txt")
```

- Path("jamo_vocab.txt")
    - 현재 작업 디렉토리(스크립트를 실행하는 위치)에 jamo_vocab.txt라는 파일 경로 객체를 만든다.
- 나중에 open(vocab_path, ...)로 실제 파일을 생성하게 된다.

```python
with open(vocab_path, "w", encoding="utf-8") as f:
    for tok in VOCAB:
        f.write(tok + "\n")
```

### vocab 파일 쓰기

- open(..., "w", encoding="utf-8")

    - "w": 쓰기 모드. 파일이 존재하면 내용이 덮어쓰이고, 없으면 새로 생성된다.
    - encoding="utf-8": 자모는 모두 유니코드 문자이므로 UTF-8 인코딩으로 저장.

- for tok in VOCAB: f.write(tok + "\n")

    - 각 토큰을 한 줄에 하나씩 저장.
    - 결과 파일 형태 예시:

```python
<pad>
<unk>
<bos>
<eos>
ᄀ
ᄁ
ᄂ
...
```

이 포맷은 HuggingFace의 tokenizers나 PreTrainedTokenizerFast에서 쉽게 사용할 수 있는 아주 단순한 형태다.

```python
print("Saved vocab:", vocab_path)
print("Total tokens:", len(VOCAB))
```

- 디버깅/확인용 출력.

    - vocab 파일이 어디에 생성되었는지,
    - 총 토큰이 몇 개인지 한 번에 확인할 수 있다.

- 이후 노트북/스크립트에서

    - “자모 vocab 크기 = len(VOCAB)”
    - “`<pad>`, `<unk>`, `<bos>`, `<eos>` 인덱스” 등을 config에 기록해 두면 좋다.

## 우분투(WSL) 환경 설정 정리

이제 자모 vocab 코드와 연결되는 실제 환경 셋업 이야기를 정리한다.
(“내가 실제로 TTS 훈련하기 위해 어떤 환경을 구성해야 했는가” 기준.)

### WSL2 + Ubuntu 설치 개념

1. 윈도우에서 WSL2 활성화

- PowerShell(관리자)에서:

```python
wsl --install
```

- 기본으로 Ubuntu가 설치되거나, 여러 배포판 중 선택 가능.

2. Ubuntu 첫 설정

- 처음 실행 시, Linux 계정 이름/패스워드 설정.
- 예: 사용자 이름 ahnhs2k.

3. 기본 패키지 업데이트

```python
sudo apt update
sudo apt upgrade -y
```

4. WSL에서 사용하는 디렉토리 구조

- 예: /home/ahnhs2k/pytorch-demo
- 여기 안에 jamo_vocab_builder.py, hugging_mission_tts.py 같은 파일들을 둔다.

### Python 버전 관리 (여기서 많이 꼬이기 쉬움)

- 문제 상황 예시

    - uv 환경에서 torchaudio 설치 시, “cp313 wheel이 없다”는 에러 → 현재 Python 버전이 3.13인 상태.

- 정리

    - 2024~2025 시점 기준 PyTorch는 보통 Python 3.10~3.12까지 안정적으로 지원.
    - Python 3.13은 아직 주요 라이브러리 wheel이 부족할 수 있다.

- 해결 전략

1. WSL에 Python 3.10 설치

```python
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

2. uv 환경 만들 때도 3.10을 명시적으로 사용하거나, pyenv로 3.10을 기본 버전으로 세팅.

## uv 가상환경 + PyTorch 설치 정리

uv를 사용하는 워크플로우를 한 번 정제해서 정리해 본다.

### 프로젝트 폴더 생성

```python
mkdir -p ~/pytorch-demo
cd ~/pytorch-demo
```

여기 안에 jamo_vocab_builder.py가 있다 가정.

### uv 설치

```python
curl -LsSf https://astral.sh/uv/install.sh | sh
# 설치 후, 셸 재시작 혹은
source ~/.profile  # 또는 ~/.bashrc
```

### uv 가상환경 생성 (Python 3.10 기준)

1. 우선 시스템에 3.10이 있다는 가정:

```python
python3.10 -m venv .venv
source .venv/bin/activate
```

혹은

2. uv로 바로:

```python
uv venv --python 3.10
source .venv/bin/activate
```

중요한 것은 현재 쉘에서 python이 3.10을 가리키게 하는 것.
python --version으로 반드시 확인한다.

### PyTorch 설치 (CUDA 12.x 예시)

- NVIDIA 드라이버 + CUDA가 깔려 있고, PyTorch 공식 wheel을 사용하는 경우:

```python
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```


- 주의 사항:

    - Python 버전이 지원 범위인지 반드시 확인 (3.10 권장).
    - CUDA 버전과 wheel 태그(cu121, cu118, cu128 등) 일치.

### 이 환경에서 jamo_vocab_builder.py 실행

```python
python jamo_vocab_builder.py
```

- 성공하면 현재 폴더에 jamo_vocab.txt가 생성된다.
- 이 파일을 이후 TTS 훈련 스크립트에서:

    - 직접 읽어서 커스텀 토크나이저에 전달하거나
    - Hugging Face Tokenizers를 통해 tokenizer.json 생성에 활용 가능.

## Docker + PyTorch 컨테이너 환경 정리

WSL 말고 Docker 컨테이너 내부에서 같은 작업을 할 수도 있다.
(예: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel 이미지 사용.)

### 기본 개념 요약

- 이미지:

    - “PyTorch + CUDA + Ubuntu + 개발 도구”가 이미 설치되어 있는 하나의 패키지.
    - pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel는 PyTorch 팀이 배포한 공식 이미지 중 하나.

- 컨테이너:

    - 이 이미지를 기반으로 실제로 실행된 하나의 환경(프로세스).
    - 안에서 bash, python, pip 등을 실행하게 된다.

### 이미지 받기 + 컨테이너 생성

```python
# 1. 이미지 받기 (로컬에 없으면 자동으로 pull 됨)
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# 2. 컨테이너 실행
docker run --gpus all -it --name pytorch-dev -v /home/ahnhs2k/pytorch-demo:/workspace pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel bash
```

- --gpus all: GPU를 컨테이너에 전달.
- -v /home/ahnhs2k/pytorch-demo:/workspace:
    - WSL의 프로젝트 폴더를 컨테이너의 /workspace로 마운트.
    - 이렇게 하면 컨테이너 안에서 생성한 jamo_vocab.txt가 호스트에 그대로 남는다.

컨테이너 안으로 들어가면:

```python
cd /workspace
python jamo_vocab_builder.py
```

- 밖에서 /home/ahnhs2k/pytorch-demo를 보면 jamo_vocab.txt가 존재.

### 자주 보게 되는 메시지 이해

- Unable to find image 'pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel' locally

    - 아직 로컬에 이 이미지가 없어서, Docker Hub에서 다운로드(pull) 한다는 뜻.

- docker ps

    - 현재 실행 중인 컨테이너 목록.

- docker ps -a

    - 종료된 컨테이너 포함 전체 목록.

- docker start -ai pytorch-dev

    - 한 번 만든 pytorch-dev 컨테이너에 다시 들어가기.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn