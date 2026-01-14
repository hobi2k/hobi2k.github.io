---
layout: post
title:  "허깅페이스 MCP 코스 - Lemonade Server로 완성하는 온디바이스·프라이버시 우선 MCP Agent"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Local Tiny Agents with AMD NPU·iGPU Acceleration
*― Lemonade Server로 완성하는 온디바이스·프라이버시 우선 MCP Agent*

이번 글의 핵심은 다음 두 가지다.

1. Tiny Agents + MCP 파이프라인을 AMD NPU / iGPU 가속으로 로컬에서 실행

2. 민감한 데이터를 외부로 보내지 않는 완전 온디바이스 AI Assistant 구축

이를 위해 Lemonade Server를 사용한다.

## 1. 왜 Lemonade Server인가

Lemonade Server는 다음 요구사항을 동시에 만족한다.

- 로컬 LLM 실행
- AMD iGPU (Vulkan) 가속
- AMD NPU (Ryzen AI 300 시리즈) 지원
- OpenAI-compatible API 제공
- Tiny Agents / MCP와 바로 연동 가능

즉, “로컬 가속 + MCP + Agent”를 가장 현실적으로 연결해 주는 서버다.

## 2. 전체 아키텍처 개요

이번 섹션에서 완성하는 구조는 다음과 같다.

```
[Tiny Agent]
   ↓
[MCP Client]
   ├─ Gradio MCP Server (도구)
   ├─ Desktop Commander MCP (파일·터미널)
   ↓
[Lemonade Server]
   ├─ LLM (Qwen / Jan-Nano 등)
   ├─ iGPU (Vulkan)
   └─ NPU (Ryzen AI)
```

결과적으로:

- 모델 추론 -> 로컬 가속
- 파일·이력서·개인정보 -> 외부 유출 없음
- Agent 행동 -> MCP로 통제 가능

## 3. Lemonade Server 설치
### 3.1 Windows

최신 설치 파일 다운로드 후 실행
https://github.com/lemonade-sdk/lemonade/releases/latest

지원 요약


| 항목   | 지원                         |
| ---- | -------------------------- |
| CPU  | 전체                         |
| iGPU | Vulkan (Ryzen AI / Radeon) |
| NPU  | Ryzen AI 300 (Windows 전용)  |
| 엔진   | llama.cpp, OGA             |


설치 후 바탕화면의 Lemonade 아이콘으로 실행.

### 3.2 Linux

(1) 가상환경 생성
```bash
uv venv --python 3.11
source .venv/bin/activate
```

(2) 설치
```bash
uv pip install lemonade-sdk==8.0.3
```

또는 소스 설치:
```bash
git clone https://github.com/lemonade-sdk/lemonade-sdk.git
cd lemonade-sdk
pip install -e .
```

(3) 실행
```bash
lemonade-server-dev serve
```

NPU 가속은 Windows + Ryzen AI 300 전용

## 4. Tiny Agents + Lemonade 연동
### 4.1 기본 개념

Tiny Agents는 OpenAI-compatible endpoint만 있으면 된다.
Lemonade Server는 이를 제공한다.

### 4.2 agent.json 예시 (Gradio MCP Server 연결)
Windows
```json
{
  "model": "Qwen3-8B-GGUF",
  "endpointUrl": "http://localhost:8000/api/",
  "servers": [
    {
      "type": "stdio",
      "command": "C:\\Program Files\\nodejs\\npx.cmd",
      "args": [
        "mcp-remote",
        "http://localhost:7860/gradio_api/mcp/sse"
      ]
    }
  ]
}
```

Linux
```json
{
  "model": "Qwen3-8B-GGUF",
  "endpointUrl": "http://localhost:8000/api/",
  "servers": [
    {
      "type": "stdio",
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:7860/gradio_api/mcp/sse"
      ]
    }
  ]
}
```

- 모델 추론: Lemonade (로컬)
- 도구 호출: MCP
- Agent 제어: Tiny Agents

## 5. 민감 데이터 전용 로컬 Assistant 만들기

이제 진짜 의미 있는 시나리오를 만든다.

“이력서, 채용 문서, 개인 정보는 절대 외부 전송 금지”

이를 위해 Desktop Commander MCP Server를 사용한다.

## 6. Desktop Commander MCP Server

이 MCP Server는 다음을 제공한다.

- 로컬 파일 읽기/쓰기
- 디렉토리 탐색
- 코드 편집
- 터미널 명령 실행
- 프로세스 제어

즉, Agent에게 ‘로컬 OS의 손과 발’을 쥐여주는 MCP

## 7. 파일 처리 전용 Agent 구성
### 7.1 프로젝트 생성
```bash
mkdir file-assistant
cd file-assistant
```

### 7.2 agent.json
Windows
```json
{
  "model": "user.jan-nano",
  "endpointUrl": "http://localhost:8000/api/",
  "servers": [
    {
      "type": "stdio",
      "command": "C:\\Program Files\\nodejs\\npx.cmd",
      "args": [
        "-y",
        "@wonderwhy-er/desktop-commander"
      ]
    }
  ]
}
```

Linux
```json
{
  "model": "user.jan-nano",
  "endpointUrl": "http://localhost:8000/api/",
  "servers": [
    {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@wonderwhy-er/desktop-commander"
      ]
    }
  ]
}
```

### 7.3 Jan Nano 모델 추가

Lemonade UI에서:

- Model Name: user.jan-nano
- Checkpoint:
  Menlo/Jan-nano-gguf:jan-nano-4b-Q4_0.gguf
- Recipe: llamacpp

## 8. 실제 시나리오: 로컬 채용 Assistant
### 8.1 입력 데이터

- job_description.md
- candidates/john_resume.md

### 8.2 Agent가 할 수 있는 일

1. 채용 공고 읽기
2. 이력서 분석
3. 적합성 평가
4. 결과 요약
5. 초대장 문서 생성

모든 과정이:

- 로컬 파일
- 로컬 모델
- 로컬 가속
- 외부 전송 없음

### 8.3 실행
```bash
tiny-agents run agent.json
```

Agent 로딩 시:

```csharp
Agent loaded with 18 tools:
 • read_file
 • write_file
 • list_directory
 • execute_command
 • edit_block
 ...
```

### 8.4 실제 프롬프트 예시
```
Inside the same folder you can find a candidates folder.
Check for john_resume.md and let me know if he is a good fit for the job.

Create a file called "invitation.md" and write a short interview invitation.
```

-> Agent가 파일을 직접 읽고, 판단하고, 새 문서를 생성

## 9. 성능·확장 포인트

- Jan-Nano: Vulkan 기반 iGPU 가속
- Hybrid 모델: NPU + iGPU 혼합 (Ryzen AI 300)
- Tool-calling 특화 모델 사용 가능
- MCP Server 추가 시 즉시 Agent 능력 확장

## 10. 결론

이 섹션이 보여준 핵심은 명확하다.

- MCP는 연결 표준
- Tiny Agents는 가벼운 두뇌
- Lemonade Server는 로컬 가속 심장

참고자료
Huggingface, agents course, https://huggingface.co/learn