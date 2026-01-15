---
layout: post
title:  "허깅페이스 MCP 코스 -  Project Setup: Hugging Face Hub Pull Request Agent"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# Project Setup: Hugging Face Hub Pull Request Agent

이 글에서는 Hugging Face Hub 기반 Pull Request Agent를 개발하기 위한  
**프로젝트 초기 세팅과 개발 환경 구성**을 다룬다.

목표는 다음과 같다.

- 최신 Python 툴링(`uv`)을 사용한 **일관된 의존성 관리**
- MCP Server + Webhook Agent 구조에 맞는 **명확한 프로젝트 레이아웃**
- 로컬 개발과 Hugging Face Spaces 배포를 모두 고려한 **환경 설정**

## 1. 프로젝트 구조 개요

본 프로젝트는 Hugging Face Spaces 템플릿을 기반으로 시작한다.

```bash
git clone https://huggingface.co/spaces/mcp-course/tag-this-repo
```

최종적으로 사용할 프로젝트 구조는 다음과 같다.

```
hf-pr-agent/
├── mcp_server.py        # 핵심 MCP Server (모델 태그 관련 Tools 제공)
├── app.py               # FastAPI Webhook Listener + Agent 로직
├── requirements.txt     # 배포 호환성을 위한 의존성 목록
├── pyproject.toml       # uv 기반 프로젝트 설정
├── env.example          # 환경 변수 템플릿 (시크릿 문서화용)
├── cleanup.py           # 개발/테스트 보조 유틸리티
```

### 구조 설계 의도

- mcp_server.py
  - MCP Protocol에 노출되는 “행위 가능한 도구”만 담당

- app.py
  - 외부 세계(Hub Webhook)와 MCP Server를 연결하는 Agent 레이어

- 설정 파일 분리
  - 로컬 / 배포 환경 모두에서 재현 가능한 구성

## 2. 의존성 및 프로젝트 설정
### 2.1 pyproject.toml 구성
본 프로젝트는 uv 기반의 modern Python packaging을 사용한다.
의존성, Python 버전, 빌드 설정을 단일 파일로 관리한다.

```python
[project]
name = "mcp-course-unit3-example"
version = "0.1.0"
description = "FastAPI and Gradio app for Hugging Face Hub discussion webhooks"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "gradio>=4.0.0",
    "huggingface-hub[mcp]>=0.32.0",
    "pydantic>=2.0.0",
    "python-multipart>=0.0.6",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "fastmcp>=2.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

#### 주요 포인트

- huggingface-hub[mcp]
  - Hub API + MCP 통합 기능 사용

- fastmcp
  - MCP Server 구현 핵심 라이브러리

- fastapi + uvicorn
  - Webhook Listener 및 Agent API

- gradio
  - 테스트 및 디버깅용 UI

### 2.2 requirements.txt
배포 환경(Hugging Face Spaces 등)과의 호환성을 위해
동일한 의존성을 requirements.txt에도 명시한다.

pyproject.toml + requirements.txt 이중화는
현업 배포에서 매우 흔한 패턴이다.

### 2.3 가상환경 생성 및 의존성 설치
```bash
uv venv
source .venv/bin/activate   # Windows: .venv/Scripts/activate
```
```bash
uv sync
```

이 단계까지 완료되면:

- Python 3.11+
- 모든 MCP / FastAPI / Hub 의존성
- 재현 가능한 개발 환경

이 준비된다.

## 3. 환경 변수 설정
### 3.1 env.example
실제 시크릿을 담지 않고,
필수 환경 변수를 문서화하기 위한 템플릿 파일이다.

```bash
# Hugging Face API Token (required)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here

# Webhook Secret (required for production)
# Use a strong, random string
WEBHOOK_SECRET=your-webhook-secret-here

# Model for the agent (optional)
HF_MODEL=owner/model

# Provider for MCP agent (optional)
HF_PROVIDER=huggingface
```

### 3.2 Hugging Face API Token
- 발급 위치: https://huggingface.co/settings/tokens
- 권한: repo / write 필요
- 사용처:
  - 모델 메타데이터 읽기
  - PR 생성

### 3.3 Webhook Secret 생성
Webhook 위조 방지를 위해 반드시 랜덤 시크릿 사용.

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

생성된 값을 .env 파일에 추가한다.

```bash
WEBHOOK_SECRET=generated_secret_here
```

.env 파일은 반드시 .gitignore에 포함해야 한다.

## 4. 설정 설계 철학
- 코드와 시크릿 분리
- 문서화된 env.example
- 로컬 / 배포 동일 동작 보장
- Webhook 보안 기본 내장

이 구조는:
- 개인 프로젝트
- 팀 서비스
- Hugging Face Spaces 배포
모두에 적합하다.

참고자료
Huggingface, agents course, https://huggingface.co/learn