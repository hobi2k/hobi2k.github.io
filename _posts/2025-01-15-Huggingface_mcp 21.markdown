---
layout: post
title:  "허깅페이스 MCP 코스 - Build a Pull Request Agent on the Hugging Face Hub"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# Build a Pull Request Agent on the Hugging Face Hub

이 글에서는 **Hugging Face Hub의 실시간 이벤트(Webhook)**와  
**MCP 서버**를 결합하여,  
모델 리포지토리의 메타데이터(특히 tags)를 자동으로 개선하는  
**실전형 Pull Request Agent**를 구축한다.

이 PR Agent는 토론(discussion)과 댓글(comment)을 감시하고,  
적절한 태그를 추론하여 **자동으로 PR을 생성**한다.

## 학습 목표

이 글을 통해 다음을 학습한다.

- Hugging Face Hub API와 상호작용하는 **MCP Server 구현**
- Hub Webhook을 수신하는 **실시간 이벤트 처리**
- 토론 기반 **자동 태깅 PR 워크플로우 설계**
- MCP + Webhook 기반 애플리케이션을 **Hugging Face Spaces에 배포**

최종적으로,  
사람이 직접 태그를 관리하지 않아도  
**“토론 -> 분석 -> PR 생성”** 이 자동으로 이루어지는 Agent를 완성한다.

## 사전 요구사항 (Prerequisites)

이 실습을 진행하기 전에 다음이 필요하다.

```
- MCP 개념 이해
- Python, FastAPI, Webhook 기본 지식
- Hugging Face Hub의 모델 / PR / Discussion 흐름 이해
- 개발 환경:
  - Python 3.11+
  - Hugging Face 계정 및 API 토큰
```

## Pull Request Agent 개요

이 프로젝트의 PR Agent는 다음 네 가지 핵심 구성 요소로 이루어진다.

1. **MCP Server**
   - Hub 메타데이터를 읽고 수정하는 Tool 제공
2. **Webhook Listener**
   - Hub Discussion 이벤트 실시간 수신
3. **Agent Logic**
   - 댓글을 분석하여 태그를 추론하고 PR 생성
4. **Deployment Infrastructure**
   - Hugging Face Spaces에 서비스 배포

이 Agent의 목적은 다음과 같다.

> 모델 작성자가 토론을 통해 태그를 논의하면,  
> Agent가 이를 감지해 **바로 사용할 수 있는 PR을 대신 만들어주는 것**

## 프로젝트 파일 구조


| 파일 | 역할 | 설명 |
|----|----|----|
| `mcp_server.py` | Core MCP Server | FastMCP 기반 서버, 모델 태그 읽기/수정 Tool 제공 |
| `app.py` | Webhook Listener & Agent | FastAPI 기반 Webhook 수신 + PR 생성 로직 |
| `requirements.txt` | Dependencies | FastMCP, FastAPI, huggingface-hub 등 |
| `pyproject.toml` | Project Config | uv 기반 현대적 패키징 |
| `Dockerfile` | Deployment | Hugging Face Spaces용 컨테이너 |
| `env.example` | Config Template | 환경 변수 / 시크릿 템플릿 |
| `cleanup.py` | Utility | 테스트 및 개발용 정리 스크립트 |


## MCP Server (`mcp_server.py`)

프로젝트의 핵심이다.  
FastMCP 기반 서버로, Claude(또는 다른 MCP Client)가 호출할 수 있는 Tool을 제공한다.

### 주요 기능

- 모델 리포지토리의 현재 태그 조회
- 새로운 태그를 추가하는 PR 생성
- 입력 검증 및 에러 처리

**의사결정 로직은 MCP 서버에 없고**,  
서버는 “행위 가능한 도구”만 제공한다.

## Webhook 연동

Hugging Face Hub의 공식 Webhook 가이드를 따른다.

- Discussion / Comment 이벤트 수신
- Webhook signature 검증 (보안)
- 태그 제안 및 멘션 처리
- PR 자동 생성

Webhook은 **Agent의 트리거** 역할을 하며,  
실시간 자동화를 가능하게 만든다.

## Agent 동작 방식

Agent는 토론 내용에서 다음을 수행한다.

- 명시적 태그 추출  
  - 예: `tag: pytorch`, `#transformers`
- 암묵적 태그 추론  
  - 자연어 기반 의미 분석
- 태그 검증  
  - ML / AI 카테고리 기준
- PR 설명 자동 생성  
  - “왜 이 태그가 필요한지” 포함

이 과정은 **MCP Tool 호출 + Agent 로직**의 결합으로 이루어진다.

## Webhook 처리 흐름

Webhook 기반 실시간 흐름은 다음과 같다.

1. 사용자가 모델 리포지토리 토론에 댓글 작성
2. Hugging Face Hub -> Webhook POST 전송
3. 서버에서 Webhook 시크릿 검증
4. Agent가 댓글 분석
5. 관련 태그 발견 시 PR 생성
6. Webhook 응답은 즉시 반환, PR 생성은 백그라운드 처리

Hub의 Discussion Bot과 동일한 패턴을 따른다.

## 배포 (Deployment)

- Docker 기반 컨테이너 구성
- Hugging Face Spaces에 배포
- 환경 변수로 API 토큰 및 시크릿 관리
- 백그라운드 작업으로 Webhook 응답 지연 방지
- Gradio UI로 테스트 및 상태 확인 가능

## 정리

이 글은 다음을 명확히 보여준다.

- MCP는 **정적 도구 호출**이 아니라
- **실시간 이벤트 + 자동화 워크플로우**에 적합한 프로토콜이다
- Hugging Face Hub 같은 플랫폼과 결합하면
  - “사람이 하던 유지보수 작업”을
  - **Agent가 PR로 대신 처리**할 수 있다

참고자료
Huggingface, agents course, https://huggingface.co/learn