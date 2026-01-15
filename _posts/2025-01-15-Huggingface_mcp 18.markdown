---
layout: post
title:  "허깅페이스 MCP 코스 - Module 2: GitHub Actions Integration"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# Module 2: GitHub Actions Integration
## Real-Time CI/CD Monitoring with MCP Webhooks and Prompts

Module 2에서는 Module 1에서 구축한 **정적 PR 분석 MCP 서버**를 확장하여,  
**GitHub Actions CI/CD 이벤트를 실시간으로 수집·분석하는 시스템**을 구현한다.

핵심은 다음 전환이다.

- Module 1: 요청 기반 분석 (pull-based, static)
- Module 2: 이벤트 기반 감시 (push-based, dynamic)

## 문제 배경: Silent CI Failures

현실적인 CI/CD 문제는 단순하다.

- 테스트는 실패했지만
- 아무도 즉시 확인하지 않았고
- 결국 장애는 프로덕션에서 발생했다

GitHub Actions는 실패를 기록하지만,  
**“기다리지 않고 알려주지는 않는다”**는 것이 문제다.

> 수십 개 저장소, 수십 개 워크플로우 환경에서  
> 사람이 모든 CI 결과를 수동으로 확인하는 것은 불가능하다.

## Module 2의 목표

Module 2의 목표는 다음과 같다.

- GitHub Actions 이벤트를 **실시간으로 수신**
- 이벤트를 **지속적으로 저장**
- Claude가 언제든지 “최근 CI 상황”을 질의·분석 가능
- MCP Prompts를 통해 **일관된 CI 분석 워크플로우 제공**

결과적으로 PR Agent는 **“코드만 보는 도구” -> “개발 상태를 감시하는 에이전트”**로 진화한다.

## Module 2에서 추가되는 구성 요소

Module 1 대비 추가되는 요소는 다음과 같다.

- Webhook Server (HTTP)
- GitHub Actions 이벤트 저장소
- CI/CD 분석용 MCP Tools
- 워크플로우 표준화를 위한 MCP Prompts
- Cloudflare Tunnel 기반 로컬 테스트 환경

## 핵심 개념 정리

### 1. Webhook 기반 이벤트 수집

MCP 서버는 stdio 기반으로 Claude와 통신한다.  
GitHub Webhook은 HTTP 기반이다.

따라서 **두 개의 서버를 분리**한다.

1. Webhook Server (HTTP, port 8080)
2. MCP Server (Claude 연동, stdio)

> 이 분리는 “관심사의 분리”라는 관점에서 매우 중요하다.

### 2. 이벤트 저장 전략 (File-based)

Webhook 서버는 수신한 이벤트를 **JSON 파일에 누적 저장**한다.

```python
# File where webhook server stores events
EVENTS_FILE = Path(__file__).parent / "github_events.json"
```

이 방식의 장점:

- 데이터베이스 불필요
- 디버깅이 쉬움
- MCP Tool에서 단순 파일 읽기로 처리 가능
- 테스트 시 이벤트를 수동으로 추가 가능

## 3. MCP Prompts의 역할
Module 1에서는 Tools만 사용했다.
Module 2에서는 Prompts가 처음 등장한다.

차이점은 명확하다.


| 구분    | Tools  | Prompts   |
| ----- | ------ | --------- |
| 호출 주체 | Claude | 사용자       |
| 역할    | 데이터 접근 | 사고 흐름 가이드 |
| 반환값   | JSON   | 문자열 지시문   |


Prompts는 **“Claude에게 일을 시키는 방법을 표준화”**한다.

### 프로젝트 구조
```bash
github-actions-integration/
├── starter/
│   ├── server.py          # Module 1 + TODO
│   ├── webhook_server.py  # GitHub webhook receiver
│   └── pyproject.toml
└── solution/
    ├── server.py
    ├── webhook_server.py
    └── README.md
```

### 구현 단계

#### Step 1. Webhook Server 실행
Webhook 서버는 GitHub 이벤트를 받아 JSON 파일에 저장한다.

```bash
uv sync
python webhook_server.py
```

동작 방식 요약:

- /webhook/github 엔드포인트로 이벤트 수신
- 이벤트 payload + timestamp 저장
- github_events.json 파일이 이벤트 로그 역할 수행

#### Step 2. MCP 서버에서 이벤트 접근
MCP 서버는 HTTP를 직접 다루지 않는다.
대신 이벤트 파일을 읽는다.

```python
EVENTS_FILE = Path(__file__).parent / "github_events.json"
```

이 구조 덕분에 MCP 서버는 순수 분석 로직에만 집중할 수 있다.

#### Step 3. GitHub Actions 분석용 MCP Tools
Module 2에서는 기존 PR 분석 Tools에 더해,
CI/CD 상태를 다루는 Tools를 추가한다.

Tool 1: get_recent_actions_events

- 최근 GitHub 이벤트 조회
- 이벤트 개수 제한
- 파일이 없으면 빈 리스트 반환

Tool 2: get_workflow_status

- workflow_run 이벤트 필터링
- 워크플로우별 최신 상태 요약
- 성공/실패 여부 판단 가능

이 Tools들은 Claude가 **“현재 CI 상황을 이해”**하는 데 사용된다.

#### Step 4. MCP Prompts 구현
Prompts는 Claude에게 **“어떻게 분석할지”**를 알려주는 템플릿이다.

구현 대상 Prompts:

1. analyze_ci_results
-> CI 실패 원인 분석

2. create_deployment_summary
-> 팀 공유용 요약 생성

3. generate_pr_status_report
-> PR 변경 + CI 상태 통합 보고서

4. troubleshoot_workflow_failure
-> 실패 워크플로우 디버깅 가이드

Prompts는 반드시 `@mcp.prompt()` 데코레이터를 사용해야 한다.

#### Step 5. Cloudflare Tunnel을 통한 실전 테스트
로컬 webhook 서버를 GitHub에 노출하기 위해 Cloudflare Tunnel을 사용한다.

```bash
cloudflared tunnel --url http://localhost:8080
```

GitHub Webhook 설정:

- Payload URL: <tunnel-url>/webhook/github
- Event: workflow_run (또는 전체)

이제 실제 CI 실패 이벤트가 로컬 MCP 시스템으로 유입된다.

### 실습 과제 (Exercises)
#### Exercise 1. PR Review Prompt
- PR 변경 분석 (Module 1)
- CI 상태 (Module 2)
- 리뷰 체크리스트 생성

#### Exercise 2. Event Filtering 강화
- 성공/실패 기준 필터
- 저장소별 그룹핑
- 마지막 실행 시간 표시

#### Exercise 3. Notification 사전 분석
- 아직 “확인되지 않은” 실패 이벤트 추적
- 누구에게 알릴지 Claude가 추천

### 자주 발생하는 문제
1. Webhook 이벤트가 안 들어올 때
- cloudflared 실행 여부 확인
- GitHub Webhook delivery 로그 확인
- URL에 /webhook/github 포함 여부 확인

2. Prompt가 인식되지 않을 때
- `@mcp.prompt()` 데코레이터 확인
- 반환값은 문자열이어야 함

3. 포트 충돌
```bash
lsof -i :8080
```

## Module 2에서 얻은 것

기술적 성과
- Webhook 기반 이벤트 수집
- 파일 기반 이벤트 로그 아키텍처
- MCP Prompts를 통한 워크플로우 표준화

2. 개념적 확장
- 정적 분석 -> 실시간 감시
- 요청 기반 -> 이벤트 기반
- Tool 중심 -> Tool + Prompt 결합

참고자료
Huggingface, agents course, https://huggingface.co/learn