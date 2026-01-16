---
layout: post
title:  "허깅페이스 MCP 코스 - Solution Walkthrough"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# Solution Walkthrough  
## Building a Pull Request Agent with MCP (End-to-End)

이 글은 지금까지의 실습 전체 과제를 관통하는 **Pull Request Agent MCP 서버의 최종 해설**이다.  
이 솔루션은 MCP의 세 가지 핵심 primitive인 **Tools / Resources / Prompts**를 모두 활용하여,
실제 개발팀에서 사용할 수 있는 **자동화 워크플로우 서버**를 구현한다.

## 전체 아키텍처 개요

PR Agent는 다음과 같은 단계적 구조로 구성된다.

1. **Module 1 – Build MCP Server**
   - 코드 변경 분석
   - PR 템플릿 자동 추천
2. **Module 2 – GitHub Actions Integration**
   - CI/CD 웹훅 수신
   - 표준화된 분석 워크플로우
3. **Module 3 – Slack Notification**
   - 팀 커뮤니케이션 자동화
4. **확장 요소**
   - Hugging Face Hub 특화 워크플로우
   - 모델 / 데이터셋 PR 대응

이 구조는 “단일 Agent”가 아니라  
**여러 MCP primitive가 협력하는 워크플로우 서버**라는 점이 핵심이다.

## Module 1: Build MCP Server

### 목표
Git diff 기반으로 변경 내용을 분석하고,  
Claude가 **적절한 PR 템플릿을 선택하도록 돕는 MCP 서버**를 구축한다.

### 핵심 구성 요소

#### 1. 서버 초기화 (`server.py`)

```python
# 서버는 다음 3개의 MCP Tool을 등록한다.
# - analyze_file_changes
# - get_pr_templates
# - suggest_template
```

이 서버는 “결론을 내리는 로직”을 직접 갖지 않는다.
대신 구조화된 데이터를 제공하고, 판단은 Claude에게 위임한다.

#### 2. analyze_file_changes Tool
이 Tool은 git diff를 분석하여 다음 정보를 제공한다.

- 변경된 파일 목록
- 파일 확장자 및 유형
- 추가 / 삭제 라인 수
- 테스트 / 문서 / 설정 파일 여부

이 데이터는 의사결정용 입력값으로만 사용된다.

#### 3. PR 템플릿 관리
PR 템플릿은 단순 Markdown 파일로 관리된다.

- bug.md
- feature.md
- docs.md
- refactor.md

Claude는 이 목록을 읽고,
분석 결과에 가장 적합한 템플릿을 선택한다.

#### Claude의 Tool 사용 흐름
1. analyze_file_changes 호출 -> 변경 요약 확보
2. get_pr_templates 호출 -> 선택지 파악
3. suggest_template 호출 -> 추천 + 이유 생성
3. 추천 템플릿을 상황에 맞게 수정

#### Module 1 학습 포인트
- Tool schema 설계
- 데이터 수집과 판단 로직의 분리
- “규칙 엔진”이 아닌 “맥락 기반 판단” 구조

## Module 2: Smart File Analysis & GitHub Actions Integration

### 목표

정적 코드 분석을 넘어 실시간 CI/CD 이벤트를 처리하는 MCP 서버로 확장한다.

### 핵심 구성 요소
#### 1. Webhook 서버
GitHub Actions 이벤트를 수신하기 위해
별도의 HTTP 서버를 실행한다.

```python
# 수신 이벤트 예시
# - workflow_run
# - check_run
# - pull_request
```

웹훅 서버는 이벤트를 github_events.json 파일에 저장하고,
MCP 서버는 파일만 읽는다.

HTTP 복잡도와 MCP 로직을 분리한 구조.

#### 2. MCP Prompts 도입
Module 2부터 Prompts가 핵심이 된다.

Prompts는 “도구”가 아니라
Claude에게 제공하는 표준화된 사고 절차다.

예시:

- CI 결과 분석
- 상태 요약 생성
- 후속 조치 제안

#### 3. 이벤트 처리 파이프라인
1. GitHub -> Webhook
2. 이벤트 JSON 저장
3. Claude가 Tool로 이벤트 읽음
4. Prompt에 따라 분석
5. 결과를 사용자에게 제공

#### Prompt 사용 예시
```python
prompt_data = {
    "event_type": "workflow_run",
    "status": "failure",
    "failed_jobs": ["unit-tests", "lint"],
    "error_logs": "...",
}
```

Claude는 이 데이터를 바탕으로:

- 실패 원인 요약
- 영향 범위 판단
- 다음 행동 제안

#### Module 2 학습 포인트
- 이벤트 기반 아키텍처
- Prompts를 통한 워크플로우 표준화
- “항상 같은 품질의 분석”을 보장하는 방법

## Module 3: Slack Notification

### 목표
CI/CD 결과와 PR 상태를 팀 전체에 자동 공유한다.

### 핵심 구성 요소
#### 1. Slack 연동 Tool

```python
# Slack Webhook을 호출하는 MCP Tool
# - send_slack_message
```

이 Tool은 외부 API 호출이라는 점에서
실제 운영 환경과 가장 가까운 예제다.

#### 2. 알림용 Prompts
Prompts는 메시지의 “형식과 톤”을 통제한다.

- CI 실패 알림
- 배포 성공 요약
- 중요도에 따른 메시지 스타일

#### 3. 통합 워크플로우 예시
CI 실패 발생 시:

1. Tool: CI 이벤트 조회
2. Resource: 팀 알림 규칙 확인
3. Prompt: 메시지 포맷 결정
4. Tool: Slack 메시지 전송

#### Module 3 학습 포인트
- 외부 API 연동 Tool 설계
- Prompts + Tools 결합 패턴
- “사람이 하던 소통을 자동화”하는 구조

## Hugging Face Hub 확장
이 솔루션은 Hub 특화 워크플로우로 확장 가능하다.

### Hub 전용 Tool 예시
```python
# - analyze_model_changes
# - validate_dataset_format
# - generate_model_card
```

### Hub 전용 Resource 예시
python
코드 복사
# hub://model-cards/
# hub://dataset-formats/
# hub://community-standards/

### Hub 전용 Prompt 예시
```python
# - Generate Benchmark Summary
# - Draft Model Card Update
```

LLM / Dataset PR에 특화된 자동화가 가능해진다.

### 전체 워크플로우 요약
하나의 PR에 대해 시스템은 다음을 수행한다.

1. 코드 변경 분석 (Tools)
2. 프로젝트 / 팀 컨텍스트 참조 (Resources)
3. 판단 및 설명 생성 (Prompts)
4. CI 이벤트 모니터링
5. 팀 커뮤니케이션 자동화

단일 Agent가 아니라 “워크플로우 서버”

### 테스트 전략

#### 단위 테스트

- Tool schema 검증
- Resource 접근 테스트
- Prompt 렌더링 확인

#### 통합 테스트

- PR -> CI -> 알림 전체 흐름
- 실패 / 복구 시나리오

### 설계 패턴 & 베스트 프랙티스

Tools

- 단일 책임
- 구조화된 출력
- 에러도 JSON으로 반환

Resources

- URI 기반 계층 구조
- 버전 관리 가능
- 캐시 고려

Prompts

- 구체적이되 유연하게
- 팀 표준 반영
- 재사용 가능하게 설계

## 결론
이 Pull Request Agent는 다음을 증명한다.

- MCP는 단순 “툴 호출 프로토콜”이 아니다
- Tools = 능력
- Resources = 맥락
- Prompts = 사고 절차

참고자료
Huggingface, mcp course, https://huggingface.co/learn