---
layout: post
title:  "허깅페이스 MCP 코스 - Advanced MCP Development"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Advanced MCP Development
## Custom Workflow Servers for Claude Code

여기서부터는 **Claude Code에 팀·워크플로우 맥락을 부여하는 실전 MCP 서버**를 구축한다.  
단순 Tool 호출을 넘어서, **개발 팀의 실제 업무 흐름(PR -> CI/CD -> 커뮤니케이션)**를 자동화하는 것이 목표다.

## 이 Unit에서 만들 것

### PR Agent Workflow MCP Server

Claude Code가 단순 코드 어시스턴트를 넘어 **팀 인지형(workflow-aware) 개발 에이전트**로 동작하도록 만드는 MCP 서버를 구현한다.

주요 기능은 다음과 같다.

- **Smart PR Management**
  - 코드 변경 내용을 분석해 적절한 PR 템플릿 자동 제안
  - MCP Tools 기반

- **CI/CD Monitoring**
  - GitHub Actions 상태를 실시간으로 수신
  - Cloudflare Tunnel을 통한 webhook 처리
  - 표준화된 MCP Prompts로 결과 요약

- **Team Communication**
  - 빌드/배포 성공·실패 시 Slack 자동 알림
  - MCP Tools + Prompts + Integration 결합 사례

## 실제 개발 팀 시나리오

### Before (수동 워크플로우)

- 개발자가 PR 생성
- GitHub Actions 완료까지 대기
- 결과를 직접 확인
- 팀에 수동으로 공유

### After (MCP 기반 자동화)

Claude Code + Workflow MCP Server 조합으로:

- 변경 파일을 분석해 **적절한 PR 템플릿 자동 제안**
- GitHub Actions 실행 상태를 **실시간 모니터링**
- 결과를 **요약된 형태로 Claude가 설명**
- 성공/실패 시 **Slack 자동 알림**
- 팀별 리뷰 규칙을 Prompt로 가이드

## 학습 목표 (Key Learning Outcomes)

이 실습을 통해 다음을 실전 수준으로 익힌다.

1. **MCP 핵심 프리미티브 활용**
   - Tools와 Prompts를 실제 워크플로우에 적용
2. **MCP Server 구조 설계**
   - 실사용 가능한 서버 구성 및 에러 처리
3. **GitHub Actions 연동**
   - Cloudflare Tunnel을 이용한 webhook 수신
4. **Hugging Face Hub 워크플로우**
   - LLM 개발 팀에 특화된 자동화 패턴
5. **다중 시스템 통합**
   - GitHub · Slack · Hugging Face를 MCP로 연결
6. **Claude Code 확장**
   - Claude가 팀 규칙과 맥락을 이해하도록 설계

## MCP Primitives 적용 구조

실습에서는 MCP의 모든 핵심 요소를 실제로 사용한다.

- **Tools (Module 1)**
  - 파일 변경 분석
  - PR 템플릿 제안

- **Prompts (Module 2)**
  - CI/CD 결과 요약 포맷
  - 팀 표준 워크플로우 정의

- **Integration (Module 3)**
  - GitHub Actions 이벤트 처리
  - Slack 알림 전송
  - 모든 프리미티브 결합

## Module 구성

1. **Module 1: Build MCP Server**
   - PR 템플릿 추천을 위한 기본 MCP Tool 구현
2. **Module 2: GitHub Actions Integration**
   - Cloudflare Tunnel 기반 CI/CD 이벤트 처리
   - Prompts를 통한 표준화된 요약
3. **Module 3: Slack Notification**
   - 팀 커뮤니케이션 자동화
   - MCP 전체 프리미티브 통합

## Claude Code 설치 및 준비

이 실습은 **Claude Code 환경에서 MCP Server를 직접 테스트**한다.

### 필수 사항

- Claude Code 설치 및 인증 완료

### 설정 개요

- npm을 통한 Claude Code 설치
- 프로젝트 디렉토리에서 `claude` 실행
- console.anthropic.com을 통한 인증

문제가 발생할 경우 공식 troubleshooting 가이드를 참고한다.

## 핵심

> **MCP는 단순히 LLM에 도구를 붙이는 기술이 아니라,  
> 팀의 실제 개발 워크플로우를 AI가 이해하고 실행하게 만드는 표준이다.**

Claude Code는 더 이상 “코드만 아는 AI”가 아니라  
**팀의 개발 흐름을 이해하고 함께 일하는 에이전트**가 된다.

참고자료
Huggingface, agents course, https://huggingface.co/learn