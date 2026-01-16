---
layout: post
title:  "허깅페이스 MCP 코스 - Model Context Protocol (MCP) 종합 요약"
date:   2025-01-13 00:10:22 +0900
categories: Huggingface_mcp
---

# Model Context Protocol (MCP) 종합 요약

지금까지 Model Context Protocol(MCP)의 목적, 구조, 통신 방식, 그리고 실제 사용 생태계까지 전반적인 개념 지도를 완성하는 데 초점을 두었다. 이 정리는 “MCP가 무엇이고, 왜 필요하며, 어떻게 쓰이는가”를 한 번에 복기하기 위한 요약이다.

## 1. Model Context Protocol(MCP)란 무엇인가

**MCP(Model Context Protocol)**는 AI 모델을 외부 도구, 데이터 소스, 실행 환경과 연결하기 위한 **표준화된 통신 프로토콜**이다.  
기존 AI 시스템이 가진 한계(고립된 추론, 정적 컨텍스트, 툴 통합의 파편화)를 해결하기 위해 설계되었다.

핵심 목표는 다음과 같다.

- AI 애플리케이션과 외부 시스템 간 **상호운용성(interoperability)** 확보
- 실시간 데이터·도구 접근을 통한 **확장된 능력**
- M×N 통합 문제를 표준 인터페이스로 단순화

MCP는 흔히 “**AI 애플리케이션의 USB-C**”로 비유된다.

## 2. MCP의 기본 아키텍처: Client–Server 모델

MCP는 **클라이언트–서버(Client–Server) 아키텍처**를 따른다.  
이 구조는 역할 분리를 통해 모듈성과 확장성을 확보한다.

### 2.1 Host
- 사용자가 직접 상호작용하는 **AI 애플리케이션**
- 예: Chat UI, IDE(Cursor), 에이전트 애플리케이션
- 사용자 입력 처리, LLM 호출, 전체 흐름 오케스트레이션 담당

### 2.2 Client
- Host 내부에 존재하는 **MCP 통신 전담 컴포넌트**
- 특정 MCP Server와 **1:1 연결**
- 프로토콜 처리, Capability Discovery, 요청·응답 중계 담당

### 2.3 Server
- 외부 기능을 제공하는 **프로그램 또는 서비스**
- Tools, Resources, Prompts, Sampling을 MCP 표준으로 노출
- 기존 시스템을 감싸는 경량 래퍼 역할

이 구조 덕분에 MCP는 새로운 Server를 추가해도 Host를 수정하지 않는 **M+N 확장 구조**를 가능하게 한다.

## 3. MCP의 네 가지 Capability (핵심 원시 개념)

MCP는 외부 기능을 “기능 종류”가 아니라 **통제 주체와 위험도** 기준으로 네 가지로 나눈다.

### 3.1 Tools
- **실행 가능한 함수**
- 모델(LLM)이 호출 결정
- 부작용 가능 -> 사용자 승인 필요
- 예: API 호출, 메시지 전송, 계산 수행

### 3.2 Resources
- **읽기 전용 데이터 소스**
- 애플리케이션(Host)이 사용 시점 결정
- 부작용 없음
- 예: 파일 읽기, 문서·DB 조회

### 3.3 Prompts
- **사전 정의된 템플릿 / 워크플로우**
- 사용자 제어
- LLM의 사고 방식과 출력 구조를 규정
- 예: 코드 리뷰 프롬프트, 요약 템플릿

### 3.4 Sampling
- **Server가 Client/Host에 LLM 호출을 요청**
- 에이전트적·재귀적 작업 가능
- 사용자 승인 필요
- 예: 다단계 분석, 서버 주도 추론 루프

이 네 가지 구분은 MCP가 강력한 기능을 제공하면서도 **안전한 통제 경계**를 유지하게 하는 핵심 설계다.

## 4. 통신 프로토콜 핵심

### 4.1 메시지 포맷
- MCP는 **JSON-RPC 2.0**을 사용
- 메시지 유형:
  - Request
  - Response
  - Notification

### 4.2 전송 방식(Transport)
- **stdio**: 로컬 서버 (서브프로세스, stdin/stdout)
- **HTTP + SSE**: 원격 서버, 스트리밍 지원

### 4.3 상호작용 라이프사이클
1. Initialization (버전·Capability 협상)
2. Discovery (tools/list 등)
3. Execution (tools/call 등)
4. Termination (shutdown / exit)

## 5. Discovery 메커니즘

MCP Client는 Server의 기능을 **동적으로 탐색**한다.

- `tools/list`
- `resources/list`
- `prompts/list`

이를 통해 Client는 Server 기능을 사전에 하드코딩하지 않고,  
서버별 Capability 차이에 유연하게 적응할 수 있다.

## 6. MCP SDK

MCP는 언어 독립적이며, 공식 SDK를 제공한다.

SDK의 역할:
- JSON-RPC 통신 처리
- Capability 등록·Discovery
- 연결·에러 관리

대표 SDK:
- Python, TypeScript (Anthropic)
- Java, Kotlin, C#, Swift, Rust, Dart 등

SDK 덕분에 개발자는 **프로토콜 구현이 아니라 Capability 설계에 집중**할 수 있다.

## 7. Gradio + MCP

Gradio는 MCP Server를 만드는 **가장 쉬운 방법**을 제공한다.

- Python 함수 -> MCP Tool 자동 변환
- 웹 UI + MCP Server 동시 제공
- `launch(mcp_server=True)` 한 줄로 활성화
- Hugging Face Spaces에 무료 호스팅 가능

Gradio는 MCP를:
- 실험용
- 프로토타이핑
- 커뮤니티 공유
로 빠르게 확산시키는 핵심 도구다.

## 8. Hugging Face MCP 생태계

### 8.1 Hugging Face MCP Server
- MCP-compatible Client를 Hugging Face Hub에 직접 연결
- Models / Datasets / Spaces / Papers 검색 Tool 제공
- Gradio 기반 커뮤니티 MCP Tools(Spaces) 통합

### 8.2 사용 편의성 강화
- VS Code, Cursor, Zed, Claude Desktop 지원
- `https://huggingface.co/settings/mcp`에서 **원클릭 설정 스니펫**
- MCP-compatible Spaces를 설정으로 즉시 추가 가능

## 9. 핵심

- MCP는 AI와 외부 세계를 연결하는 **표준 연결 계층**
- Client–Server 분리로 확장성과 재사용성 확보
- Tools / Resources / Prompts / Sampling이라는 명확한 Capability 모델
- JSON-RPC 기반의 안정적인 통신
- SDK, Gradio, Hugging Face를 통해 **이미 실사용 가능한 생태계** 형성

참고자료
Huggingface, mcp course, https://huggingface.co/learn