---
layout: post
title:  "허깅페이스 MCP 코스 - MCP(Model Context Protocol) 통신 프로토콜 정리"
date:   2025-01-13 00:10:22 +0900
categories: Huggingface_mcp
---

# MCP(Model Context Protocol) 통신 프로토콜 정리

이 섹션은 MCP가 **실제로 어떻게 메시지를 주고받는지**, 즉 Client와 Server 간 통신이 어떤 규칙과 흐름으로 이루어지는지를 다룬다.  
앞선 아키텍처(Host–Client–Server)가 “구조”라면, 이 장은 그 구조 위에서 동작하는 **프로토콜의 세부 규약**에 해당한다.

## 1. MCP 통신 프로토콜의 목적

MCP는 Client와 Server가 메시지를 주고받을 때 다음을 보장하는 것을 목표로 한다.

- **일관성(Consistency)**: 모든 MCP 구현체가 동일한 규칙으로 통신
- **예측 가능성(Predictability)**: 요청–응답–에러 처리 패턴이 명확
- **상호운용성(Interoperability)**: 언어·플랫폼·환경이 달라도 호환 가능

이를 위해 MCP는 자체적인 메시지 포맷을 새로 정의하지 않고, **검증된 기존 표준 위에 설계**되었다.

## 2. JSON-RPC 2.0: MCP 통신의 기반

MCP의 모든 Client–Server 통신은 **JSON-RPC 2.0**을 기반으로 한다.

### JSON-RPC를 채택한 이유
- JSON 기반 -> 사람이 읽고 디버깅하기 쉬움
- 언어 독립적 -> 어떤 언어로든 구현 가능
- 명확한 스펙 -> 요청/응답/에러 처리 규칙이 엄격히 정의됨
- 이미 널리 쓰이는 표준 -> 학습 비용과 구현 리스크 감소

MCP는 “무엇을 보낼지”뿐 아니라 “어떤 형태로 보내야 하는지”를 JSON-RPC로 고정함으로써, 구현체 간 차이를 최소화한다.

## 3. MCP 메시지 유형

JSON-RPC 2.0 기반으로 MCP는 **세 가지 메시지 유형**을 사용한다.

### 3.1 Request (요청)

Client -> Server로 보내는 메시지로, **어떤 작업을 수행해 달라**는 요청이다.

요청 메시지의 핵심 요소:
- `id`: 요청을 식별하는 고유 ID
- `method`: 호출할 메서드 이름 (예: `tools/call`)
- `params`: 메서드에 전달할 파라미터

요청의 목적은 항상 “Server의 Capability 실행”이다.

### 3.2 Response (응답)

Server -> Client로 보내는 메시지로, Request에 대한 **결과 또는 실패 정보**를 담는다.

응답 메시지의 핵심 특징:
- Request와 **동일한 `id`**를 포함
- 성공 시 `result`
- 실패 시 `error`

이 구조 덕분에 Client는:
- 여러 요청을 동시에 보내더라도
- 어떤 응답이 어떤 요청에 대응하는지 정확히 추적할 수 있다.

### 3.3 Notification (알림)

**응답이 필요 없는 단방향 메시지**다.

특징:
- `id`가 없음
- 주로 Server -> Client 방향
- 상태 업데이트, 진행 상황(progress) 전달에 사용

예:
- “현재 처리 중”
- “50% 완료”
- “백그라운드 작업 상태 변경”

Notification은 장시간 실행되는 Tool이나 스트리밍 시나리오에서 특히 중요하다.

## 4. 메시지 포맷과 역할 분리의 의미

이 세 가지 메시지 타입(Request / Response / Notification)을 명확히 분리함으로써 MCP는 다음을 달성한다.

- 동기 요청–응답 패턴 지원
- 비동기 진행 상황 보고 가능
- 네트워크/로컬 환경 모두에서 동일한 통신 모델 유지

즉, MCP는 “AI 전용 특수 프로토콜”이 아니라, **기존 분산 시스템의 통신 원칙을 그대로 계승**한다.

## 5. Transport Mechanism (전송 방식)

JSON-RPC는 **메시지 형식**만 정의한다.  
MCP는 여기에 더해, 메시지를 **어떻게 실제로 전달할지(Transport)**도 표준화한다.

MCP에서 지원하는 대표적 전송 방식은 두 가지다.

### 5.1 stdio (Standard Input / Output)

#### 개념
- Client와 Server가 **같은 머신**에서 실행될 때 사용
- Host가 Server를 **서브프로세스**로 실행
- stdin/stdout을 통해 JSON-RPC 메시지 교환

#### 주요 사용 사례
- 로컬 파일 시스템 접근
- 로컬 스크립트 실행
- 개발 도구용 로컬 MCP Server

#### 장점
- 네트워크 설정 불필요
- 구현 단순
- OS 수준 샌드박싱으로 비교적 안전

stdio는 “로컬 MCP 서버”를 매우 가볍게 구성할 수 있게 해준다.

### 5.2 HTTP + SSE (Server-Sent Events) / Streamable HTTP

#### 개념
- Client와 Server가 **서로 다른 머신**에 있을 때 사용
- HTTP 기반 요청/응답
- Server는 **SSE(Server-Sent Events)**를 통해 지속적인 업데이트를 푸시

#### 주요 사용 사례
- 원격 API
- 클라우드 서비스
- 공유 리소스 접근

#### 장점
- 네트워크 환경 전반에서 사용 가능
- 웹 서비스와 자연스럽게 통합
- 서버리스 환경과 호환 가능

#### Streamable HTTP
최근 MCP 표준에서는 “Streamable HTTP” 개념이 도입/정교화되었다.

- 기본은 일반 HTTP
- 필요 시 SSE로 **동적으로 업그레이드**
- 스트리밍과 서버리스 호환성을 동시에 만족

이는 “항상 스트리밍”도, “완전 단발 HTTP”도 아닌 **유연한 중간 해법**이다.

## 6. MCP 상호작용 라이프사이클 (Interaction Lifecycle)

MCP는 Client–Server 간 상호작용을 **단계별 라이프사이클**로 정의한다.

### 6.1 Initialization (초기화)

목적:
- 프로토콜 버전 협상
- Server가 제공하는 Capability 범위 확인

흐름:
1. Client -> `initialize` 요청
2. Server -> 지원 버전 및 Capability 정보 응답
3. Client -> `initialized` Notification 전송

이 단계가 끝나야 실제 Capability 호출이 가능하다.

### 6.2 Discovery (기능 탐색)

목적:
- Server가 제공하는 기능을 동적으로 파악

흐름:
1. Client -> `tools/list` (또는 resources/prompts 관련 요청)
2. Server -> 사용 가능한 Capability 목록 응답

Discovery 덕분에 Client는:
- Server마다 다른 기능 구성을
- 사전 하드코딩 없이 처리할 수 있다.

### 6.3 Execution (실행)

목적:
- 실제 Capability 호출

흐름:
1. Client -> `tools/call` 등 실행 요청
2. (선택적) Server -> Notification으로 진행 상황 전달
3. Server -> 최종 Response 반환

이 단계에서 MCP의 Tool / Resource / Prompt / Sampling이 실제로 사용된다.

### 6.4 Termination (종료)

목적:
- 연결을 안전하게 종료

흐름:
1. Client -> `shutdown` 요청
2. Server -> 응답
3. Client -> `exit` Notification

명시적인 종료 절차를 둠으로써:
- 리소스 누수 방지
- 프로세스/연결 정리 보장

## 7. 프로토콜 진화(Evolution) 전략

MCP는 “한 번 정하고 끝”인 프로토콜이 아니다.

### 핵심 설계 포인트
- **버전 협상**: 초기화 단계에서 상호 지원 버전 확인
- **Capability 기반 적응**: Server가 제공하는 기능만 사용
- **후방 호환성**: 구버전 Client/Server와 공존 가능

이 설계 덕분에:
- 단순한 Server와
- 고급 기능을 가진 Server가
같은 MCP 생태계 안에서 동시에 동작할 수 있다.

## 8. 이 섹션의 핵심 요약

- MCP는 통신 표준으로 **JSON-RPC 2.0**을 사용한다.
- 메시지는 Request / Response / Notification으로 구분된다.
- 전송 방식은 로컬(stdio)과 원격(HTTP+SSE)을 모두 지원한다.
- 통신은 Initialization -> Discovery -> Execution -> Termination의 명확한 라이프사이클을 따른다.
- 버전 협상과 Capability Discovery를 통해 **확장성과 장기 진화**를 보장한다.

이 통신 프로토콜 덕분에 MCP는 “이론적 표준”이 아니라,
실제로 **다양한 환경에서 안정적으로 동작하는 AI 연결 표준**이 된다.

참고자료
Huggingface, mcp course, https://huggingface.co/learn