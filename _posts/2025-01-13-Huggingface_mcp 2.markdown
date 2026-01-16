---
layout: post
title:  "허깅페이스 MCP 코스 - MCP(Model Context Protocol) 아키텍처 구성요소 정리"
date:   2025-01-13 00:10:22 +0900
categories: Huggingface_mcp
---

# MCP(Model Context Protocol) 아키텍처 구성요소 정리

앞선 글에서 MCP의 개념과 용어를 다뤘다면, 이 글은 **MCP가 실제로 어떻게 구성되고 동작하는지**를 설명하는 아키텍처 중심 내용이다.  
MCP의 핵심은 “AI 애플리케이션과 외부 시스템을 어떻게 분리·연결할 것인가”에 대한 명확한 역할 정의에 있다.

## 1. MCP 아키텍처 개요

MCP(Model Context Protocol)는 **클라이언트–서버(Client–Server) 아키텍처**를 기반으로 설계되었다.  
이 구조를 통해 AI 모델(LLM)과 외부 도구·데이터·서비스 간의 통신을 **표준화되고 예측 가능하게** 만든다.

MCP 아키텍처는 다음 세 가지 핵심 컴포넌트로 구성된다.

- **Host**
- **Client**
- **Server**

각 구성요소는 책임이 명확히 분리되어 있으며, 이 분리가 MCP 확장성과 유지보수성을 떠받치는 핵심 설계 원칙이다.

## 2. Host / Client / Server 역할 상세

### 2.1 Host

**Host**는 사용자가 직접 상호작용하는 **사용자-facing AI 애플리케이션**이다.  
즉, MCP 관점에서 Host는 “AI 앱 그 자체”다.

#### Host의 대표적 예시
- AI 채팅 애플리케이션 (예: ChatGPT, Claude Desktop 계열)
- AI 기반 IDE 및 개발 도구 (예: Cursor, Continue.dev 연동 IDE)
- LangChain, smolagents 등으로 구축된 커스텀 AI 에이전트/앱

#### Host의 핵심 책임
- 사용자 입력 및 인터랙션 관리
- 사용자 권한 및 승인 관리
- LLM 호출 및 프롬프트 구성
- MCP Client를 통해 MCP Server 연결 시작
- 전체 처리 흐름(오케스트레이션) 제어
- 최종 결과를 사용자에게 이해 가능한 형태로 렌더링

중요한 점은, **사용자는 Host를 기준으로 MCP 생태계에 진입**한다는 것이다.  
즉, MCP는 “사용자가 선택한 Host를 중심으로 외부 기능을 확장하는 구조”를 취한다.

### 2.2 Client

**Client**는 Host 내부에 존재하는 **MCP 통신 전담 컴포넌트**다.  
Client는 “앱의 일부”이지, 독립적인 사용자-facing 요소가 아니다.

#### Client의 구조적 특징
- **Client <-> Server는 1:1 연결**
- 하나의 Client는 하나의 MCP Server만 담당
- 여러 Server를 쓰려면 여러 Client 인스턴스가 존재할 수 있음

#### Client의 책임
- MCP 프로토콜 레벨 통신 처리
- Server와의 연결 관리
- Capability 탐색(Discover)
- Capability 호출 요청 전달
- Server 응답 수신 및 Host로 전달

즉, Client는 **Host의 비즈니스 로직과 Server의 실제 기능 구현 사이를 잇는 중간 계층**이다.

> 주의  
> 많은 자료에서 Host와 Client를 혼용하지만,  
> - Host = 사용자-facing 애플리케이션  
> - Client = MCP 통신 담당 모듈  
> 로 구분하는 것이 MCP 설계 이해에 필수적이다.

### 2.3 Server

**Server**는 MCP 프로토콜을 통해 기능을 제공하는 **외부 프로그램 또는 서비스**다.

#### Server의 성격
- 기존 기능(툴, DB, API 등)을 감싸는 **경량 래퍼**
- MCP 표준 인터페이스를 통해 기능을 노출
- Host와 직접 연결되지 않고 Client를 통해서만 접근됨

#### Server의 주요 특징
- 로컬 실행 가능 (Host와 동일 머신)
- 원격 실행 가능 (네트워크 너머 서비스)
- 기능을 표준화된 형식으로 노출
- Client가 동적으로 기능을 탐색(discover) 가능

Server는 “AI 전용으로 새로 만든 시스템”일 필요는 없으며,  
기존 시스템을 MCP Server로 **감싸서 재사용**하는 것이 일반적인 사용 패턴이다.

## 3. MCP 통신 흐름 (Communication Flow)

MCP 아키텍처는 단순한 구조를 가지지만, 그 위에서 일어나는 상호작용 흐름은 매우 체계적이다.  
일반적인 MCP 워크플로우는 다음 단계로 진행된다.

### 1. 사용자 입력
- 사용자가 Host 애플리케이션에 질문, 명령, 의도를 입력한다.

### 2. Host 처리
- Host는 입력을 분석한다.
- 필요 시 LLM을 호출해 사용자의 의도(intent)를 해석한다.
- 외부 기능이 필요한지 판단한다.

### 3. Client 연결
- Host는 적절한 MCP Client를 통해 특정 Server에 연결한다.

### 4. Capability 탐색 (Discovery)
- Client는 Server에 질의하여,
  - 어떤 Tools
  - 어떤 Resources
  - 어떤 Prompts
를 제공하는지 확인한다.

### 5. Capability 호출
- Host(또는 LLM의 판단)에 따라
- Client가 특정 Capability 실행을 Server에 요청한다.

### 6. Server 실행
- Server는 요청된 기능을 실행한다.
- 실행 결과를 Client에 반환한다.

### 7. 결과 통합
- Client는 결과를 Host로 전달한다.
- Host는 결과를
  - LLM 컨텍스트에 반영하거나
  - 사용자에게 직접 출력한다.

## 4. MCP 아키텍처의 핵심 장점: 모듈성(Modularity)

이 구조의 가장 큰 장점은 **강력한 모듈성**이다.

- 하나의 Host는 여러 MCP Server에 동시에 연결 가능
- Server 추가 시 기존 Host 수정 불필요
- 서로 다른 Server의 Capability를 조합해 사용 가능
- 기능 단위 확장이 매우 쉬움

이 모듈성 덕분에 MCP는 기존의 **M×N 통합 문제**를,
- Host는 한 번,
- Server는 한 번만
MCP 표준을 구현하면 되는 **M+N 구조**로 전환한다.

## 5. 설계 철학 및 원칙

MCP 아키텍처는 다음 원칙을 중심으로 설계되었다.

- **표준화(Standardization)**  
  AI 애플리케이션과 외부 시스템 연결을 위한 공통 규약 제공

- **단순성(Simplicity)**  
  핵심 프로토콜은 단순하게 유지하되, 고급 기능 확장을 허용

- **안전성(Safety)**  
  민감한 작업은 사용자 명시적 승인 요구

- **발견 가능성(Discoverability)**  
  Client가 Server의 Capability를 동적으로 탐색 가능

- **확장성(Extensibility)**  
  버전 관리와 Capability 협상을 통한 점진적 진화 가능

- **상호운용성(Interoperability)**  
  다양한 구현체·환경 간 호환 보장

## 6. 정리

- MCP 아키텍처는 Host / Client / Server의 명확한 역할 분리를 기반으로 한다.
- Host는 사용자 경험과 오케스트레이션을 담당한다.
- Client는 MCP 통신의 단일 책임자다.
- Server는 외부 기능을 표준 인터페이스로 제공한다.
- 이 구조는 확장성, 재사용성, 유지보수성을 동시에 확보한다.

이 아키텍처 위에서, 다음 단계에서는 **실제 MCP 통신 프로토콜**이 어떻게 설계되고 동작하는지를 다루게 된다.


참고자료
Huggingface, mcp course, https://huggingface.co/learn