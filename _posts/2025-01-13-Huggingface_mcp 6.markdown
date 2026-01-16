---
layout: post
title:  "허깅페이스 MCP 코스 - MCP Clients 정리: Host와 Server를 잇는 실질적 연결 계층"
date:   2025-01-13 00:10:22 +0900
categories: Huggingface_mcp
---

# MCP Clients 정리: Host와 Server를 잇는 실질적 연결 계층

이 글은 MCP(Model Context Protocol) 생태계에서 **MCP Client가 어떤 역할을 수행하며, 실제로 어떻게 설정·사용되는지**를 다룬다.  
앞서 MCP Server와 Capability를 살펴봤다면, 이제 “그 기능을 누가, 어떻게 쓰게 만드는가”가 핵심 주제다.

## 1. MCP Client란 무엇인가

**MCP Client**는 AI 애플리케이션(Host) 내부에 존재하며,  
Host와 MCP Server 사이의 **모든 통신을 전담하는 연결 계층**이다.

개념적으로 보면:
- **Host**: 사용자-facing AI 애플리케이션
- **Server**: 외부 기능(Tools, Resources, Prompts, Sampling)을 제공
- **Client**: Host 내부에서 Server와 MCP 프로토콜로 대화하는 전용 모듈

즉, Client는 “보이지 않지만 필수적인 브리지”다.

## 2. MCP Client의 핵심 책임

MCP Client는 단순한 네트워크 커넥터가 아니다. 다음 책임을 가진다.

- MCP Server와의 연결 관리
- 초기화 및 버전 협상
- Capability Discovery (tools/list, resources/list, prompts/list)
- JSON-RPC 요청·응답 처리
- Server에서 오는 Notification 처리
- Host와 Server 사이의 데이터 중계

중요한 점은, **하나의 Client는 하나의 Server와 1:1로 연결**된다는 것이다.  
여러 Server를 쓰려면, Host 내부에 여러 Client 인스턴스가 존재한다.

## 3. MCP Client의 주요 유형

### 3.1 사용자 인터페이스(UI) 기반 Client

사용자가 직접 사용하는 AI 도구에 **내장된 MCP Client**다.

#### Chat Interface Clients
- Claude Desktop (Anthropic)

#### Interactive Development Clients
- Cursor IDE (MCP Client 내장)
- VS Code 확장 (예: Continue)
- Zed Editor

이들은 다음 특징을 가진다.
- 여러 MCP Server 동시 연결 가능
- 실시간 Tool 호출
- UI 기반 승인(특히 Tool, Sampling)

## 4. Hugging Face MCP Server 빠른 연결

Hugging Face는 **호스팅된 MCP Server**를 제공하며,  
모델·데이터셋·Spaces·논문 탐색용 Tool을 기본 제공한다.

### 연결 절차
1. https://huggingface.co/settings/mcp 접속
2. 사용 중인 MCP Client 선택 (VS Code, Cursor, Zed, Claude Desktop 등)
3. 자동 생성된 설정 스니펫 복사
4. Client 설정에 붙여넣기 후 재시작

> 팁  
> 수동 설정 대신 **자동 생성된 스니펫 사용 권장**  
> (Client별 세부 설정이 반영되어 있음)

연결이 완료되면 Client에 “Hugging Face” MCP Server가 표시된다.

## 5. MCP Client 설정 방식

### 5.1 `mcp.json` 개요

대부분의 MCP Host는 `mcp.json`이라는 **공통 설정 파일 포맷**을 사용한다.

```json
{
  "servers": [
    {
      "name": "Server Name",
      "transport": {
        "type": "stdio | sse"
      }
    }
  ]
}
```

핵심 요소:

- servers: 연결할 MCP Server 목록
- name: UI 및 내부 식별용 이름
- transport: 통신 방식 정의

### 5.2 stdio Transport 설정 (로컬 서버)
로컬 MCP Server를 서브프로세스로 실행할 때 사용한다.

```json
{
  "servers": [
    {
      "name": "File Explorer",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/file_explorer_server.py"]
      }
    }
  ]
}
```

특징:

- Host가 Server 프로세스를 직접 실행
- stdin/stdout 기반 통신
- 로컬 툴, 파일 접근, 개발용 서버에 적합

### 5.3 HTTP + SSE Transport 설정 (원격 서버)
원격 MCP Server에 연결할 때 사용한다.

```json
{
  "servers": [
    {
      "name": "Weather API",
      "transport": {
        "type": "sse",
        "url": "https://example.com/mcp-server"
      }
    }
  ]
}
```

특징:

- 네트워크 기반 연결
- 클라우드 API, 원격 서비스에 적합

## 6. 환경 변수(Environment Variables) 설정
MCP Server 실행 시 **비밀 정보(API Key, Token)**는 환경 변수로 전달하는 것이 원칙이다.

### mcp.json에서 환경 변수 지정
```json
{
  "servers": [
    {
      "name": "GitHub API",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/github_server.py"],
        "env": {
          "GITHUB_TOKEN": "your_github_token"
        }
      }
    }
  ]
}
```

### Server 코드에서 접근

- Python: os.environ.get("GITHUB_TOKEN")
- JavaScript: process.env.GITHUB_TOKEN

이 방식은:

- 설정과 코드 분리
- 보안 강화
- 배포 환경별 설정 용이

라는 장점을 가진다.

## 7. 코드 기반 MCP Client: Tiny Agents
UI 기반 Client 외에도, 코드로 직접 MCP Client를 사용하는 방식이 있다.
대표적인 예가 Hugging Face의 tiny-agents다.

Tiny Agents는:

- LLM + MCP Server를 결합한 경량 에이전트 런타임
- MCP Server를 CLI 환경에서 실행
- Agent 설정 파일(JSON)로 전체 구성 정의

### 7.1 사전 준비

-npx 설치

```bash
npm install -g npx
```

- Python 환경

```bash
pip install "huggingface_hub[mcp]>=0.32.0"
huggingface-cli login
```

Hugging Face 토큰 권한에서
“Make calls to Inference Providers” 체크 필수

### 7.2 Tiny Agent 설정 예시 (Python / 공통 JSON)
```json
{
  "name": "playwright-agent",
  "description": "Agent with Playwright MCP server",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "provider": "nebius",
  "servers": [
    {
      "type": "stdio",
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    }
  ]
}
```

이 설정은:

- LLM: Qwen2.5-72B-Instruct
- MCP Server: Playwright (브라우저 제어)
- stdio 기반 실행

실행:

```bash
tiny-agents run agent.json
```

### 7.3 JavaScript Tiny Agents
```bash
npm install @huggingface/tiny-agents
```

```json
{
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "provider": "nebius",
  "servers": [
    {
      "type": "stdio",
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    }
  ]
}
```

```bash
npx @huggingface/tiny-agents run ./my-agent
```

이 방식은:

- UI 없이
- 코드/CLI 환경에서
- MCP 기반 에이전트를 빠르게 실험할 수 있게 해준다.

## 8. MCP Client 관점에서의 핵심 포인트 정리

- MCP Client는 Host 내부의 통신 전담 모듈
- Server와 1:1 관계
- 설정은 mcp.json으로 단순·표준화
- stdio(로컬)와 sse(원격) 모두 지원
- UI Client와 코드 기반 Client(tiny-agents) 모두 공존
- 실제 “에이전트 경험”은 Client 설계에 크게 좌우됨

참고자료
Huggingface, mcp course, https://huggingface.co/learn