---
layout: post
title:  "허깅페이스 MCP 코스 - MCP SDK 정리: 공식 SDK로 Client·Server 구현하기"
date:   2025-01-13 00:10:22 +0900
categories: Huggingface_mcp
---

# MCP SDK 정리: 공식 SDK로 Client·Server 구현하기

이 글은 MCP(Model Context Protocol)를 **이론이 아니라 실제 코드로 구현하는 단계**를 다룬다.  
MCP SDK의 핵심 목적은 “프로토콜의 복잡함은 SDK가 감추고, 개발자는 Capability 구현에만 집중하게 한다”는 데 있다.

## 1. MCP SDK의 역할과 의의

MCP는 표준 프로토콜이기 때문에, 이를 직접 구현하려면 다음과 같은 저수준 작업이 필요하다.

- JSON-RPC 메시지 포맷 구성
- 요청–응답 매칭(id 관리)
- Capability 등록 및 탐색
- Transport(stdio / HTTP+SSE) 처리
- 오류 및 연결 상태 관리

**MCP SDK**는 이러한 저수준 구현을 모두 캡슐화한다.  
개발자는 SDK가 제공하는 고수준 API를 사용해 다음에만 집중하면 된다.

- 어떤 Tool을 제공할지
- 어떤 Resource를 노출할지
- 어떤 Prompt를 정의할지
- (필요하다면) Sampling을 어떻게 활용할지

## 2. SDK 공통 기능 개요

공식 MCP SDK(JavaScript, Python 등)는 언어는 달라도 **동일한 핵심 기능 집합**을 제공한다.

SDK가 담당하는 영역:
- MCP 프로토콜 레벨 통신
- Capability 등록 및 Discovery
- 메시지 직렬화/역직렬화(JSON-RPC)
- 연결 관리(stdio, HTTP 등)
- 에러 처리 및 상태 관리

결과적으로 MCP SDK는 **“프로토콜 구현체”이자 “개발자 경험(DX) 레이어”**다.

## 3. MCP Server 구현 개념

SDK를 사용한 MCP Server 구현의 기본 구조는 다음과 같다.

1. MCP Server 인스턴스 생성
2. Tool / Resource / Prompt 등록
3. Transport를 통해 서버 실행
4. Inspector 또는 Client로 연결해 테스트

아래 예시는 “날씨 서비스”라는 단순한 MCP Server를 구현한 것이다.

## 4. Python SDK 예제 해설

### 4.1 Server 생성

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Service")
```

- FastMCP는 Python SDK에서 제공하는 고수준 Server 래퍼
- 서버 이름은 Capability Discovery 시 식별 정보로 사용됨

### 4.2 Tool 구현
```python
@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    return f"Weather in {location}: Sunny, 72°F"
```

- @mcp.tool() 데코레이터로 Tool 등록
- 함수 시그니처가 Tool의 입력 스펙이 됨
- LLM이 “행동”으로 호출 가능한 Capability

### 4.3 Resource 구현
```python
@mcp.resource("weather://{location}")
def weather_resource(location: str) -> str:
    """Provide weather data as a resource."""
    return f"Weather data for {location}: Sunny, 72°F"
```

- URI 템플릿 기반 Resource 정의
- 읽기 전용 컨텍스트 제공
- Tool보다 안전한 Capability

### 4.4 Prompt 구현
```python
@mcp.prompt()
def weather_report(location: str) -> str:
    """Create a weather report prompt."""
    return f"You are a weather reporter. Weather report for {location}?"
```

- 사전 정의된 프롬프트 템플릿
- 사용자 또는 Host가 선택해 사용
- “어떻게 사고할지”를 규정하는 역할

### 4.5 Server 실행
```bash
mcp dev server.py
```
- 개발 모드에서 MCP Server 실행
- 내부적으로 stdio transport 사용
- Inspector와 연동 가능

## 5. JavaScript SDK 예제 해설
JavaScript SDK는 Node.js 환경을 기준으로 설계되어 있으며, 구조는 Python과 동일하되 문법만 다르다.

### 5.1 Server 생성
```javascript
const server = new McpServer({
  name: "Weather Service",
  version: "1.0.0",
});
```
- 서버 이름과 버전 명시
- 버전은 프로토콜 진화 시 중요

### 5.2 Tool 구현
```javascript
server.tool(
  "get_weather",
  { location: z.string() },
  async ({ location }) => ({
    content: [{ type: "text", text: `Weather in ${location}: Sunny, 72°F` }],
  })
);
```
- 입력 스키마를 zod로 명시
- 반환값은 MCP 메시지 포맷에 맞춰 구성

### 5.3 Resource 구현
```javascript
server.resource(
  "weather",
  new ResourceTemplate("weather://{location}", { list: undefined }),
  async (uri, { location }) => ({
    contents: [
      { uri: uri.href, text: `Weather data for ${location}: Sunny, 72°F` },
    ],
  })
);
```

- ResourceTemplate로 URI 패턴 정의
- Discovery 시 자동 노출됨

### 5.4 Prompt 구현
```javascript
server.prompt(
  "weather_report",
  { location: z.string() },
  async ({ location }) => ({
    messages: [
      { role: "assistant", content: { type: "text", text: "You are a weather reporter." } },
      { role: "user", content: { type: "text", text: `Weather report for ${location}?` } },
    ],
  })
);
```

- Prompt는 메시지 배열 형태로 정의
- 멀티 메시지 컨텍스트 구성 가능

### 5.5 Server 실행
```bash
npx @modelcontextprotocol/inspector node ./index.mjs
```

- Inspector를 통해 서버 실행
- stdio transport 기반 연결

## 6. MCP Inspector의 역할

서버를 실행하면 MCP Inspector가 자동으로 연결된다.

Inspector의 기능:

- Server가 노출한 Capability 시각화
- Tool / Resource / Prompt 직접 호출
- 요청·응답 디버깅

이는 MCP 개발에서 매우 중요한 도구로,

- Server 구현 검증
- Capability 설계 점검

에 사용된다.

## 7. 공식 MCP SDK 생태계
MCP는 언어 중립적이며, 공식·커뮤니티 SDK가 다수 제공된다.

주요 SDK 현황 요약:

- TypeScript / Python: Anthropic 유지보수, 핵심 SDK
- Java / Kotlin: Spring AI, JetBrains 주도
- C#: Microsoft (프리뷰)
- Swift / Rust / Dart: 커뮤니티 중심 확장

이로 인해 MCP는:

- 웹
- 백엔드
- 데스크톱
- 모바일

환경 전반에서 동일한 개념으로 적용 가능하다.

## 8. MCP SDK 사용의 핵심 가치
- 프로토콜 구현 부담 제거
- Capability 중심 개발 가능
- 언어별로 동일한 추상화 제공
- Inspector 기반 빠른 실험 가능

즉, MCP SDK는 **“표준을 현실적인 개발 경험으로 바꾸는 계층”**이다.

참고자료
Huggingface, mcp course, https://huggingface.co/learn