---
layout: post
title:  "허깅페이스 MCP 코스 - Gradio MCP Server부터 End-to-End Agent까지"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Gradio MCP Server부터 End-to-End Agent까지

이 글에서는 지금까지 만든 모든 구성요소를 하나의 엔드투엔드 MCP 애플리케이션으로 연결한다.
핵심 목표는 다음이다.

- Gradio로 만든 MCP Server(감정 분석 도구)를
- Tiny Agents 기반 Agent가 MCP Client로 연결하여
- LLM이 Tool을 자연스럽게 사용하도록 만드는 것

## 1. Tiny Agents란 무엇인가

Tiny Agents는 Hugging Face에서 제안한 초경량 Agent 런타임이다.

특징을 정리하면:

- JSON 설정 파일만으로 Agent 정의
- MCP Server를 “플러그인”처럼 연결
- Python / TypeScript 동일한 개념
- CLI 기반 실행 (배포·자동화 친화적)
- Hugging Face Hub 및 로컬 모델 모두 지원

즉, Tiny Agents는 “MCP Client를 가장 간단하게 실행하는 방법”이다.

## 2. 전체 아키텍처 복기

지금까지 완성한 구조는 아래와 같다.

```
[사용자]
   ↓
[Tiny Agent]
   ↓  (MCP Client)
[MCP Server 목록]
   ├─ Filesystem / Browser / 기타 MCP
   └─ Gradio Sentiment MCP Server
   ↓
[LLM (Qwen 등)]
```

Agent는:

- 여러 MCP Server를 동시에 연결
- Tool을 동적으로 discovery
- 상황에 맞게 Tool 호출
- 결과를 종합해 응답 생성

## 3. 설치 및 사전 준비
### 3.1 Node.js / npx

Tiny Agents는 MCP Server 실행을 위해 npx를 사용한다.

```bash
npm install -g npx
npm install mcp-remote
```

이유

일부 MCP Client(Claude Desktop 등)는
SSE 기반 MCP Server를 직접 지원하지 않음
-> mcp-remote가 stdio ↔ SSE 브리지 역할 수행

### 3.2 언어별 패키지 설치
JavaScript / TypeScript

```bash
npm install @huggingface/tiny-agents
```

```Python
pip install "huggingface_hub[mcp]>=0.32.0"
```

## 4. Tiny Agents 기본 MCP Client (CLI)

Tiny Agents는 코드 없이 JSON만으로 MCP Client를 생성할 수 있다.

### 4.1 프로젝트 구조
```bash
mkdir my-agent
cd my-agent
touch agent.json
```

### 4.2 agent.json (Gradio MCP Server 연결)
```json
{
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "provider": "nebius",
  "servers": [
    {
      "type": "stdio",
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:7860/gradio_api/mcp/sse"
      ]
    }
  ]
}
```

필드 의미


| 필드       | 설명                |
| -------- | ----------------- |
| model    | Agent가 사용할 LLM    |
| provider | 추론 제공자 (Nebius 등) |
| servers  | 연결할 MCP Server 목록 |


### 4.3 실행
JavaScript

```bash
npx @huggingface/tiny-agents run agent.json
```

Python

```bash
tiny-agents run agent.json
```

이 시점에서 Agent는:

- Gradio MCP Server에 연결
- 감정 분석 Tool을 discovery
- 프롬프트에 따라 Tool 사용 가능

## 5. 로컬 모델 + Tiny Agents (확장 패턴)

Tiny Agents는 로컬 OpenAI-compatible 서버도 지원한다.

예시:

```json
{
  "model": "Qwen/Qwen3-32B",
  "endpointUrl": "http://localhost:1234/v1",
  "servers": [
    {
      "type": "stdio",
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:1234/v1/mcp/sse"
      ]
    }
  ]
}
```

-> 로컬 모델 + 로컬 MCP Server + 완전 오프라인 Agent 가능

## 6. 코드 기반 Custom Tiny Agent

CLI를 넘어서 코드로 Agent를 구성할 수도 있다.

### 6.1 TypeScript 예제

```ts
const agent = new Agent({
  provider: process.env.PROVIDER ?? "nebius",
  model: process.env.MODEL_ID ?? "Qwen/Qwen2.5-72B-Instruct",
  apiKey: process.env.HF_TOKEN,
  servers: [
    {
      command: "npx",
      args: [
        "mcp-remote",
        "http://localhost:7860/gradio_api/mcp/sse"
      ]
    }
  ],
});
```

### 6.2 Python 예제

```python
from huggingface_hub import Agent

agent = Agent(
    model="Qwen/Qwen2.5-72B-Instruct",
    provider="nebius",
    servers=[
        {
            "command": "npx",
            "args": [
                "mcp-remote",
                "http://localhost:7860/gradio_api/mcp/sse"
            ]
        }
    ],
)
```

## 7. Hugging Face Spaces 배포 후 연결

Gradio MCP Server를 Spaces에 배포했다면,
Agent 설정의 URL만 바꾸면 된다.

```json
{
  "command": "npx",
  "args": [
    "mcp-remote",
    "https://YOUR_USERNAME-mcp-sentiment.hf.space/gradio_api/mcp/sse"
  ]
}
```

전 세계 어디서든 동일한 Agent 사용 가능

## 8. 완성된 End-to-End MCP 흐름

이번 Unit에서 완성한 전체 파이프라인은 다음과 같다.

1. Gradio로 MCP Server 제작
2. 감정 분석 Tool 노출
3. Tiny Agents로 MCP Client 생성
4. Agent가 Tool을 동적으로 탐색
5. LLM이 Tool을 선택·호출
6. 결과를 조합해 응답 생성

즉, “도구는 분리하고, 연결은 표준화하고, Agent는 가볍게”
라는 MCP 철학이 그대로 구현된 사례다.

## 9. 핵심 정리

- Tiny Agents는 가장 단순한 MCP Client 런타임
- Gradio MCP Server와 궁합이 매우 좋음
- CLI / Python / TypeScript 모두 동일한 개념
- 로컬·호스티드 모델 모두 사용 가능
- MCP의 “도구 분리 + 조합” 철학을 가장 잘 보여주는 예시

참고자료
Huggingface, mcp course, https://huggingface.co/learn