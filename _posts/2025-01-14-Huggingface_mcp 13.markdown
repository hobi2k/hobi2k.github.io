---
layout: post
title:  "허깅페이스 MCP 코스 - Gradio를 MCP Client로 사용하는 방법 정리"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Gradio를 MCP Client로 사용하는 방법 정리

이 글에서는 **Gradio를 MCP Server가 아니라 MCP Client로 사용하는 패턴**을 정리한다.  
핵심은 Gradio를 **UI 레이어**, MCP Client를 **원격 MCP Server와의 연결 레이어**,  
그리고 LLM + Agent를 **추론 및 도구 오케스트레이션 레이어**로 분리하는 것이다.

즉, 구조적으로는 다음과 같다.

> Gradio(UI) -> MCP Client -> MCP Server(Tools) -> 결과 -> LLM -> UI

## 1. Gradio를 MCP Client로 쓰는 이유

Gradio는 기본적으로:
- 웹 UI 제작
- MCP Server 제작

에 가장 최적화된 도구다.  
하지만 다음 상황에서는 **MCP Client로서도 매우 실용적**이다.

- 이미 존재하는 MCP Server를 **소비(consumption)**하고 싶을 때
- MCP Tool을 사용하는 **에이전트 데모 UI**를 만들고 싶을 때
- IDE나 CLI가 아닌 **웹 기반 인터페이스**가 필요할 때

이 패턴은 특히:
- Tool 기반 Agent 실험
- 비개발자 대상 데모
- MCP Tool UX 검증
에 적합하다.

## 2. 전체 흐름 개요

Gradio MCP Client 패턴은 항상 다음 단계를 따른다.

1. MCP Client로 원격 MCP Server에 연결
2. Server가 제공하는 Tool 목록을 Discovery
3. Tool을 Agent에 주입
4. Agent를 Gradio Chat UI로 감싸기
5. 사용자 입력 -> Agent -> MCP Tool 호출 -> 결과 반환

## 3. 예제 MCP Server 연결

### 3.1 사용 예제 서버

이번 예제에서는 Hugging Face Spaces에 이미 배포된 MCP Server를 사용한다.

- MCP Server:
https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse

이 서버는 다음과 같은 MCP Tool들을 제공한다.

- 정수 소인수 분해
- 이미지 생성
- 이미지 방향 판별
- 이미지 필터 적용

### 3.2 MCP Client로 Tool 목록 확인

```python
from smolagents.mcp_client import MCPClient

with MCPClient(
  {
      "url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse",
      "transport": "sse",
  }
) as tools:
  print("\n".join(f"{t.name}: {t.description}" for t in tools))
```

이 코드는:

- MCP Server에 연결
- tools/list Discovery 수행
- Tool 이름과 설명을 출력

즉, MCP Client는 Server의 Capability를 사전 정보 없이 동적으로 탐색한다.

## 4. Gradio + MCP Client + Agent 구성
이제 MCP Tool을 실제로 사용하는 Gradio 애플리케이션을 만든다.

### 4.1 의존성 설치
```bash
pip install "smolagents[mcp]" "gradio[mcp]" mcp fastmcp
```

### 4.2 기본 임포트
```python
import gradio as gr
import os

from smolagents import InferenceClientModel, CodeAgent, MCPClient
```

### 4.3 MCP Client 생성 및 Tool 수집
```python
mcp_client = MCPClient(
    {
        "url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse",
        "transport": "sse",
    }
)

tools = mcp_client.get_tools()
```

get_tools() 호출 시:

- MCP 초기화
- Capability Discovery
- Tool 메타데이터 수집

### 4.4 LLM 모델 준비
```python
model = InferenceClientModel(token=os.getenv("HF_TOKEN"))
```

중요 사항:

- Hugging Face Inference API 사용
- HF_TOKEN 환경 변수 필요
- 토큰에 Inference Provider 호출 권한이 있어야 함

### 4.5 Agent 생성
```python
agent = CodeAgent(
    tools=[*tools],
    model=model
)
```

이 Agent는:

- 사용자 질의를 분석
- Tool 사용 여부 판단
- MCP Tool 호출 및 결과 반영
- 최종 응답 생성

## 5. Gradio Chat UI로 노출
```python
demo = gr.ChatInterface(
    fn=lambda message, history: str(agent.run(message)),
    type="messages",
    examples=["Prime factorization of 68"],
    title="Agent with MCP Tools",
    description="This is a simple agent that uses MCP tools to answer questions."
)

demo.launch()
```

이제 완성된 구성은 다음과 같다.

- 프론트엔드: Gradio Chat UI
- 추론 엔진: LLM + CodeAgent
- 도구 실행: 원격 MCP Server

## 6. 전체 예제 코드
```python
import gradio as gr
import os

from smolagents import InferenceClientModel, CodeAgent, MCPClient

try:
    mcp_client = MCPClient(
        {
            "url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse",
            "transport": "sse",
        }
    )
    tools = mcp_client.get_tools()

    model = InferenceClientModel(
        token=os.getenv("HUGGINGFACE_API_TOKEN")
    )

    agent = CodeAgent(
        tools=[*tools],
        model=model,
        additional_authorized_imports=["json", "ast", "urllib", "base64"]
    )

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    demo.launch()
finally:
    mcp_client.disconnect()
```

중요한 포인트

- finally 블록에서 disconnect() 호출
- MCP Client는 지속 연결 객체
- 종료 시 명시적 정리가 필수

## 7. Hugging Face Spaces에 배포
Gradio MCP Client 또한 Spaces에 그대로 배포 가능하다.

### 7.1 Space 생성
- SDK: Gradio
- 예시 이름: mcp-client

### 7.2 MCP Server URL 확인
```python
mcp_client = MCPClient(
    {
        "url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse",
        "transport": "sse",
    }
)
```

### 7.3 requirements.txt
```
gradio[mcp]
smolagents[mcp]
```

### 7.4 배포
```bash
git init
git add app.py requirements.txt
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/mcp-client
git push -u origin main
```

## 8. 핵심 요약
- Gradio는 MCP Client 역할도 수행 가능
- 기존 MCP Server를 그대로 소비 가능
- MCP Tool + Agent + 웹 UI를 빠르게 결합
- Tool 기반 Agent 데모 제작에 최적
- Hugging Face Spaces로 손쉬운 공유 가능

참고자료
Huggingface, mcp course, https://huggingface.co/learn