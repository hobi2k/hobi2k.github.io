---
layout: post
title:  "허깅페이스 MCP 코스 - Gradio를 MCP Client로 사용하는 방법 정리"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Gradio를 MCP Client로 사용하는 방법 정리

앞선 글에서는 **Gradio로 MCP Server를 만드는 방법**을 다뤘다.  
이번 글의 핵심은 반대 방향이다.

**Gradio를 “MCP Client”로 사용해, 외부 MCP Server에 연결하고  그 위에 UI를 얹는 방법**을 정리한다.

즉, 이 글은 **“Gradio = UI + MCP Client 래퍼”**로 활용하는 패턴을 다룬다.

## 1. 왜 Gradio를 MCP Client로 쓰는가

Gradio는 본래:
- 웹 UI 제작 도구
- MCP Server 제작 도구

로 가장 적합하지만, 다음과 같은 경우에는 **MCP Client로도 매우 유용**하다.

- MCP Tool들을 사용하는 **에이전트 데모 UI**를 만들고 싶을 때
- CLI가 아닌 **웹 기반 Agent 인터페이스**가 필요할 때
- MCP Server를 직접 만들지 않고 **기존 MCP Server를 소비**하고 싶을 때

즉, Gradio를 MCP Client로 쓰는 패턴은:

> “원격 MCP Server + 로컬/웹 UI Agent”

구조를 빠르게 만들기 위한 선택지다.

## 2. Gradio MCP Client의 기본 구조

Gradio MCP Client 패턴은 다음 흐름을 따른다.

1. **MCP Client로 원격 MCP Server 연결**
2. Server가 제공하는 Tool 목록 가져오기
3. Tool들을 Agent에 주입
4. Agent를 Gradio UI(ChatInterface)로 감싸기

## 3. 예제 MCP Server에 연결하기

### 3.1 예제 서버

이번 예제에서는 Hugging Face Spaces에 이미 배포된 MCP Server를 사용한다.

- MCP Server URL  
https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse

이 서버는 여러 MCP Tool(수학, 이미지 생성 등)을 제공한다.

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

이 코드를 실행하면, 원격 MCP Server가 제공하는 Tool 목록이 출력된다.

핵심 포인트:

- MCP Client는 서버의 Capability를 동적으로 Discovery
- 사전 하드코딩 없이 Tool을 받아온다

## 4. Gradio + MCP Client + Agent 결합
이제 MCP Tool을 실제로 사용하는 에이전트 UI를 만든다.

### 4.1 의존성 설치
```bash
pip install "smolagents[mcp]" "gradio[mcp]" mcp fastmcp
```

### 4.2 기본 구성 요소
```python
import gradio as gr
import os

from smolagents import InferenceClientModel, CodeAgent, MCPClient
```

### 4.3 MCP Server 연결 및 Tool 수집
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

- MCP Discovery 실행
- Tool 메타데이터 + 호출 스펙 확보

### 4.4 LLM 모델 준비
```python
model = InferenceClientModel(
    token=os.getenv("HF_TOKEN")
)
```

중요:

- Hugging Face Inference API 사용
- HF_TOKEN 환경 변수 필요
- 토큰에는 Inference Provider 호출 권한이 있어야 함

### 4.5 Agent 생성
```python
agent = CodeAgent(
    tools=[*tools],
    model=model
)
```

이 Agent는:

- 자연어 입력을 받으면
- MCP Tool 호출 여부를 판단
- 필요 시 Tool 실행 후 응답 생성

## 5. Gradio Chat UI로 감싸기
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

이제 결과적으로 만들어진 것은:

- 프론트엔드: Gradio Chat UI
- 백엔드 추론: LLM + MCP Tool
- Tool 실행: 원격 MCP Server

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
- MCP Client는 장기 연결 객체
- 종료 시 반드시 정리 필요

## 7. Hugging Face Spaces에 배포하기
Gradio MCP Client 역시 Spaces에 배포 가능하다.

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

## 8. 핵심 정리
- Gradio는 MCP Client 역할도 수행 가능
- 기존 MCP Server를 그대로 소비 가능
- MCP Tool + Agent + UI를 빠르게 결합
- “웹 기반 Agent 데모” 제작에 최적
- Hugging Face Spaces로 손쉽게 배포 가능

이 패턴은 특히:

- MCP Tool 실험
- Agent UX 검증
- 비기술 사용자 대상 데모

에 매우 강력하다.

참고자료
Huggingface, mcp course, https://huggingface.co/learn