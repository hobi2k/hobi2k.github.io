---
layout: post
title:  "허깅페이스 MCP 코스 - Gradio 기반 MCP Server 구축 정리"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Gradio 기반 MCP Server 구축 정리

이 글에서는 **Gradio를 활용해 실제로 동작하는 MCP Server를 구축하는 전체 과정을 예제로 정리**한다.  
목표는 하나의 서버가 다음 두 역할을 동시에 수행하도록 만드는 것이다.

1. **사람을 위한 웹 인터페이스**
2. **AI(Client)를 위한 MCP Tool 서버**

예제로는 감정 분석(Sentiment Analysis) 서버를 구현한다.

## 1. Gradio MCP Integration 개요

Gradio는 `launch(mcp_server=True)` 옵션을 통해 MCP Server를 자동으로 생성한다.  
이 한 줄의 옵션으로 Gradio는 다음 작업을 내부적으로 수행한다.

1. Python 함수를 **MCP Tool**로 자동 변환
2. 입력 컴포넌트를 Tool의 **argument schema**로 매핑
3. 출력 컴포넌트를 MCP **response format**으로 매핑
4. **JSON-RPC over HTTP + SSE** 통신 설정
5. 웹 UI와 MCP Server 엔드포인트를 동시에 생성

즉, 개발자는 MCP의 복잡한 프로토콜을 직접 구현하지 않고도  
“함수 중심으로 MCP Server를 설계”할 수 있다.

## 2. 프로젝트 초기 설정

### 2.1 프로젝트 디렉터리 및 가상환경

```bash
mkdir mcp-sentiment
cd mcp-sentiment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2.2 의존성 설치
```bash
pip install "gradio[mcp]" textblob
```

- gradio[mcp]: MCP Server 기능 포함
- textblob: 감정 분석 라이브러리

## 3. 서버 파일 구성 (app.py)
Hugging Face Spaces는 app.py 파일을 기준으로 앱을 빌드하므로,
파일 이름은 반드시 app.py여야 한다.

### 3.1 전체 코드
```python
import json
import gradio as gr
from textblob import TextBlob

def sentiment_analysis(text: str) -> str:
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A JSON string containing polarity, subjectivity, and assessment
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment

    result = {
        "polarity": round(sentiment.polarity, 2),
        "subjectivity": round(sentiment.subjectivity, 2),
        "assessment": (
            "positive" if sentiment.polarity > 0
            else "negative" if sentiment.polarity < 0
            else "neutral"
        )
    }

    return json.dumps(result)

demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Enter text to analyze..."),
    outputs=gr.Textbox(),
    title="Text Sentiment Analysis",
    description="Analyze the sentiment of text using TextBlob"
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
```

## 4. 코드 핵심 해설
### 4.1 감정 분석 함수
- 입력: text: str
- 출력: JSON 문자열
- TextBlob을 이용해:
  - polarity (−1 ~ 1)
  - subjectivity (0 ~ 1)
  - 종합 평가(positive / neutral / negative) 계산

출력을 JSON 문자열(str)로 반환한 이유

- MCP Client 호환성을 높이기 위함
- 복잡한 구조는 문자열로 반환 후 Client에서 파싱하는 방식이 가장 안정적

### 4.2 Docstring과 타입 힌트의 중요성
- Gradio는 Docstring + Type Hint를 기반으로 MCP Tool 스키마를 생성
- 반드시 다음을 포함해야 한다.
  - 함수 설명
  - Args: 블록
  - 파라미터 타입

이는 MCP Client가 Tool을 올바르게 호출하기 위한 계약(Contract) 역할을 한다.

### 4.3 Gradio Interface 정의
```python
gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(...),
    outputs=gr.Textbox()
)
```


이 한 줄로:

- 웹 UI 생성
- MCP Tool 정의

가 동시에 이루어진다.

### 4.4 MCP Server 활성화
```python
demo.launch(mcp_server=True)
```

또는 환경 변수로도 가능:

```bash
export GRADIO_MCP_SERVER=True
```

활성화되면 MCP Server 엔드포인트가 자동으로 열린다.

## 5. 서버 실행 및 접근
### 5.1 실행
```bash
python app.py
```

### 5.2 접근 URL
- 웹 UI

```
http://localhost:7860
```

- MCP Server (SSE)

```bash
http://localhost:7860/gradio_api/mcp/sse
```

- MCP 스키마

```bash
http://localhost:7860/gradio_api/mcp/schema
```

스키마 URL은 MCP Client가 Tool 구조를 이해하는 핵심 엔드포인트다.

## 6. 테스트 방법
### 6.1 웹 UI 테스트

- 브라우저에서 텍스트 입력
- 감정 분석 결과 확인

### 6.2 MCP 관점 테스트

- /gradio_api/mcp/schema 접속
- Tool 이름, 입력, 출력 구조 확인
- MCP Client가 사용할 정확한 인터페이스 검증

## 7. 문제 해결 가이드
### 7.1 타입 힌트 & Docstring 누락
- MCP Tool 스키마 생성 실패의 가장 흔한 원인
- 항상 타입과 Args: 명시

### 7.2 입력 타입 호환성
- MCP Client마다 타입 처리 차이가 있음
- 입력은 가급적 str로 받고 내부에서 변환 권장

### 7.3 SSE 미지원 Client
일부 MCP Client는 SSE 서버를 직접 지원하지 않는다.
이 경우 mcp-remote를 사용해 중계한다.

```json
{
  "mcpServers": {
    "gradio": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:7860/gradio_api/mcp/sse"
      ]
    }
  }
}
```

### 7.4 연결 문제
- Server / Client 재시작
- 포트 충돌 여부 확인
- /mcp/schema 접근 가능 여부 확인

## 8. Hugging Face Spaces 배포
### 8.1 Space 생성
- https://huggingface.co/spaces
- SDK: Gradio
- 이름 예: mcp-sentiment

### 8.2 requirements.txt
```
gradio[mcp]
textblob
```

### 8.3 배포
```bash
git init
git add app.py requirements.txt
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/mcp-sentiment
git push -u origin main
```

### 8.4 배포 후 MCP Server URL
```
https://YOUR_USERNAME-mcp-sentiment.hf.space/gradio_api/mcp/sse
```

이제 누구나 MCP Client에서 이 서버를 Tool로 사용할 수 있다.

## 9. 핵심 정리

- Gradio는 MCP Server를 만드는 가장 빠른 방법
- Python 함수 -> MCP Tool 자동 변환
- 웹 UI + MCP API 동시 제공
- 로컬 실행과 Hugging Face Spaces 배포 모두 지원
- 실전 MCP Server 구현의 표준적인 출발점

참고자료
Huggingface, mcp course, https://huggingface.co/learn