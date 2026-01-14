---
layout: post
title:  "허깅페이스 MCP 코스 - Local·오픈소스 모델과 MCP 사용하기 정리"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Local·오픈소스 모델과 MCP 사용하기 정리

이 글에서는 **MCP를 클라우드 API 모델이 아닌, 로컬·오픈소스 LLM과 결합하는 방법**을 다룬다.  
핵심은 **Continue**를 MCP Host로 사용해,  
로컬 모델(Ollama, llama.cpp, LM Studio 등) + MCP Server를 하나의 개발 워크플로로 묶는 것이다.

결과적으로 이 구성은 다음 목표를 달성한다.

- 코드와 데이터 **완전 로컬 실행**
- 외부 도구(MCP Server)와의 **표준화된 연동**
- IDE 안에서 동작하는 **에이전트형 코딩 어시스턴트**

## 1. Continue란 무엇인가

**Continue**는 VS Code / JetBrains에서 동작하는  
**AI 코딩 어시스턴트 프레임워크**다.

특징:
- 로컬 모델 지원
- Tool Calling 지원
- MCP Server 연동 지원
- IDE 내 Agent 모드 제공

즉, Continue는:
> *“MCP를 가장 자연스럽게 쓰는 로컬 개발용 Host”*라고 볼 수 있다.


## 2. Continue 설치

### 2.1 VS Code 확장

1. VS Code Marketplace에서 Continue 설치  
   https://marketplace.visualstudio.com/items?itemName=Continue.continue
2. 설치 후 사이드바에 Continue 아이콘 표시
3. 사용성 향상을 위해 **오른쪽 사이드바로 이동** 권장

> 참고: JetBrains용 Continue 플러그인도 제공됨

## 3. 로컬 모델 선택지

Continue는 다양한 로컬 모델 실행 환경을 지원한다.

### 3.1 주요 로컬 모델 런타임


| 런타임 | 특징 |
|------|------|
| **Ollama** | 설치·사용 간편, 모델 관리 쉬움 |
| **llama.cpp** | 고성능 C++ 기반, OpenAI 호환 서버 제공 |
| **LM Studio** | GUI 기반, 비개발자 친화적 |


Hugging Face Hub에서는 각 모델 페이지에서:
- Ollama / LM Studio / llama.cpp용 **즉시 실행 링크** 제공

### 3.2 예시: 모델 실행

#### Ollama
```bash
ollama run hf.co/unsloth/Devstral-Small-2505-GGUF:Q4_K_M
```

- 약 14GB 모델
- 시스템 RAM 여유 필요

#### llama.cpp
```bash
llama-server -hf unsloth/Devstral-Small-2505-GGUF:Q4_K_M
```

- OpenAI API 호환 서버 실행

#### LM Studio
- Hub 링크 클릭 -> 모델 다운로드
- “Local Server” 탭에서 서버 실행

## 4. Continue에 로컬 모델 등록
Continue는 .continue/models 디렉터리의 YAML 파일로 모델을 설정한다.

### 4.1 기본 구조
```
.continue/
 └─ models/
    └─ local-models.yaml
```

### 4.2 Ollama 예시 설정
```yaml
name: Ollama Devstral model
version: 0.0.1
schema: v1
models:
  - provider: ollama
    model: unsloth/devstral-small-2505-gguf:Q4_K_M
    defaultCompletionOptions:
      contextLength: 8192
    name: Ollama Devstral-Small
    roles:
      - chat
      - edit
```

### 4.3 llama.cpp 예시 설정
```yaml
name: Llama.cpp model
version: 0.0.1
schema: v1
models:
  - provider: llama.cpp
    model: unsloth/Devstral-Small-2505-GGUF
    apiBase: http://localhost:8080
    defaultCompletionOptions:
      contextLength: 8192
    name: Llama.cpp Devstral-Small
    roles:
      - chat
      - edit
```

중요
- MCP 사용 시, Tool 호출과 다중 요청을 위해 컨텍스트 길이 여유 필요
- Tool Calling을 지원하는 모델 사용 권장 (Codestral, Qwen, Llama 3.1 계열 등)

## 5. MCP Tool Handshake 개념
MCP Tool 사용은 단순 함수 호출이 아니라 명확한 핸드셰이크 흐름를 따른다.

### 5.1 Tool 전달 방식
- Tool은 이름 + arguments schema를 가진 JSON 객체로 모델에 전달됨
- 모델은 이를 보고 “이 도구를 쓸지 말지” 결정

### 5.2 전체 흐름
- 사용자 요청 + 사용 가능한 Tool 목록 -> 모델 전달
- 모델이 Tool 호출 의사 표현
- (필요 시) 사용자 승인
- Continue가 MCP Server 또는 내장 기능으로 Tool 실행
- Tool 결과를 모델에 다시 전달
- 모델이 최종 응답 또는 추가 Tool 호출

이 루프 덕분에 에이전트적 행동이 가능해진다.

## 6. Continue + MCP Server 연결
이제 로컬 모델 위에 MCP Server를 추가한다.

### 6.1 MCP Server 설정 디렉터리
```
.continue/
 └─ mcpServers/
    └─ playwright-mcp.yaml
```

### 6.2 Playwright MCP Server 예시
```yaml
name: Playwright mcpServer
version: 0.0.1
schema: v1
mcpServers:
  - name: Browser search
    command: npx
    args:
      - "@playwright/mcp@latest"
```

이 설정으로 Continue는:

- Playwright MCP Server를 실행
- 브라우저 자동화 Tool을 모델에 제공

## 7. 실제 테스트 예제
다음 프롬프트를 Continue Agent 모드에서 실행:

```
1. Using playwright, navigate to https://news.ycombinator.com.

2. Extract the titles and URLs of the top 4 posts on the homepage.

3. Create a file named hn.txt in the root directory of the project.

4. Save this list as plain text in the hn.txt file, with each line containing the title and URL separated by a hyphen.

Do not output code or instructions—just complete the task and confirm when it is done.
```

결과:

- 실제 브라우저 자동화 수행
- hn.txt 파일 생성
- IDE 내에서 작업 완료

이것이 로컬 모델 + MCP Tool + IDE 에이전트의 전형적인 활용 사례다.

## 8. 이 구성의 핵심 장점
- 완전 로컬 실행 (프라이버시 보호)
- MCP를 통한 표준화된 Tool 확장
- 로컬 LLM의 에이전트화
- 웹 자동화, 파일 조작, 시스템 작업까지 확장 가능

즉, Continue + MCP는 “로컬 AI를 단순 채팅 모델에서 실제 작업 수행 에이전트로 확장”시킨다.

## 9. 정리
- Continue는 MCP를 가장 자연스럽게 지원하는 IDE Host
- Ollama / llama.cpp / LM Studio로 로컬 모델 실행 가능
- MCP Server를 추가해 로컬 모델의 능력을 무한 확장
- 코드·데이터·도구 모두 로컬에서 제어 가능
- 진정한 local-first AI 개발 환경 완성

참고자료
Huggingface, agents course, https://huggingface.co/learn