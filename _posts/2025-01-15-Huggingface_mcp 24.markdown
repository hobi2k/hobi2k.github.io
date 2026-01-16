---
layout: post
title:  "허깅페이스 MCP 코스 -  MCP Client"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# MCP Client  
**Pull Request Agent – MCP Server와 Webhook을 연결하는 실행 계층**

이 글에서는 **MCP Client**를 구현한다.  
MCP Client는 단순한 보조 컴포넌트가 아니라, 다음을 책임지는 핵심 계층이다.

- Webhook으로 들어온 이벤트를 **Agent가 이해할 수 있는 문제**로 변환
- Agent가 **MCP Server의 Tool을 올바른 순서로 사용**하도록 중개
- 자연어 지시 -> Tool 호출 -> 결과 요약의 전체 흐름을 담당

즉, **Webhook Handler - MCP Client - MCP Server**를 연결하는 **지능형 브리지**다.

## 1. MCP Client의 위치와 역할

본 실습에서는 MCP Client를 **별도 프로세스**로 두지 않고,  
메인 FastAPI 애플리케이션(`app.py`) 내부에 통합한다.

### 이유

- Webhook 처리와 Agent 판단은 **강하게 결합된 로직**
- 단일 프로세스 내에서 상태 공유(Agent singleton)가 유리

> 실제 서비스에서는 MCP Server / Client를 분리한 레포 구조도 충분히 가능하다.

## 2. MCP Client 아키텍처 개요

MCP Client는 다음 구성으로 동작한다.

1. **Agent 생성 및 관리**
2. MCP Server(`mcp_server.py`)와 stdio 연결
3. Tool 자동 탐색 및 로딩
4. 자연어 지시 기반 Tool 오케스트레이션

![MCP Client Integration](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit3/app.png)

## 3. Agent 기반 MCP Client

우리는 `huggingface_hub`에서 제공하는 **Agent 클래스**를 사용한다.  
이 클래스는 다음을 동시에 제공한다.

- LLM 추론 능력
- MCP Tool 호출 능력
- Tool 선택 및 순서 결정 로직

즉, **“판단 + 실행”이 결합된 고수준 Client**다.

## 4. Agent 설정 및 Singleton 관리

### 4.1 기본 설정

```python
from huggingface_hub.inference._mcp.agent import Agent
from typing import Optional, Literal

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
DEFAULT_PROVIDER: Literal["hf-inference"] = "hf-inference"

# Global agent instance
agent_instance: Optional[Agent] = None
```

**설계 포인트**
- Agent 생성 비용은 비싸므로 singleton 패턴 사용
- Webhook 요청마다 재생성 안 함
- 한 번 생성 후 재사용

### 4.2 Agent 생성 함수
```python
async def get_agent():
    """Get or create Agent instance"""
    print("🤖 get_agent() called...")
    global agent_instance
    if agent_instance is None and HF_TOKEN:
        print("🔧 Creating new Agent instance...")
        print(f"🔑 HF_TOKEN present: {bool(HF_TOKEN)}")
        print(f"🤖 Model: {HF_MODEL}")
        print(f"🔗 Provider: {DEFAULT_PROVIDER}")
```

이 함수는:

- Agent가 이미 존재하면 그대로 반환
- 없을 경우에만 새로 생성
- 토큰이 없으면 생성 자체를 차단

### 4.3 MCP Server 연결 포함 Agent 생성
```python
        try:
            agent_instance = Agent(
                model=HF_MODEL,
                provider=DEFAULT_PROVIDER,
                api_key=HF_TOKEN,
                servers=[
                    {
                        "type": "stdio",
                        "command": "python",
                        "args": ["mcp_server.py"],
                        "cwd": ".",
                        "env": {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},
                    }
                ],
            )
            print("✅ Agent instance created successfully")
            print("🔧 Loading tools...")
            await agent_instance.load_tools()
            print("✅ Tools loaded successfully")
        except Exception as e:
            print(f"❌ Error creating/loading agent: {str(e)}")
            agent_instance = None
```

**핵심 포인트**
- type: "stdio"
    - MCP Server를 서브프로세스로 실행
- args: ["mcp_server.py"]
    - 만든 MCP Server 직접 실행
- load_tools()
    - MCP Server에 등록된 Tool 자동 탐색

이 한 줄로 Agent는
get_current_tags, add_new_tag의 존재를 스스로 인지한다.

### 4.4 실패 처리
```python
        try:
            agent_instance = Agent(
                model=HF_MODEL,
                provider=DEFAULT_PROVIDER,
                api_key=HF_TOKEN,
                servers=[
                    {
                        "type": "stdio",
                        "command": "python",
                        "args": ["mcp_server.py"],
                        "cwd": ".",
                        "env": {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},
                    }
                ],
            )
            print("✅ Agent instance created successfully")
            print("🔧 Loading tools...")
            await agent_instance.load_tools()
            print("✅ Tools loaded successfully")
        except Exception as e:
            print(f"❌ Error creating/loading agent: {str(e)}")
            agent_instance = None
```

Agent 생성 실패 시:

- 상태를 명확히 로그
- 이후 호출에서 재시도 가능

## 5. MCP Tool 자동 사용 방식
Agent는 Tool을 직접 호출하지 않는다.
자연어 지시를 기반으로 스스로 판단한다.

**사용 가능한 Tool**
- get_current_tags(repo_id)
- add_new_tag(repo_id, new_tag)

### Tool 사용 예시
```python
async def example_tool_usage():
    agent = await get_agent()
    
    if agent:
        response = await agent.run(
            "Check the current tags for microsoft/DialoGPT-medium and add the tag 'conversational-ai' if it's not already present"
        )
        print(response)
```

Agent 내부에서 일어나는 일:

- 현재 태그 확인 필요 -> get_current_tags
- 태그 존재 여부 판단
- 없을 경우 -> add_new_tag
- 수행 결과 요약

Tool orchestration 로직을 직접 짜지 않는다
-> 이것이 MCP + Agent의 가장 큰 장점

## 6. Webhook 처리와 MCP Client 연결
이제 MCP Client는 Webhook 이벤트 처리 파이프라인에 들어간다.

### 6.1 Webhook 데이터 처리 흐름
```python
async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """Process webhook to detect and add tags"""
    print("🏷️ Starting process_webhook_comment...")

    try:
        comment_content = webhook_data["comment"]["content"]
        discussion_title = webhook_data["discussion"]["title"]
        repo_name = webhook_data["repo"]["name"]
        
        # Extract potential tags from the comment and discussion title
        comment_tags = extract_tags_from_text(comment_content)
        title_tags = extract_tags_from_text(discussion_title)
        all_tags = list(set(comment_tags + title_tags))

        print(f"🔍 All unique tags: {all_tags}")

        if not all_tags:
            return ["No recognizable tags found in the discussion."]
```

### 6.2 태그 후보 추출
```python
        # Get agent instance
        agent = await get_agent()
        if not agent:
            return ["Error: Agent not configured (missing HF_TOKEN)"]

        # Process each tag
        result_messages = []
        for tag in all_tags:
            try:
                # Use agent to process the tag
                prompt = f"""
                For the repository '{repo_name}', check if the tag '{tag}' already exists.
                If it doesn't exist, add it via a pull request.
                
                Repository: {repo_name}
                Tag to check/add: {tag}
                """
                
                print(f"🤖 Processing tag '{tag}' for repo '{repo_name}'")
                response = await agent.run(prompt)
                
                # Parse agent response for success/failure
                if "success" in response.lower():
                    result_messages.append(f"✅ Tag '{tag}' processed successfully")
                else:
                    result_messages.append(f"⚠️ Issue with tag '{tag}': {response}")
                    
            except Exception as e:
                error_msg = f"❌ Error processing tag '{tag}': {str(e)}"
                print(error_msg)
                result_messages.append(error_msg)

        return result_messages
```

**설계 의도**
- 제목 + 댓글 양쪽 모두 분석
- 중복 제거
- 태그 누락 가능성 최소화

### 6.3 Agent를 통한 태그 처리
```python
        agent = await get_agent()
        if not agent:
            return ["Error: Agent not configured (missing HF_TOKEN)"]

        result_messages = []
        for tag in all_tags:
            prompt = f"""
            For the repository '{repo_name}', check if the tag '{tag}' already exists.
            If it doesn't exist, add it via a pull request.
            """
            response = await agent.run(prompt)
```

여기서 중요한 점:

- MCP Client는 Tool을 직접 호출하지 않는다
- Agent에게 “업무 지시”만 전달
- Tool 선택·순서·결과 해석은 Agent 책임

## 7. 태그 추출 로직
### 7.1 허용 태그 목록
```python
import re
from typing import List

# Recognized ML/AI tags for validation
RECOGNIZED_TAGS = {
    "pytorch", "tensorflow", "jax", "transformers", "diffusers",
    "text-generation", "text-classification", "question-answering",
    "text-to-image", "image-classification", "object-detection",
    "fill-mask", "token-classification", "translation", "summarization",
    "feature-extraction", "sentence-similarity", "zero-shot-classification",
    "image-to-text", "automatic-speech-recognition", "audio-classification",
    "voice-activity-detection", "depth-estimation", "image-segmentation",
    "video-classification", "reinforcement-learning", "tabular-classification",
    "tabular-regression", "time-series-forecasting", "graph-ml", "robotics",
    "computer-vision", "nlp", "cv", "multimodal",
}
```

**목적**
- 스팸 태그 방지
- ML/AI 도메인 태그만 허용

### 7.2 텍스트 기반 태그 추출
```python
def extract_tags_from_text(text: str) -> List[str]:
    """Extract potential tags from discussion text"""
    text_lower = text.lower()
    explicit_tags = []

    # Pattern 1: "tag: something" or "tags: something"
    tag_pattern = r"tags?:\s*([a-zA-Z0-9-_,\s]+)"
    matches = re.findall(tag_pattern, text_lower)
    for match in matches:
        tags = [tag.strip() for tag in match.split(",")]
        explicit_tags.extend(tags)

    # Pattern 2: "#hashtag" style
    hashtag_pattern = r"#([a-zA-Z0-9-_]+)"
    hashtag_matches = re.findall(hashtag_pattern, text_lower)
    explicit_tags.extend(hashtag_matches)

    # Pattern 3: Look for recognized tags mentioned in natural text
    mentioned_tags = []
    for tag in RECOGNIZED_TAGS:
        if tag in text_lower:
            mentioned_tags.append(tag)

    # Combine and deduplicate
    all_tags = list(set(explicit_tags + mentioned_tags))

    # Filter to only include recognized tags or explicitly mentioned ones
    valid_tags = []
    for tag in all_tags:
        if tag in RECOGNIZED_TAGS or tag in explicit_tags:
            valid_tags.append(tag)

    return valid_tags
```

사용 전략:
1. tags: pytorch, transformers
2. #hashtag
3. 자연어 언급 (“this transformers model…”)
-> 이후 화이트리스트 필터링

## 8. 성능 및 운영 고려사항
### 8.1 Agent Singleton
- MCP Server 재기동 비용 절감
- Tool 로딩 1회로 제한

### 8.2 Async 처리
- Webhook 다중 요청 대응
- FastAPI 이벤트 루프 블로킹 방지

### 8.3 Background Task 패턴
```python
from fastapi import BackgroundTasks

@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle webhook and process in background"""
    
    # Validate webhook quickly
    if request.headers.get("X-Webhook-Secret") != WEBHOOK_SECRET:
        return {"error": "Invalid secret"}
    
    webhook_data = await request.json()
    
    # Process in background to return quickly
    background_tasks.add_task(process_webhook_comment, webhook_data)
    
    return {"status": "accepted"}
```

**이유**
- Webhook 응답은 빠르게
- 실제 작업은 백그라운드

대부분의 플랫폼은
Webhook 응답이 10초 이상 걸리면 실패로 간주한다.

## 9. 정리
이 MCP Client는 다음을 수행한다.

- Webhook 이벤트를 Agent 친화적 문제로 변환
- Agent가 MCP Tool을 자율적으로 조합하도록 지원
- 복잡한 조건 분기를 코드가 아닌 LLM 추론에 위임

참고자료
Huggingface, agents course, https://huggingface.co/learn