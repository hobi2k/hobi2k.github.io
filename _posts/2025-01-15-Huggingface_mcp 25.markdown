---
layout: post
title:  "허깅페이스 MCP 코스 -  Webhook Listener"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# Webhook Listener  
Pull Request Agent의 실시간 이벤트 진입점

Webhook Listener는 Pull Request Agent의 **가장 앞단(entry point)** 이다.  
Hugging Face Hub에서 발생하는 **Discussion / Comment 이벤트**를 실시간으로 수신하고,  
이를 **MCP Client + Agent 기반 태깅 워크플로우**로 연결하는 역할을 한다.

이 섹션에서는 FastAPI를 사용해 다음을 구현한다.

- Hugging Face Hub Webhook 수신
- Webhook 보안 검증
- 이벤트 필터링
- 비동기 처리(BackgroundTasks)
- MCP Client 연동
- 모니터링 및 디버깅 엔드포인트

## 1. Webhook의 역할과 전체 흐름

Webhook은 **Hub -> 애플리케이션**으로 전달되는 *push 이벤트*다.  
폴링(polling)과 달리, 이벤트가 발생하는 즉시 호출되므로 **실시간 자동화**가 가능하다.

### 이벤트 처리 흐름

1. **User Action**  
   - 모델 리포지토리 Discussion에 댓글 작성
2. **Hub Event 생성**
3. **Webhook POST 전송**
4. **Secret 검증**
5. **이벤트 필터링**
6. **Background Task 등록**
7. **MCP Agent가 태그 처리**
8. **필요 시 PR 생성**

> Webhook Listener는 “결정”을 하지 않는다.  
> 단지 **이벤트를 정제해서 Agent에게 넘기는 관문**이다.

## 2. FastAPI 애플리케이션 기본 구성

### 2.1 Imports 및 기본 설정

```python
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
```

- FastAPI: Webhook 서버
- BackgroundTasks: 비동기 처리 핵심
- Pydantic: Webhook payload 구조 이해 및 검증

### 2.2 환경 변수 및 앱 초기화
```python
# Configuration
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
HF_TOKEN = os.getenv("HF_TOKEN")

# Simple storage for processed operations
tag_operations_store: List[Dict[str, Any]] = []

app = FastAPI(title="HF Tagging Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

**설계 포인트**
- WEBHOOK_SECRET: 보안의 핵심
- tag_operations_store:
    - 디버깅 / 관찰용 in-memory 로그
    - 실서비스에서는 DB 또는 크기 제한 필요

## 3. Webhook Payload 구조 이해
Hugging Face 공식 문서에 따른 구조를 모델로 정의한다.

```python
class WebhookEvent(BaseModel):
    event: Dict[str, str]
    comment: Dict[str, Any]
    discussion: Dict[str, Any]
    repo: Dict[str, str]
```

**사용할 핵심 필드**
- event.action -> "create"
- event.scope -> "discussion.comment"
- comment.content
- discussion.title
- repo.name

## 4. 핵심 Webhook Handler 구현
### 4.1 엔드포인트 정의
```python
@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming webhooks from Hugging Face Hub
    Following the pattern from: https://raw.githubusercontent.com/huggingface/hub-docs/refs/heads/main/docs/hub/webhooks-guide-discussion-bot.md
    """
    print("🔔 Webhook received!")
    
    # Step 1: Validate webhook secret (security)
    webhook_secret = request.headers.get("X-Webhook-Secret")
    if webhook_secret != WEBHOOK_SECRET:
        print("❌ Invalid webhook secret")
        return {"error": "incorrect secret"}, 400
```

이 엔드포인트는 Hub에서 직접 호출된다.

### 4.2 Step 1: Webhook Secret 검증
```python
    webhook_secret = request.headers.get("X-Webhook-Secret")
    if webhook_secret != WEBHOOK_SECRET:
        return {"error": "incorrect secret"}, 400
```

이 검증이 없으면 누구나 PR을 생성하게 된다.

### 4.3 Step 2: JSON 파싱 및 구조 검증
```python
    # Step 2: Parse webhook data
    try:
        webhook_data = await request.json()
        print(f"📥 Webhook data: {json.dumps(webhook_data, indent=2)}")
    except Exception as e:
        print(f"❌ Error parsing webhook data: {str(e)}")
        return {"error": "invalid JSON"}, 400
    
    # Step 3: Validate event structure
    event = webhook_data.get("event", {})
    if not event:
        print("❌ No event data in webhook")
        return {"error": "missing event data"}, 400
```

### 4.4 Step 3: 이벤트 필터링
```python
    # Step 4: Check if this is a discussion comment creation
    # Following the webhook guide pattern:
    if (
        event.get("action") == "create" and 
        event.get("scope") == "discussion.comment"
    ):
        print("✅ Valid discussion comment creation event")
        
        # Process in background to return quickly to Hub
        background_tasks.add_task(process_webhook_comment, webhook_data)
        
        return {
            "status": "accepted",
            "message": "Comment processing started",
            "timestamp": datetime.now().isoformat()
        }
    else:
        print(f"ℹ️ Ignoring event: action={event.get('action')}, scope={event.get('scope')}")
        return {
            "status": "ignored",
            "reason": "Not a discussion comment creation"
        }
```

**설계 철학**
- Webhook Listener는 선별만 한다
- 실제 처리는 Background Task로 위임
- 10초 이내 응답 보장

## 5. Background Task: Comment 처리 로직
### 5.1 기본 정보 추출
```python
async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """
    Process webhook comment to detect and add tags
    Integrates with our MCP client for Hub interactions
    """
    print("🏷️ Starting process_webhook_comment...")
    
    try:
        # Extract comment and repository information
        comment_content = webhook_data["comment"]["content"]
        discussion_title = webhook_data["discussion"]["title"]
        repo_name = webhook_data["repo"]["name"]
        discussion_num = webhook_data["discussion"]["num"]
        comment_author = webhook_data["comment"]["author"].get("id", "unknown")
        
        print(f"📝 Comment from {comment_author}: {comment_content}")
        print(f"📰 Discussion: {discussion_title}")
        print(f"📦 Repository: {repo_name}")
```

태그는 제목과 댓글 어디에든 등장할 수 있다.

### 5.2 태그 추출 및 작업 기록
```python
        # Extract potential tags from comment and title
        comment_tags = extract_tags_from_text(comment_content)
        title_tags = extract_tags_from_text(discussion_title)
        all_tags = list(set(comment_tags + title_tags))
        
        print(f"🔍 Found tags: {all_tags}")
        
        # Store operation for monitoring
        operation = {
            "timestamp": datetime.now().isoformat(),
            "repo_name": repo_name,
            "discussion_num": discussion_num,
            "comment_author": comment_author,
            "extracted_tags": all_tags,
            "comment_preview": comment_content[:100] + "..." if len(comment_content) > 100 else comment_content,
            "status": "processing"
        }
        tag_operations_store.append(operation)
```

### 5.3 MCP Agent 연동
```python
agent = await get_agent()
if not agent:
    operation["status"] = "error"
    return
```

### 5.4 Agent에게 업무 위임
```python
        if not all_tags:
            operation["status"] = "no_tags"
            operation["message"] = "No recognizable tags found"
            print("❌ No tags found to process")
            return
        
        # Get MCP agent for tag processing
        agent = await get_agent()
        if not agent:
            operation["status"] = "error"
            operation["message"] = "Agent not configured (missing HF_TOKEN)"
            print("❌ No agent available")
            return
        
        # Process each extracted tag
        operation["results"] = []
        for tag in all_tags:
            try:
                print(f"🤖 Processing tag '{tag}' for repo '{repo_name}'")
                
                # Create prompt for agent to handle tag processing
                prompt = f"""
                Analyze the repository '{repo_name}' and determine if the tag '{tag}' should be added.
                
                First, check the current tags using get_current_tags.
                If '{tag}' is not already present and it's a valid tag, add it using add_new_tag.
                
                Repository: {repo_name}
                Tag to process: {tag}
                
                Provide a clear summary of what was done.
                """
                
                response = await agent.run(prompt)
                print(f"🤖 Agent response for '{tag}': {response}")
                
                # Parse response and store result
                tag_result = {
                    "tag": tag,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                operation["results"].append(tag_result)
                
            except Exception as e:
                error_msg = f"❌ Error processing tag '{tag}': {str(e)}"
                print(error_msg)
                operation["results"].append({
                    "tag": tag,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        operation["status"] = "completed"
        print(f"✅ Completed processing {len(all_tags)} tags")
```

여기서 Tool 호출 순서 / 조건 분기 / PR 생성 여부는
전부 Agent가 판단한다.

## 6. 모니터링 및 헬스 체크 엔드포인트

Root

```python
@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "name": "HF Tagging Bot",
        "status": "running",
        "description": "Webhook listener for automatic model tagging",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "operations": "/operations"
        }
    }
```

Health Check

```python
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    agent = await get_agent()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "webhook_secret": "configured" if WEBHOOK_SECRET else "missing",
            "hf_token": "configured" if HF_TOKEN else "missing",
            "mcp_agent": "ready" if agent else "not_ready"
        }
    }
```

- Secret 설정 여부
- HF_TOKEN 여부
- MCP Agent 준비 상태

Operation 로그 조회
```python
@app.get("/operations")
async def get_operations():
    """Get recent tag operations for monitoring"""
    # Return last 50 operations
    recent_ops = tag_operations_store[-50:] if tag_operations_store else []
    return {
        "total_operations": len(tag_operations_store),
        "recent_operations": recent_ops
    }
```

- 최근 Webhook 처리 내역 확인
- 디버깅 필수

## 7. Hugging Face Hub Webhook 설정

### 설정 항목
- Repository 선택
- Webhook URL
- https://<space>.hf.space/webhook
- Secret: WEBHOOK_SECRET
- Event: Community (PR & discussions)

## 8. 테스트 전략
### 8.1 로컬 테스트
```python
# test_webhook_local.py
import requests
import json

# Test data matching webhook format
test_webhook_data = {
    "event": {
        "action": "create",
        "scope": "discussion.comment"
    },
    "comment": {
        "content": "This model needs tags: pytorch, transformers",
        "author": {"id": "test-user"}
    },
    "discussion": {
        "title": "Missing tags",
        "num": 1
    },
    "repo": {
        "name": "test-user/test-model"
    }
}

# Send test webhook
response = requests.post(
    "http://localhost:8000/webhook",
    json=test_webhook_data,
    headers={"X-Webhook-Secret": "your-test-secret"}
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

### 8.2 Simulation Endpoint (개발용)
```python
@app.post("/simulate_webhook")
async def simulate_webhook(
    repo_name: str, 
    discussion_title: str, 
    comment_content: str
) -> str:
    """Simulate webhook for testing purposes"""
    
    # Create mock webhook data
    mock_webhook_data = {
        "event": {
            "action": "create",
            "scope": "discussion.comment"
        },
        "comment": {
            "content": comment_content,
            "author": {"id": "test-user"}
        },
        "discussion": {
            "title": discussion_title,
            "num": 999
        },
        "repo": {
            "name": repo_name
        }
    }
    
    # Process the simulated webhook
    await process_webhook_comment(mock_webhook_data)
    
    return f"Simulated webhook processed for {repo_name}"
```

- 실제 Discussion 없이 시뮬레이션 가능
- 태그 추출 / Agent 처리 검증에 매우 유용

## 9. 설계 핵심 요약
이 Webhook Listener의 특징은 다음과 같다.

- 보안 우선: Secret 검증
- 빠른 응답: BackgroundTasks
- 결정 위임: Agent에게 로직 위탁
- 관찰 가능성: Operation 로그 제공
- 확장성: 다른 이벤트 타입 추가 가능

참고자료
Huggingface, agents course, https://huggingface.co/learn