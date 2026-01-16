---
layout: post
title:  "허깅페이스 MCP 코스 -  Creating the MCP Server"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# Creating the MCP Server  

*Hugging Face Hub Pull Request Agent – Core Implementation*

이 글에서는 Pull Request Agent의 핵심인 **MCP Server**를 구현한다.  
이 서버는 Hugging Face Hub와 직접 통신하며,  
에이전트가 사용할 **행동 가능한 도구(Tools)**를 제공한다.

본 MCP Server의 책임은 명확하다.

- Hugging Face 모델 리포지토리의 **현재 태그 조회**
- 새로운 태그를 **Pull Request 형태로 안전하게 추가**

Agent는 “무엇을 할지” 판단하고,  
MCP Server는 “어떻게 할지” 실행한다.

## 1. MCP Server 아키텍처 개요

본 서버는 **두 개의 핵심 MCP Tool**만 제공한다.


| Tool | 역할 |
|---|---|
| `get_current_tags` | 모델 리포지토리의 기존 태그 조회 |
| `add_new_tag` | 새 태그를 추가하는 Pull Request 생성 |


이 설계의 핵심은 다음 원칙이다.

- Hub API의 복잡성은 **서버 내부에 은닉**
- Agent는 **단순한 JSON 인터페이스**만 사용
- 모든 변경은 **PR 기반(비파괴적)** 으로 수행

## 2. MCP Server 전체 구현 구조

서버 파일은 `mcp_server.py` 하나로 구성된다.  
FastMCP + Hugging Face Hub SDK 조합을 사용한다.

## 3. Imports 및 기본 설정

### 3.1 필수 모듈 로드

```python
#!/usr/bin/env python3
"""
Simplified MCP Server for HuggingFace Hub Tagging Operations using FastMCP
"""

import os
import json
from fastmcp import FastMCP
from huggingface_hub import HfApi, model_info, ModelCard, ModelCardData
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv

load_dotenv()
```

**설계 포인트**
- FastMCP
  - MCP Server 런타임
- huggingface_hub
  - 모델 정보 조회, 커밋, PR 생성
- load_dotenv()
  - 로컬 개발 시 .env 자동 로딩

uv run 환경이라면 load_dotenv() 없이도 동작 가능하지만,
명시적으로 두는 편이 문서 가독성과 이식성에 유리하다.

### 3.2 서버 초기화 및 인증 설정
```python
# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize HF API client
hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

# Create the FastMCP server
mcp = FastMCP("hf-tagging-bot")
```

이 블록은 세 가지를 보장한다.

1. 환경 변수 기반 인증
2. 토큰이 없어도 서버 구조는 기동 가능
3. MCP 서버를 명확한 이름으로 식별 가능

토큰이 없을 경우 Tool은 정상 등록되지만
실행 시 명확한 에러 JSON을 반환하도록 설계한다.

## 4. Tool 1. get_current_tags
### 4.1 Tool 목적
- 특정 모델 리포지토리의 현재 태그 상태 조회
- Agent의 “태그 추가 필요 여부 판단”을 위한 기초 데이터 제공

### 4.2 Tool 시그니처 및 초기 검증
```python
@mcp.tool()
def get_current_tags(repo_id: str) -> str:
    """Get current tags from a HuggingFace model repository"""
    print(f"🔧 get_current_tags called with repo_id: {repo_id}")

    if not hf_api:
        error_result = {"error": "HF token not configured"}
        json_str = json.dumps(error_result)
        print(f"❌ No HF API token - returning: {json_str}")
        return json_str
```

**MCP Tool 설계 원칙**
- 반환값은 반드시 문자열
- 구조화 데이터는 json.dumps()로 직렬화
- 인증 실패는 즉시 명확히 반환

### 4.3 Hub API 호출 및 결과 구성
```python
    try:
        print(f"📡 Fetching model info for: {repo_id}")
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []
        print(f"🏷️ Found {len(current_tags)} tags: {current_tags}")

        result = {
            "status": "success",
            "repo_id": repo_id,
            "current_tags": current_tags,
            "count": len(current_tags),
        }
        json_str = json.dumps(result)
        print(f"✅ get_current_tags returning: {json_str}")
        return json_str
```

**특징**
- Hub 응답을 Agent 친화적 JSON으로 변환
- 태그 개수까지 포함해 후속 판단을 쉽게 만듦

### 4.4 예외 처리
```python
    except Exception as e:
        print(f"❌ Error in get_current_tags: {str(e)}")
        error_result = {"status": "error", "repo_id": repo_id, "error": str(e)}
        json_str = json.dumps(error_result)
        print(f"❌ get_current_tags error returning: {json_str}")
        return json_str
```

MCP Tool은 “조용히 실패”하면 안 된다.
항상 상태가 명시된 JSON을 반환해야 한다.

## 5. Tool 2. add_new_tag
### 5.1 Tool 목적
- 새로운 태그를 직접 수정하지 않고
- README.md 갱신 -> Pull Request 생성

이는 Hugging Face Hub의 표준 워크플로우를 따른다.

### 5.2 초기 검증 및 현재 상태 확인
```python
@mcp.tool()
def add_new_tag(repo_id: str, new_tag: str) -> str:
    """Add a new tag to a HuggingFace model repository via PR"""
    print(f"🔧 add_new_tag called with repo_id: {repo_id}, new_tag: {new_tag}")

    if not hf_api:
        error_result = {"error": "HF token not configured"}
        json_str = json.dumps(error_result)
        print(f"❌ No HF API token - returning: {json_str}")
        return json_str
```

### 5.3 중복 태그 방지 로직
```python
    try:
        # Get current model info and tags
        print(f"📡 Fetching current model info for: {repo_id}")
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []
        print(f"🏷️ Current tags: {current_tags}")

        # Check if tag already exists
        if new_tag in current_tags:
            print(f"⚠️ Tag '{new_tag}' already exists in {current_tags}")
            result = {
                "status": "already_exists",
                "repo_id": repo_id,
                "tag": new_tag,
                "message": f"Tag '{new_tag}' already exists",
            }
            json_str = json.dumps(result)
            print(f"🏷️ add_new_tag (already exists) returning: {json_str}")
            return json_str
```

**중요한 설계 포인트**
- 상태 확인 -> 변경
- 중복 PR 생성 방지
- Agent가 “이미 처리됨”을 인식 가능

### 5.4 Model Card 처리 (README.md)
```python
        # Add the new tag to existing tags
        updated_tags = current_tags + [new_tag]
        print(f"🆕 Will update tags from {current_tags} to {updated_tags}")

        # Create model card content with updated tags
        try:
            # Load existing model card
            print(f"📄 Loading existing model card...")
            card = ModelCard.load(repo_id, token=HF_TOKEN)
            if not hasattr(card, "data") or card.data is None:
                card.data = ModelCardData()
        except HfHubHTTPError:
            # Create new model card if none exists
            print(f"📄 Creating new model card (none exists)")
            card = ModelCard("")
            card.data = ModelCardData()

        # Update tags - create new ModelCardData with updated tags
        card_dict = card.data.to_dict()
        card_dict["tags"] = updated_tags
        card.data = ModelCardData(**card_dict)
```

**의미**
- 기존 카드가 없어도 안전하게 동작
- 메타데이터 조작은 ModelCard API를 통해 수행

### 5.5 Pull Request 생성
```python
        # Create commit with updated model card using CommitOperationAdd
        from huggingface_hub import CommitOperationAdd

        commit_info = hf_api.create_commit(
            repo_id=repo_id,
            operations=[
                CommitOperationAdd(
                    path_in_repo="README.md", path_or_fileobj=str(card).encode("utf-8")
                )
            ],
            commit_message=pr_title,
            commit_description=pr_description,
            token=HF_TOKEN,
            create_pr=True,
        )

        # Extract PR URL from commit info
        pr_url_attr = commit_info.pr_url
        pr_url = pr_url_attr if hasattr(commit_info, "pr_url") else str(commit_info)

        print(f"✅ PR created successfully! URL: {pr_url}")

        result = {
            "status": "success",
            "repo_id": repo_id,
            "tag": new_tag,
            "pr_url": pr_url,
            "previous_tags": current_tags,
            "new_tags": updated_tags,
            "message": f"Created PR to add tag '{new_tag}'",
        }
        json_str = json.dumps(result)
        print(f"✅ add_new_tag success returning: {json_str}")
        return json_str
```

**핵심 포인트**
- create_pr=True
- PR 기반 변경
- Hub 운영 정책에 완벽히 부합

### 5.6 성공 응답
```python
        result = {
            "status": "success",
            "repo_id": repo_id,
            "tag": new_tag,
            "pr_url": commit_info.pr_url,
            "previous_tags": current_tags,
            "new_tags": current_tags + [new_tag],
        }
        return json.dumps(result)
```

### 5.7 예외 및 트레이스 처리
```python
    except Exception as e:
        print(f"❌ Error in add_new_tag: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")

        error_result = {
            "status": "error",
            "repo_id": repo_id,
            "tag": new_tag,
            "error": str(e),
        }
        json_str = json.dumps(error_result)
        print(f"❌ add_new_tag error returning: {json_str}")
        return json_str
```
자동화 시스템에서 traceback 로그는 생명선이다.

## 6. 설계 상 주의점

**무한 PR 루프 위험**
- PR 생성 -> Webhook -> Agent 재호출 -> 또 PR
- 반드시:
  - 기존 PR 존재 여부 확인
  - tag 중복 여부 체크
  - 이벤트 타입 필터링

## 7. 정리
이 MCP Server는 다음 특성을 가진다.

- Hub API를 안전한 도구로 캡슐화
- Agent는 판단만, Server는 행동만
- 모든 변경은 투명한 PR 기반
- 실패도 항상 구조화된 JSON

참고자료
Huggingface, agents course, https://huggingface.co/learn