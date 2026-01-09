---
layout: post
title:  "허깅페이스 에이전트 코스 - Building Your First LangGraph"
date:   2025-01-9 00:10:22 +0900
categories: Huggingface_agent
---

# Building Your First LangGraph

이번 글에서는 지금까지 살펴본 LangGraph의 구성 요소(State, Node, Edge)를 실제로 사용하여 **동작하는 첫 번째 그래프**를 만들어본다.  
예제 시나리오는 집사 **Alfred**의 이메일 처리 시스템이며, 다음과 같은 흐름을 가진다.

## 목표 워크플로우

Alfred는 다음 작업을 순차적으로 수행해야 한다.

1. 수신된 이메일을 읽는다
2. 이메일을 스팸(spam) 또는 정상(legitimate)으로 분류한다
3. 정상 이메일인 경우, 임시 답변 초안을 작성한다
4. 정상 이메일인 경우, Mr. Hugg에게 내용을 보고한다 (출력만 수행)

이 예제는 **LLM 기반 판단이 포함된 워크플로우를 LangGraph로 어떻게 구조화하는지**에 초점을 둔다.  
툴 호출은 포함하지 않으므로 “에이전트”라기보다는 **LangGraph 흐름 설계 예제**에 가깝다.

## 환경 설정

### 필수 패키지 설치

```python
# LangGraph와 OpenAI 연동을 위한 패키지 설치
%pip install langgraph langchain_openai
```

### 모듈 임포트

```python
import os
from typing import TypedDict, List, Dict, Any, Optional

# LangGraph 핵심 구성 요소
from langgraph.graph import StateGraph, START, END

# OpenAI 기반 LLM 래퍼
from langchain_openai import ChatOpenAI

# LangChain 메시지 타입 (LLM 입력용)
from langchain_core.messages import HumanMessage
```

## Step 1: State 정의

LangGraph에서 **State는 워크플로우 전반을 관통하는 핵심 데이터 구조**다.  
이메일 처리 과정에서 Alfred가 기억해야 할 모든 정보를 State에 명시적으로 정의한다.

```python
class EmailState(TypedDict):
    # 현재 처리 중인 이메일 전체 정보
    # sender, subject, body 등을 포함
    email: Dict[str, Any]

    # 이메일 분류 결과 (예: inquiry, complaint 등)
    email_category: Optional[str]

    # 스팸으로 판단된 경우 그 이유
    spam_reason: Optional[str]

    # 스팸 여부 판단 결과
    is_spam: Optional[bool]
    
    # 정상 이메일에 대해 생성된 답변 초안
    email_draft: Optional[str]
    
    # LLM과의 상호작용 기록 (디버깅/분석용)
    messages: List[Dict[str, Any]]
```

> Tip  
> State는 **의사결정에 필요한 정보만 포함**하도록 설계하는 것이 중요하다.  
> 너무 빈약하면 분기가 불가능하고, 너무 크면 유지보수가 어려워진다.

## Step 2: Node 정의

각 Node는 **하나의 Python 함수**이며,  
State를 입력으로 받아 작업을 수행하고 **State의 일부를 갱신**하여 반환한다.

### LLM 초기화

```python
# 온도를 0으로 설정하여 응답의 결정성을 높임
model = ChatOpenAI(temperature=0)
```

---

### 이메일 읽기 노드

```python
def read_email(state: EmailState):
    """
    Alfred가 이메일을 읽고 기본 정보를 로그로 출력하는 단계
    실제로는 전처리나 메타데이터 분석이 들어갈 수 있음
    """
    email = state["email"]
    
    print(
        f"Alfred is processing an email from {email['sender']} "
        f"with subject: {email['subject']}"
    )
    
    # State 변경 없음
    return {}
```

---

### 이메일 분류 노드 (LLM 사용)

```python
def classify_email(state: EmailState):
    """
    LLM을 사용해 이메일이 스팸인지 정상인지 판단하고,
    정상일 경우 카테고리를 분류한다.
    """
    email = state["email"]
    
    # LLM에게 전달할 프롬프트 구성
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """
    
    # LLM 호출
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    # 단순 문자열 기반 파싱 (실서비스에서는 구조화 출력 권장)
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text
    
    # 스팸 사유 추출
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()
    
    # 정상 이메일일 경우 카테고리 추출
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break
    
    # LLM 대화 로그 누적
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }
```

### 스팸 처리 노드

```python
def handle_spam(state: EmailState):
    """
    스팸 이메일을 처리하는 단계
    실제 서비스에서는 삭제, 보관, 신고 등의 로직이 들어갈 수 있음
    """
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    
    return {}
```

### 답변 초안 작성 노드

```python
def draft_response(state: EmailState):
    """
    정상 이메일에 대해 LLM을 사용해 답변 초안을 생성
    """
    email = state["email"]
    category = state["email_category"] or "general"
    
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review.
    """
    
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    return {
        "email_draft": response.content,
        "messages": new_messages
    }
```

### Mr. Hugg에게 보고하는 노드

```python
def notify_mr_hugg(state: EmailState):
    """
    최종적으로 Mr. Hugg에게 이메일 내용과 초안을 전달
    (여기서는 콘솔 출력만 수행)
    """
    email = state["email"]
    
    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nDraft response:")
    print("-" * 50)
    print(state["email_draft"])
    print("=" * 50 + "\n")
    
    return {}
```

## Step 3: 분기 로직 정의

```python
def route_email(state: EmailState) -> str:
    """
    classify_email 이후 어떤 경로로 갈지 결정
    반환값은 conditional edge의 key와 반드시 일치해야 한다
    """
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"
```

## Step 4: StateGraph 구성 및 연결

```python
# StateGraph 생성
email_graph = StateGraph(EmailState)

# 노드 등록
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# 시작 지점
email_graph.add_edge(START, "read_email")

# 기본 흐름
email_graph.add_edge("read_email", "classify_email")

# 조건 분기
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

# 종료 지점 연결
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# 그래프 컴파일
compiled_graph = email_graph.compile()
```

## Step 5: 실행 테스트

```python
# 정상 이메일 예시
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": (
        "Dear Mr. Hugg, I was referred to you by a colleague and "
        "I'm interested in learning more about your consulting services."
    )
}

# 스팸 이메일 예시
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": (
        "CONGRATULATIONS! You have been selected as the winner of our lottery!"
    )
}

compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})
```


## 그래프 시각화

```python
# Mermaid 다이어그램으로 그래프 구조 시각화
compiled_graph.get_graph().draw_mermaid_png()
```

## 핵심 정리

- State를 중심으로 모든 흐름이 구성된다
- Node는 “작업”, Edge는 “의사결정”을 담당한다
- 조건 분기를 통해 LLM 기반 판단을 안전하게 통제할 수 있다
- LangGraph는 복잡한 LLM 워크플로우를 **설계 가능한 구조**로 만든다

참고자료
Huggingface, agents course, https://huggingface.co/learn