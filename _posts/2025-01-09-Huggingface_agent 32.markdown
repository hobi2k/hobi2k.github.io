---
layout: post
title:  "허깅페이스 에이전트 코스 - Creating a RAG Tool for Guest Stories"
date:   2025-01-9 00:10:22 +0900
categories: Huggingface_agent
---

# Creating a RAG Tool for Guest Stories

이번 글에서는 **갈라(Gala)에 참석한 손님들의 정보를 즉시 조회**할 수 있도록,  
Alfred를 위한 **전용 RAG(Retrieval Augmented Generation) 도구**를 직접 구축한다.

이 예제의 핵심은 다음이다.

- 손님 전용 커스텀 데이터셋을 사용한다
- RAG를 **도구(tool)** 형태로 구현한다
- 해당 도구를 에이전트(Alfred)에 장착한다
- 실제 질의응답 시나리오에서 활용한다

즉, “RAG 파이프라인”이 아니라  
**Agentic RAG의 구성 요소 중 하나로서의 RAG Tool**을 만드는 단계다.

## 왜 갈라에 RAG가 필요한가?

Alfred가 파티 중 손님들과 어울리며 즉각적으로 정보를 떠올려야 하는 상황을 가정해보자.

일반적인 LLM만으로는 다음 문제를 겪는다.

1. 손님 목록은 **사전학습 데이터에 존재하지 않는다**
2. 손님 정보는 **수시로 변경**될 수 있다
3. 이메일 주소처럼 **정확성이 중요한 정보**가 필요하다

이러한 문제는 RAG로 해결할 수 있다.

> **RAG의 핵심 역할**  
> - 최신 데이터
> - 이벤트 전용 데이터
> - 정밀 검색이 필요한 정보  
> 를 LLM 응답에 안전하게 반영

## 애플리케이션 구성 방식

이후 글에서는 에이전트를 **Hugging Face Space**에서 실행 가능한  
**구조화된 Python 프로젝트**로 구성한다.

### 프로젝트 구조

- `tools.py`  
  -> Alfred가 사용할 보조 도구 정의
- `retriever.py`  
  -> 검색(Retrieval) 로직 구현
- `app.py`  
  -> 에이전트와 도구를 통합한 최종 실행 파일

이 구조는 실제 배포 환경에서도 그대로 사용할 수 있는 형태다.

## 데이터셋 개요

사용하는 데이터셋은 다음이다.

- `agents-course/unit3-invitees`

각 손님은 다음 필드를 가진다.

- **Name**: 손님 이름
- **Relation**: 호스트와의 관계
- **Description**: 약력 및 흥미로운 정보
- **Email Address**: 연락처

> 실제 환경에서는  
> 식이 제한, 관심사, 피해야 할 대화 주제 등도 추가 가능하다.

## Guestbook RAG Tool 구축 개요

전체 과정은 다음 3단계로 나뉜다.

1. 데이터셋 로드 및 문서화
2. Retriever Tool 생성
3. Alfred 에이전트에 통합

## Step 1: 데이터셋 로드 및 문서화

### LangChain 기반 Document 변환

```python
import datasets
from langchain_core.documents import Document

# 데이터셋 로드
guest_dataset = datasets.load_dataset(
    "agents-course/unit3-invitees",
    split="train"
)

# 각 손님 정보를 하나의 Document로 변환
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]
```

이 단계의 핵심은 다음이다.

- 정형 데이터 -> **텍스트 문서**로 변환
- 검색 대상이 되는 최소 단위는 `Document`
- 메타데이터는 추후 필터링 및 후처리에 활용 가능

## Step 2: Retriever Tool 생성 (BM25 기반)

### 왜 BM25인가?

- 임베딩 불필요
- 빠르고 단순
- 이름, 관계 등 키워드 검색에 적합

> 이후 semantic search가 필요해지면  
> sentence-transformers 기반 임베딩 검색으로 교체 가능

### LangChain + BM25Retriever + Tool

```python
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool

# BM25 Retriever 초기화
bm25_retriever = BM25Retriever.from_documents(docs)

def retrieve_guest_info(query: str) -> str:
    """
    손님 이름 또는 관계를 기반으로
    관련 정보를 검색하여 반환한다.
    """
    results = bm25_retriever.invoke(query)

    if results:
        return "\n\n".join(
            [doc.page_content for doc in results[:3]]
        )
    else:
        return "No matching guest information found."

# Tool로 래핑
guest_info_tool = Tool(
    name="guest_info_retriever",
    func=retrieve_guest_info,
    description=(
        "Retrieves detailed information about gala guests "
        "based on their name or relation."
    )
)
```

### 이 Tool의 역할

- 입력: 문자열 query
- 처리: BM25 기반 문서 검색
- 출력: 손님 정보 텍스트

에이전트 입장에서는  
**“손님 정보가 필요할 때 호출하는 도구”**로 인식된다.

## Step 3: Alfred 에이전트에 통합

아래는 서로 다른 프레임워크에서의 통합 예시다.  
핵심 개념은 모두 동일하다.

### 예시 1: smolagents 기반 CodeAgent

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()

alfred = CodeAgent(
    tools=[guest_info_tool],
    model=model
)

response = alfred.run(
    "Tell me about our guest named 'Lady Ada Lovelace'."
)

print(response)
```

### 예시 2: llama-index AgentWorkflow

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
)

alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm
)

response = await alfred.run(
    "Tell me about our guest named 'Lady Ada Lovelace'."
)

print(response)
```


### 예시 3: LangGraph 기반 Agentic RAG

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# LLM 초기화
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm)
chat_with_tools = chat.bind_tools([guest_info_tool])

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [
            chat_with_tools.invoke(state["messages"])
        ]
    }

# Graph 구성
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([guest_info_tool]))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge("tools", "assistant")

alfred = builder.compile()

messages = [
    HumanMessage(
        content="Tell me about our guest named 'Lady Ada Lovelace'."
    )
]

response = alfred.invoke({"messages": messages})
print(response["messages"][-1].content)
```

## 실제 갈라 중 상호작용 예시

**You**  
“Alfred, who is that gentleman talking to the ambassador?”

**Alfred**  
“That's Dr. Nikola Tesla, sir. He's an old friend from your university days…”

이 응답은 다음 JSON 데이터를 기반으로 생성된다.

```json
{
  "name": "Dr. Nikola Tesla",
  "relation": "old friend from university days",
  "description": "Recently patented a new wireless energy transmission system...",
  "email": "nikola.tesla@gmail.com"
}
```

## 확장 아이디어

현재 시스템을 다음 방향으로 확장할 수 있다.

1. 임베딩 기반 Retriever 도입
2. 대화 메모리 추가
3. 웹 검색과 결합
4. 다중 인덱스(검증된 출처) 통합
5. 대화 주제 추천 기능 추가

> 예:  
> “이 손님에게 어울리는 스몰토크 주제는?”


## 정리

- RAG는 **정확한 이벤트 전용 정보**를 제공한다
- Agentic RAG에서는 RAG가 **도구 중 하나**가 된다
- BM25 기반 Retriever는 간단하면서 효과적인 출발점이다
- 동일한 RAG Tool을 여러 에이전트 프레임워크에서 재사용할 수 있다

이제 Alfred는  
**갈라 현장에서 어떤 손님 질문에도 막힘없이 대응할 수 있는 에이전트**가 되었다.


참고자료
Huggingface, agents course, https://huggingface.co/learn