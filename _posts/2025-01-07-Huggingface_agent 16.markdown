---
layout: post
title:  "허깅페이스 에이전트 코스 - Building Agentic RAG Systems"
date:   2025-01-07 00:10:22 +0900
categories: Huggingface_agent
---

# Building Agentic RAG Systems

## 1. RAG란 무엇인가

**RAG (Retrieval Augmented Generation)** 는  
정보 검색(Retrieval)과 텍스트 생성(Generation)을 결합한 구조다.

기본적인 RAG의 흐름은 다음과 같다.

1. 사용자의 질의(Query)를 받는다
2. 검색 엔진 또는 지식 저장소에서 관련 문서를 검색한다
3. 검색 결과를 LLM 입력 컨텍스트로 함께 제공한다
4. LLM이 검색 결과를 참고해 답변을 생성한다

즉, **LLM이 모르는 정보를 외부에서 가져와 보완**하는 구조다.

## 2. 기존 RAG의 구조적 한계

전통적인 RAG는 다음과 같은 한계를 가진다.

### 1. 단일 Retrieval 단계
- 검색은 보통 **한 번만 수행**
- 잘못된 검색 결과가 나오면 그대로 답변 품질이 무너짐

### 2. 질의 그대로 검색
- 사용자의 원 질의를 그대로 벡터 검색
- 의미는 맞지만 **맥락적으로 중요한 정보**를 놓칠 수 있음

### 3. 검색 품질에 대한 비판·검증 없음
- 검색 결과가 적절한지 판단하지 않음
- 불필요하거나 부정확한 정보도 그대로 사용

## 3. Agentic RAG란 무엇인가

**Agentic RAG**는 RAG에 **에이전트의 자율성**을 결합한 구조다.

핵심 차이점은 다음과 같다.


| 구분 | 전통적 RAG | Agentic RAG |
|---|---|---|
| 검색 횟수 | 1회 | 다중 단계 |
| 질의 | 사용자 질의 그대로 | 에이전트가 재작성 |
| 판단 | 없음 | 검색 결과 비판·선별 |
| 흐름 제어 | 고정 파이프라인 | 에이전트가 제어 |


즉, Agentic RAG에서는  
**LLM이 “검색 전략” 자체를 사고(Thought)로 결정**한다.

## 4. Agentic RAG의 핵심 능력

Agentic RAG에서 에이전트는 다음을 수행할 수 있다.

- 검색 질의 재작성
- 검색 결과 비판 및 필터링
- 추가 검색 여부 판단
- 여러 출처 정보 통합
- 필요한 경우 다단계 검색 반복

이는 `smolagents`의 **CodeAgent + Tool 구조**와 매우 잘 맞는다.

## 5. 기본 예제: DuckDuckGo 기반 Agentic RAG

아래는 웹 검색을 통해 정보를 수집하고 종합하는  
가장 단순한 Agentic RAG 예제다.

### 코드

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

# 검색 도구 초기화
search_tool = DuckDuckGoSearchTool()

# 모델 초기화
model = InferenceClientModel()

# Agent 생성
agent = CodeAgent(
    model=model,
    tools=[search_tool],
)

# 실행
response = agent.run(
    "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
)

print(response)
```

이 Agent가 수행하는 내부 단계

- 질의 분석
    - “luxury”
    - “superhero-themed party”
    - “decorations / entertainment / catering”
- 검색 수행
    - DuckDuckGoSearchTool 호출
    - 최신 웹 정보 검색
- 정보 통합
    - 검색 결과를 요약
    - 하나의 응답으로 재구성
- 메모리 반영
    - 이후 질의에서 재사용 가능

## 6. 커스텀 지식 기반을 활용한 Agentic RAG

웹 검색만으로는 부족한 경우,
사내 문서, 매뉴얼, 도메인 지식을 기반으로 한 RAG가 필요하다.

이를 위해 벡터 DB + Retriever Tool을 사용한다.

## 7. Vector Database 개념 정리

- 문서를 임베딩 벡터로 변환
- 의미적으로 가까운 문서를 검색
- 키워드 매칭이 아닌 의미 기반 검색

이 예제에서는 간단하게 BM25 Retriever를 사용한다.

## 8. 커스텀 Retriever Tool 구현 예제

개념 구조

- Tool이 내부적으로 Retriever를 보유
- Agent는 Tool을 호출해 문서 검색
- 검색 결과를 문자열로 반환

코드: PartyPlanningRetrieverTool

```python
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool, CodeAgent, InferenceClientModel

class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = (
        "Uses semantic search to retrieve relevant party planning ideas "
        "for Alfred’s superhero-themed party at Wayne Manor."
    )

    inputs = {
        "query": {
            "type": "string",
            "description": "Query related to party planning or superhero themes.",
        }
    }

    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs,
            k=5  # 상위 5개 문서 반환
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string"

        docs = self.retriever.invoke(query)

        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {i} =====\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )
```

## 9. 지식 베이스 구성

```python
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor.", "source": "Party Ideas"},
    {"text": "Hire a professional DJ for superhero-themed music.", "source": "Entertainment"},
    {"text": "Serve dishes named after superheroes.", "source": "Catering"},
    {"text": "Decorate with Gotham skyline projections.", "source": "Decoration"},
    {"text": "Interactive VR superhero experiences.", "source": "Entertainment"},
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]
```

## 10. 문서 분할 (Chunking)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = text_splitter.split_documents(source_docs)
```

## 11. Agent 실행
```python
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)

agent = CodeAgent(
    tools=[party_planning_retriever],
    model=InferenceClientModel()
)

response = agent.run(
    "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
)

print(response)
```

## 12. Agentic RAG의 고급 Retrieval 전략
Agentic RAG에서는 다음 전략을 조합할 수 있다.

1) Query Reformulation
- “luxury superhero party” -> “high-end themed event planning with superhero IP”

2) Query Decomposition
- 하나의 질의를 decor, entertainment,catering로 분리

3) Query Expansion
- 동일 의미 질의를 여러 표현으로 검색

4) Reranking
- Cross-Encoder로 문서 재정렬

5) Multi-Step Retrieval
- 1차 검색 -> 결과 분석 -> 2차 검색

6) Source Integration
- 웹 + 내부 문서 결합

7) Result Validation
- 검색 결과 신뢰도 평가 후 사용

## 13. 설계 시 고려 사항

- Agent가 질의 유형에 따라 Tool을 선택해야 한다
- 메모리로 중복 검색을 방지해야 한다
- 검색 실패 시 fallback 전략이 필요하다
- 결과 검증 단계가 있으면 품질이 크게 향상된다

## 14. 정리

Agentic RAG는 단순한 “검색 + 생성”이 아니다.

- 검색 전략을 사고하고
- 결과를 비판하고
- 필요하면 다시 검색하며
- 최종 답변을 스스로 구성하는

능동적 정보 시스템이다.

참고자료
Huggingface, agents course, https://huggingface.co/learn