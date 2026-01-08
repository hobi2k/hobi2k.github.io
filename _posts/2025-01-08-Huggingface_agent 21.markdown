---
layout: post
title:  "허깅페이스 에이전트 코스 - What are Components in LlamaIndex?"
date:   2025-01-08 00:10:22 +0900
categories: Huggingface_agent
---

# What are Components in LlamaIndex?

## 1. LlamaIndex에서 말하는 “Component”란 무엇인가

LlamaIndex의 **Component**는  
> “LLM 기반 애플리케이션을 구성하는 재사용 가능한 논리 블록”

이다.

Agent가 스스로 사고하고 행동하기 위해서는 단순히 LLM 하나만으로는 부족하다.  
Agent는 반드시 다음을 할 수 있어야 한다.

- 데이터를 불러오고
- 의미 단위로 나누고
- 검색 가능하게 만들고
- 질문에 맞는 정보를 찾고
- 그 정보를 근거로 답변을 생성하고
- 그 결과를 평가한다

이 모든 단계가 **컴포넌트 단위로 분리되어 있는 것이 LlamaIndex의 핵심 설계**다.

## 2. 왜 QueryEngine에 집중하는가

LlamaIndex에는 많은 컴포넌트가 있지만,  
Agentic RAG 관점에서 가장 중요한 것은 **QueryEngine**이다.

### QueryEngine의 역할

QueryEngine은 다음을 한 번에 수행하는 고수준 인터페이스다.

1. 질문을 받는다
2. 관련 데이터를 검색한다 (retrieval)
3. 검색된 컨텍스트를 조합한다
4. LLM을 사용해 답변을 생성한다

즉, QueryEngine은 **RAG 파이프라인의 실행 엔진**이다.

> Agent 입장에서 보면  
> QueryEngine = “질문하면, 근거 기반 답변을 만들어주는 도구”

## 3. RAG가 필요한 이유 (맥락 정리)

LLM은:
- 일반 지식은 강하지만
- 최신 정보, 사내 문서, 개인 데이터에는 접근할 수 없다

RAG(Retrieval-Augmented Generation)는 이 한계를 해결한다.

### RAG의 핵심 아이디어

- 질문과 관련된 **외부 데이터**를 먼저 찾고
- 그 데이터를 **LLM의 입력 컨텍스트로 제공**
- LLM이 “추측”이 아니라 “근거 기반”으로 답변하게 만든다

QueryEngine은 이 RAG 전체 흐름을 캡슐화한 컴포넌트다.-

## 4. RAG 파이프라인의 5단계 (핵심 구조)

LlamaIndex에서 RAG는 다음 5단계로 정리된다.

### 1. Loading (데이터 로딩)
- 파일, PDF, 웹, DB, API 등에서 데이터 수집
- LlamaHub를 통해 수백 가지 로더 사용 가능

대표 예시:
- `SimpleDirectoryReader`
- `LlamaParse`
- LlamaHub 로더들

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()
```

### 2. Indexing (인덱싱 / 임베딩)
문서를 그대로 쓰지 않고:

- 작은 의미 단위(Node)로 분할
- 각 Node를 벡터 임베딩으로 변환

이를 통해 “의미 기반 검색”이 가능해진다.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

nodes = await pipeline.arun(documents)
```

### 3. Storing (저장)
인덱싱된 데이터는 반드시 저장해야 한다.
그래야:

- 매번 재임베딩하지 않아도 되고
- 대규모 데이터도 다룰 수 있다

대표적인 벡터 스토어:

- Chroma
- FAISS
- Weaviate
- Pinecone

```python
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

db = chromadb.PersistentClient(path="./alfred_chroma_db")
collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=collection)
```

### 4. Querying (질의)
저장된 인덱스를 기반으로 질문을 수행한다.
Index는 여러 방식으로 변환 가능하다.

- as_retriever -> 검색 결과만 반환
- as_query_engine -> 질문 -> 답변
- as_chat_engine -> 대화형 + 메모리

Agent/RAG에서는 보통 as_query_engine을 사용한다.

```python
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
)

index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)

query_engine.query("What is the meaning of life?")
```

### 5. Evaluation (평가)
LLM 출력은 항상 신뢰할 수 없기 때문에 정량적 평가가 필수다.
LlamaIndex는 LLM 기반 평가기를 제공한다.

대표 Evaluator:

- FaithfulnessEvaluator (근거 충실성)
- AnswerRelevancyEvaluator (질문 적합성)
- CorrectnessEvaluator (정확성)

```python
from llama_index.core.evaluation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query("...")
eval_result = evaluator.evaluate_response(response)
eval_result.passing
```

## 6. Observability (관측)
복잡한 RAG/Agent 시스템에서는:

- 어디서 성능이 떨어지는지
- 어느 단계가 병목인지
- LLM이 어떤 컨텍스트를 사용했는지

를 추적 가능해야 한다.

LlamaIndex는 LlamaTrace(Arize Phoenix)와 연동된다.

```bash
pip install -U llama-index-callbacks-arize-phoenix
```

```python
코드 복사
import llama_index
import os

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key=PHOENIX_API_KEY"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)
```

## 7. 정리: Agent 관점에서 Component의 의미
- Component는 Agent의 사고 기관
- QueryEngine은 Agent의 “지식 질의 도구”
- RAG 파이프라인은 Agent가 헛소리를 하지 않게 만드는 안전장치
- RAG 파이프라인은 Agent가 헛소리를 하지 않게 만드는 안전장치



참고자료
Huggingface, agents course, https://huggingface.co/learn