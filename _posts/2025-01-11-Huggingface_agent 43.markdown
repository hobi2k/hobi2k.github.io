---
layout: post
title:  "허깅페이스 에이전트 코스 - AI Agent Observability & Evaluation 실습"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# AI Agent Observability & Evaluation 실습

이 글에서는 **AI Agent의 내부 동작을 관측(Observability)**하고  
**성능을 평가(Evaluation)**하는 방법을 실제 코드와 함께 다룬다.

핵심 목표는 다음 두 가지다.

1. 에이전트가 **어떤 내부 단계를 거쳐 답을 생성했는지 추적**
2. 그 결과가 **비용·지연·정확도 측면에서 적절한지 평가**

이는 단순한 데모를 넘어,  
**에이전트를 프로덕션 수준으로 끌어올리기 위한 필수 단계**다.

## 실습 전제 조건 (Prerequisites)

이 실습은 다음 내용을 이미 이해하고 있다는 전제 하에 진행된다.

- Agents 개념 (계획, 행동, 관찰)
- smolagents 프레임워크 구조
  - Agent
  - Tool
  - Model

즉, *“Agent가 어떻게 동작하는지”* 는 이미 알고 있고,  
이제는 *“그 동작을 어떻게 관측·평가할 것인가”* 가 핵심이다.

## Step 0. 필수 라이브러리 설치

실습에는 다음 범주의 라이브러리가 필요하다.

- **Agent 실행**: smolagents
- **관측(telemetry)**: OpenTelemetry 기반 instrumentation
- **Observability 플랫폼**: Langfuse
- **UI/실습 보조**: Gradio, datasets

```python
%pip install langfuse 'smolagents[telemetry]' openinference-instrumentation-smolagents datasets 'smolagents[gradio]' gradio --upgrade
```

이 단계의 목적은 **“에이전트 실행 + 추적”이 동시에 가능하도록 환경을 구성**하는 것이다.

## Step 1. Agent Instrumentation (관측 장치 달기)

### 1-1. Langfuse 환경 변수 설정

Langfuse는 Observability 데이터를 수집하는 서버다.  
에이전트가 생성하는 trace/span을 이 서버로 전송하기 위해 API 키를 설정한다.

```python
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
```

또한 Hugging Face Inference 호출을 위해 HF_TOKEN도 필요하다.

### 1-2. Langfuse 클라이언트 초기화

```python
langfuse = get_client()
langfuse.auth_check()
```

이 단계는 **Observability 파이프라인이 정상 연결되었는지 확인**하는 용도다.

### 1-3. smolagents Instrumentation 활성화

```python
SmolagentsInstrumentor().instrument()
```

이 한 줄이 매우 중요하다.

- 이후 실행되는 **모든 smolagent**
- 내부의 **LLM 호출 / Tool 호출**
- 실행 시간, 토큰 사용량

전부 자동으로 **Trace / Span**으로 기록된다.

## Step 2. Instrumentation 테스트 (가장 단순한 Agent)

가장 단순한 CodeAgent를 실행해 instrumentation이 잘 동작하는지 확인한다.

```python
agent.run("1+1=")
```

이때 Langfuse 대시보드에서 다음이 보이면 성공이다.

- 하나의 Trace
- 내부에 LLM 호출 Span

이 단계는 **“관측 장비가 제대로 달렸는지” 확인하는 단계**다.

## Step 3. 더 복잡한 Agent 관측하기

이번에는 Tool(DuckDuckGoSearchTool)을 사용하는 Agent를 실행한다.

```python
agent.run("How many Rubik's Cubes could you fit inside the Notre Dame Cathedral?")
```

이 경우 Trace 구조는 다음처럼 보인다.

- Root Trace: Agent 실행 전체
- 하위 Span:
  - 검색 Tool 호출
  - LLM 호출
  - 후속 reasoning 호출

이를 통해 다음을 분석할 수 있다.

- 어떤 단계가 오래 걸리는지
- Tool 호출이 과도한지
- 비용이 어디서 발생하는지

## Online Evaluation (운영 중 평가)

Online Evaluation은 **실제 사용자 사용 중** 에이전트를 평가하는 방식이다.

### 운영 환경에서 주로 보는 지표

1. **Cost**
   - 토큰 사용량 기반 비용
   - 모델별 비용 비교

2. **Latency**
   - 전체 응답 시간
   - 단계별 병목

3. **User Feedback**
   - 👍 / 👎
   - 별점

4. **LLM-as-a-Judge**
   - 출력의 정확성 / 유해성 / 스타일을
   - 또 다른 LLM이 평가

### Span에 메타데이터 추가하기

```python
span.update_trace(
    input=...,
    output=...,
    user_id=...,
    session_id=...,
    tags=[...],
)
```

이렇게 하면:

- 사용자별 분석
- 세션별 분석
- 특정 태그 기반 필터링

이 가능해진다.

### Gradio UI + User Feedback 연동

Gradio UI에서 사용자가 👍 / 👎 를 누르면  
그 정보가 **Langfuse Trace에 Score로 기록**된다.

이는 **정성적 평가를 정량화하는 핵심 포인트**다.

### LLM-as-a-Judge

LLM을 심판으로 사용해 다음을 자동 평가할 수 있다.

- 독성 여부
- 정확성
- 도움됨 정도

이 방식은 **사람이 매번 평가하기 어려운 상황**에서 매우 유용하다.

## Offline Evaluation (오프라인 평가)

Offline Evaluation은 **사전에 준비된 데이터셋**으로 평가한다.

### GSM8K 예제

- 질문 + 정답이 이미 존재
- 에이전트를 동일 조건으로 반복 실행
- 결과를 비교

Langfuse Dataset 기능을 사용하면:

- 데이터셋 단위 Trace 관리
- Run 간 비교
- 모델/프롬프트별 성능 비교

가 가능하다.

## Offline Evaluation 실행 흐름 요약

1. 벤치마크 데이터셋 준비
2. Langfuse에 Dataset 등록
3. 각 데이터 항목마다 Agent 실행
4. Trace를 Dataset Item과 연결
5. 결과 비교 및 분석

**CI / 모델 교체 시 필수 단계**

## 전체 정리

이 글에서 다룬 핵심은 다음이다.

1. Observability는 **“에이전트 내부를 보는 눈”**
2. Evaluation은 **“그 행동을 판단하는 기준”**
3. Online + Offline 평가를 함께 써야 한다
4. smolagents + Langfuse는 이를 빠르게 구현할 수 있는 조합이다

이제 에이전트는 더 이상 블랙박스가 아니다.  
**추적 가능하고, 평가 가능하며, 개선 가능한 시스템**이 된다.

참고자료
Huggingface, agents course, https://huggingface.co/learn