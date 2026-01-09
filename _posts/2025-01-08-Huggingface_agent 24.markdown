---
layout: post
title:  "허깅페이스 에이전트 코스 - Creating Agentic Workflows in LlamaIndex"
date:   2025-01-08 00:10:22 +0900
categories: Huggingface_agent
---

# Creating Agentic Workflows in LlamaIndex

## 1. LlamaIndex Workflow란 무엇인가

LlamaIndex의 **Workflow**는  
에이전트의 “자율성”을 유지하면서도 **전체 실행 흐름을 구조적으로 제어**하기 위한 시스템이다.

핵심 개념은 다음과 같다.

- Workflow는 **Step들의 집합**
- Step은 **Event를 입력으로 받아 Event를 출력**
- 출력 Event가 다음 Step을 트리거
- 전체 흐름은 **이벤트 기반(Event-driven)** 으로 동작

즉, Workflow는:

> “에이전트의 사고·행동을  
> 명시적인 상태 전이(State Transition) 그래프로 만든 것”

이다.

## 2. Workflow의 핵심 장점

Workflow를 사용하면 다음 이점을 얻는다.

- 코드가 Step 단위로 명확히 분리됨
- 이벤트 기반 제어로 복잡한 분기/루프 가능
- 타입 힌트 기반으로 실행 경로가 안전하게 제한됨
- 공용 상태(Context)를 통한 상태 관리 가능
- 단일 에이전트 + 멀티 에이전트 모두 수용 가능

**Agent(자율성)** 과 **Pipeline(통제)** 사이의 균형점

## 3. 기본 Workflow 생성

### 설치

```bash
pip install llama-index-utils-workflow
```

### 최소 Workflow 예제

```python
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")

w = MyWorkflow(timeout=10, verbose=False)
result = await w.run()
```

구조 해설

- Workflow 상속 -> 실행 그래프 정의
- @step -> 실행 가능한 노드
- StartEvent -> 워크플로우 시작점
- StopEvent -> 워크플로우 종료 + 결과 반환
- run() -> StartEvent 자동 생성 후 실행

## 4. 여러 Step 연결하기 (Event 전달)
Step 간 데이터를 전달하려면 Custom Event를 정의한다.

### Custom Event 정의
```python
from llama_index.core.workflow import Event

class ProcessingEvent(Event):
    intermediate_result: str
```

### 다단계 Workflow 예제
```python
class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(timeout=10, verbose=False)
result = await w.run()
```

핵심 포인트

- 입력 Event 타입 -> 어떤 Step이 실행될지 결정
- 타입 힌트가 곧 실행 제어 규칙

## 5. 분기와 루프 (Union Type)
Workflow의 가장 강력한 기능은
타입 힌트를 이용한 분기·루프다.

### Loop Event 정의
```python
class LoopEvent(Event):
    loop_output: str
```

#### 루프가 있는 Workflow
```python
import random

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(
        self, ev: StartEvent | LoopEvent
    ) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        return StopEvent(result=f"Finished: {ev.intermediate_result
        
w = MultiStepWorkflow(verbose=False)
result = await w.run()
result
```
의미 해설
- StartEvent | LoopEvent: 처음이든 재시도든 step_one 실행

- ProcessingEvent | LoopEvent: 성공 시 다음 단계 / 실패 시 루프

- **프롬프트가 아니라 타입이 흐름을 결정**

## 6. Workflow 시각화
Workflow 구조를 HTML로 시각화할 수 있다.

```python
from llama_index.utils.workflow import draw_all_possible_flows

draw_all_possible_flows(w, "flow.html")
```

복잡한 분기 구조 디버깅에 매우 유용

## 7. Workflow 상태 관리 (Context)
Workflow 전체에서 공유되는 상태가 필요할 때
Context를 사용한다.

### Context 기반 상태 저장
```python
from llama_index.core.workflow import Context, StartEvent, StopEvent

@step
async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
    await ctx.store.set("query", "What is the capital of France?")

    query = await ctx.store.get("query")
    val = ...  # query 사용

    return StopEvent(result=val)
```

특징

- 모든 Step이 동일 Context 접근 가능
- Workflow 수준의 전역 상태
- Agent 메모리와 동일한 개념

## 8. AgentWorkflow: Workflow + Agent 자동화
지금까지는 수동 Step 정의였다.
하지만 실전에서는 Agent 간 협업이 더 자연스럽다.

이를 위해 LlamaIndex는
AgentWorkflow를 제공한다.

## 9. Multi-Agent Workflow (AgentWorkflow)

개념
- 여러 Agent를 하나의 Workflow로 묶음
- 하나의 root agent가 진입점
- Agent 간 handoff 자동 처리

### 기본 Multi-Agent Workflow 예제
```python
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
)

multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Multiplies two integers",
    system_prompt="You can multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Adds two integers",
    system_prompt="You can add numbers.",
    tools=[add],
    llm=llm,
)

workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
)

response = await workflow.run("Can you add 5 and 3?")
```

구조 해설
- root_agent가 첫 발화 담당
- 필요 시 다른 Agent에게 작업 위임
- Agent = Step + Tool + Reasoning 묶음

## 10. AgentWorkflow + 상태(State) 주입
Workflow 시작 시 초기 상태를 주입할 수 있다.

### 상태를 수정하는 Tool
```python
from llama_index.core.workflow import Context

async def add(ctx: Context, a: int, b: int) -> int:
    state = await ctx.store.get("state")
    state["num_fn_calls"] += 1
    await ctx.store.set("state", state)
    return a + b
```

### 상태 기반 AgentWorkflow
```python
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

ctx = Context(workflow)
response = await workflow.run("Can you add 5 and 3?", ctx=ctx)

state = await ctx.store.get("state")
print(state["num_fn_calls"])
```

의미

- Agent Tool이 Workflow 상태를 직접 수정
- 상태는 모든 Agent에게 공유
- Agent + Workflow = 상태 머신

11. 전체 구조 요약 (DL/시스템 관점)
개념 매핑


| 개념            | 의미                |
| ------------- | ----------------- |
| Step          | 실행 노드             |
| Event         | 상태 전이 신호          |
| Workflow      | 명시적 실행 그래프        |
| Context       | 전역 상태             |
| AgentWorkflow | Agent 기반 Workflow |
| Agent         | 추론 + Tool 실행 노드   |


언제 무엇을 쓰나?

- 단순 파이프라인 -> Workflow
- RAG + 분기 -> Workflow
- 복잡한 판단/위임 -> AgentWorkflow
- 실서비스 -> AgentWorkflow + 상태 + 평가

참고자료
Huggingface, agents course, https://huggingface.co/learn