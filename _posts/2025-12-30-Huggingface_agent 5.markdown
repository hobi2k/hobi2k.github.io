---
layout: post
title:  "허깅페이스 에이전트 코스 - Thought–Action–Observation Cycle"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Understanding AI Agents through the Thought–Action–Observation Cycle

앞선 글들에서 우리는 다음을 정리했다.

- Tool은 **System Message**를 통해 LLM에게 “알려진다”.
- Agent는 단순한 챗봇이 아니라, **추론(Reasoning), 계획(Planning), 환경 상호작용(Interaction)**을 수행하는 시스템이다.

이 글에서는 Agent의 동작을 관통하는 핵심 패턴인  
**Thought–Action–Observation(생각–행동–관찰) 사이클**을 통해
전체 Agent Workflow를 한 번에 정리한다.

## 실습 목표

- Agent의 실행 흐름을 “루프(loop)”로 이해하기
- Thought / Action / Observation의 책임 분리하기
- System Message에 Workflow 규칙을 “박아넣는” 이유 이해하기
- 간단한 예시(날씨 Agent)로 end-to-end 동작 흐름을 재구성하기

## 핵심 개념: Agent는 루프다

Agent는 단발성으로 답변을 끝내는 시스템이 아니라,  
목표가 달성될 때까지 반복하는 **제어 루프(control loop)**로 보는 것이 정확하다.

> Agent = Think -> Act -> Observe를 반복하는 시스템  
> (목표 달성 시 루프 종료)

프로그래밍 관점에서는 다음과 유사하다.

- `while` 루프가 돌면서
- 매 반복마다 “다음 행동”을 갱신하고
- 환경 피드백을 읽고
- 계획을 업데이트한다

## Core Components (3요소)

### Thought (생각)

- LLM(Brain)이 수행
- “지금 상태에서 다음에 뭘 해야 하는가?”를 결정
- 필요하다면 계획을 쪼개고(분해), 우선순위를 정함

핵심 역할:

- 문제 분해
- 다음 행동 선택
- Tool 사용 필요 여부 판단

### Action (행동)

- Agent(Runtime)가 수행
- LLM이 “이 Tool을 이 인자로 호출하라”는 형태의 출력을 만들면
- Agent가 그 출력을 해석하여 실제로 Tool을 실행

핵심 역할:

- Tool 호출
- 인자 전달
- 실행 실패/성공 처리

### Observation (관찰)

- 환경에서 돌아온 결과(도구 실행 결과)를 의미
- Tool 실행 결과를 **새 메시지**로 기록하여 LLM에 다시 제공

핵심 역할:

- 외부 데이터 주입(실시간 정보, 계산 결과, DB 조회 결과 등)
- “행동이 성공했는지/실패했는지” 피드백 제공
- 다음 Thought의 근거가 되는 컨텍스트 제공

## Thought–Action–Observation Cycle

세 요소는 다음 순서로 계속 반복된다.

1. Thought: 다음 스텝 결정
2. Action: Tool 호출(또는 작업 실행)
3. Observation: 결과 수신 및 컨텍스트 업데이트
4. (목표 달성 여부 판단 후) 반복 또는 종료

중요한 구현 포인트는 다음이다.

> 많은 Agent 프레임워크는  
> 이 루프의 규칙을 **System Message에 직접 포함**시켜  
> 매 턴이 같은 논리로 진행되게 만든다.

즉, System Message에는 보통 다음이 들어간다.

- Agent 행동 규칙(페르소나/제약)
- 사용 가능한 Tool 목록과 스펙
- Thought/Action/Observation을 어떤 형식으로 출력할지에 대한 규칙

## 예시: Alfred the Weather Agent

Alfred는 **날씨 API Tool**을 사용할 수 있는 Weather Agent다.

사용자 질문:

> “What’s the current weather in New York?”

Alfred는 “기억으로 답변”하면 안 된다.  
실시간 정보는 Tool을 통해 가져와야 한다.

### Step 1. Thought

Alfred(LLM)의 내부 판단은 대략 다음 흐름이다.

- 사용자는 “현재” 뉴욕 날씨를 원한다
- 내 지식은 고정되어 있고 최신이 아닐 수 있다
- 그러므로 `get_weather(location)` Tool을 호출해야 한다

핵심은 다음이다.

> Thought 단계는 “답변 생성”이 아니라  
> “다음 행동을 결정”하는 단계다.

### Step 2. Action (Tool Call)

LLM은 Tool을 직접 실행하지 못하므로,  
Tool 호출을 나타내는 텍스트(예: JSON)를 생성한다.

예시:

```json
{
  "action": "get_weather",
  "action_input": {
    "location": "New York"
  }
}
```

여기서 중요한 것은 Tool 호출이 명확하고 기계적으로 파싱 가능해야 한다는 점이다.

- 어떤 Tool인지: get_weather
- 인자가 무엇인지: location = "New York"

Agent(Runtime)는 이 출력을 읽고 실제 API 호출을 수행한다.

### Step 3. Observation (Tool Result)

Tool이 실행되면, 환경은 결과를 반환한다.

예시:

```
“Current weather in New York: partly cloudy, 15°C, 60% humidity.”
```

이 값은 Observation 메시지로 대화 컨텍스트에 추가된다.
즉, 다음 턴에서 LLM이 읽을 입력(prompt)이 이렇게 확장된다.

기존 대화 + Tool call + Tool result(Observation)

### Step 4. Updated Thought (반영/갱신)

LLM은 새로 들어온 Observation을 보고 판단을 갱신한다.

- 이제 뉴욕 날씨 데이터가 확보되었다
- 사용자가 원하는 답변을 정리해 전달하면 된다

### Step 5. Final Answer (종료)

이제는 Tool 호출이 아니라, 사용자에게 자연어 답변을 생성한다.

```
“The current weather in New York is partly cloudy with a temperature of 15°C and 60% humidity.”
```

이 순간 목표가 달성되었으므로 루프가 종료된다.

### 이 예시가 보여주는 것

1. Agent는 목표 달성까지 반복한다

만약 Observation이 다음과 같았다면?

- API 에러
- 위치 인식 실패
- 데이터 누락

Agent는 루프를 다시 돌릴 수 있다.

- Thought: 오류 원인 분석
- Action: 다른 Tool 사용, 재시도, 인자 수정
- Observation: 결과 확인

즉, Agent는 “한 번에 답을 맞히는 모델”이 아니라
“피드백을 통해 수렴하는 시스템”이다.

2. Tool 통합이 “정적 지식”을 넘어선다

LLM 자체는 학습 이후 지식이 고정되어 있다.
Tool은 다음을 가능하게 만든다.

- 실시간 정보 조회
- 외부 시스템 조작
- 정확한 계산 수행

Agent가 실무에서 쓸모 있어지는 이유가 여기 있다.

3. Observation이 추론을 업데이트한다

Observation은 단순 결과가 아니라,
다음 Thought의 근거가 되는 환경 피드백이다.

Thought는 Action을 만들고,
Observation은 Thought를 업데이트한다.

이 상호작용이 반복되면서
Agent는 더 복잡한 문제를 단계적으로 해결할 수 있다.

4. ReAct와의 연결

이 구조는 흔히 말하는 ReAct(Reasoning + Acting) 패턴의 핵심 아이디어와 맞닿아 있다.

- Reasoning(Thought)으로 계획하고
- Acting(Action)으로 환경을 변화시키고
- Observation으로 피드백을 받아
- 다시 Reasoning을 갱신한다

## 정리

- Agent는 Thought–Action–Observation 루프를 돈다
- Thought는 “다음 행동 결정”, Action은 “Tool 실행”, Observation은 “환경 피드백”이다
- System Message는 이 루프 규칙과 Tool 스펙을 고정해준다

이 반복 구조 덕분에 Agent는 복잡한 작업을 단계적으로 해결할 수 있다

참고자료
Huggingface, agents course, https://huggingface.co/learn