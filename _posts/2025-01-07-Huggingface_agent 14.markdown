---
layout: post
title:  "허깅페이스 에이전트 코스 - Writing Actions as Code Snippets or JSON Blobs"
date:   2025-01-07 00:10:22 +0900
categories: Huggingface_agent
---

# Writing Actions as Code Snippets or JSON Blobs

이 섹션에서는 `smolagents`에서 **Action을 표현하는 두 가지 방식**를 체계적으로 정리한다.

- **CodeAgent**: Action을 *실행 가능한 Python 코드*로 작성
- **ToolCallingAgent**: Action을 *JSON 구조*로 작성 (LLM의 built-in tool calling 활용)

이 둘은 동일한 Agent 개념(Thought -> Action -> Observation)을 공유하지만,  
**Action을 어떻게 표현하고 실행하느냐**에서 근본적인 차이가 있다.

## 1. ToolCallingAgent란 무엇인가

`ToolCallingAgent`는 `smolagents`에서 제공하는 **두 번째 Agent 유형**이다.  
이 Agent는 CodeAgent와 달리 Python 코드를 생성하지 않는다.

대신, OpenAI / Anthropic / 기타 LLM 제공자들이 지원하는  
**표준 Tool Calling 인터페이스**를 활용하여,  
Action을 **JSON 형태의 구조화된 데이터**로 생성한다.

핵심 요약:

- LLM 출력: JSON (tool 이름 + arguments)
- 시스템 역할: JSON을 파싱 -> 실제 Tool 실행
- 실행 결과: Observation으로 다시 LLM에 전달

이 방식은 현재 대부분의 상용 LLM API(OpenAI, Anthropic 등)에서  
기본적으로 채택하고 있는 방식이다.

## 2. CodeAgent vs ToolCallingAgent: Action 표현의 차이

### 2.1 동일한 목표, 다른 표현 방식

예를 들어 Alfred가 다음 두 작업을 수행해야 한다고 가정하자.

- 고담 시티의 케이터링 서비스 검색
- 슈퍼히어로 파티 테마 아이디어 검색

### CodeAgent 방식 (Python 코드)

```python
for query in [
    "Best catering services in Gotham City", 
    "Party theme ideas for superheroes"
]:
    print(web_search(f"Search for: {query}"))
```

이 경우:

- LLM은 실제 실행 가능한 Python 코드를 생성한다.
- 반복문, 변수, 문자열 포매팅 등 모든 로직이 코드 안에 포함된다.
- 실행기는 이 코드를 그대로 실행한다.

### ToolCallingAgent 방식 (JSON Action)

```python
[
    {"name": "web_search", "arguments": "Best catering services in Gotham City"},
    {"name": "web_search", "arguments": "Party theme ideas for superheroes"}
]
```

이 경우:

- LLM은 “무엇을 호출할지”만 선언한다.
- 반복, 조건, 변수 개념은 JSON 바깥(시스템 로직)에 있다.
- 시스템은 이 JSON을 파싱해서 tool을 하나씩 실행한다.

## 3. ToolCallingAgent가 동작하는 방식

ToolCallingAgent의 전체 흐름은 CodeAgent와 구조적으로 동일하다.

- Thought
- Action
- Observation
- 반복 (multi-step)

차이점은 Action 단계에서 무엇을 생성하느냐이다.

### 공통점 (CodeAgent와 동일)

- Multi-step Agent
- ReAct 스타일 루프
- Observation을 다음 Thought의 입력으로 사용
- System Prompt에 Tool 정보 포함

### 차이점 (Action 표현)


| 항목         | CodeAgent | ToolCallingAgent  |
| ---------- | --------- | ----------------- |
| Action 출력  | Python 코드 | JSON 구조           |
| 실행 방식      | 코드 직접 실행  | JSON 파싱 후 Tool 호출 |
| 로직 표현      | 매우 자유로움   | 제한적               |
| 파싱 필요성     | 거의 없음     | 필수                |
| 표준 API 호환성 | 낮음        | 매우 높음             |


## 4. ToolCallingAgent 실행 예제

이제 실제로 ToolCallingAgent를 사용하는 예제를 살펴보자.
이 예제는 앞서 CodeAgent로 사용했던 웹 검색 시나리오를 그대로 재현한다.

### 4.1 ToolCallingAgent 생성

```python
from smolagents import ToolCallingAgent, WebSearchTool, InferenceClientModel

agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel()
)

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
```

코드 해설

- ToolCallingAgent
  - Action을 JSON으로 생성하는 Agent 타입
- WebSearchTool
  - DuckDuckGo 기반 웹 검색 Tool
- InferenceClientModel
  - Hugging Face Serverless Inference API를 사용하는 모델 래퍼

Agent 실행 자체는 CodeAgent와 완전히 동일한 인터페이스를 가진다.
차이는 내부 Action 생성 방식뿐이다.

### 4.2 실행 중 출력(Trace)의 차이

CodeAgent를 실행하면 보통 다음과 같은 로그가 나온다.

```
Executing parsed code:
results = web_search(query="...")
print(results)
```

반면 ToolCallingAgent에서는 다음과 같은 로그를 보게 된다.

```
╭────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'web_search' with arguments:                                  │
│ {'query': "best music recommendations for a party at Wayne's mansion"}     │
╰────────────────────────────────────────────────────────────────────────────╯
```

이 로그가 의미하는 바는 다음과 같다.

- LLM이 JSON 기반 Tool Call을 생성했다
- 시스템이 이를 파싱했다
- web_search Tool을 지정된 arguments로 실행했다
- 실행 결과는 Observation으로 기록되었다

즉, LLM은 실행을 직접 하지 않고 “지시”만 한다는 점이 핵심이다.

## 5. 언제 ToolCallingAgent를 써야 하는가

smolagents는 기본적으로 CodeAgent를 권장한다.
그 이유는 이미 앞에서 다뤘듯이, 코드 기반 Action이 전반적으로 더 강력하기 때문이다.

그럼에도 ToolCallingAgent가 적합한 경우는 분명히 존재한다.

### ToolCallingAgent가 적합한 경우

- Action이 단순한 단일 Tool 호출 위주일 때
- 변수/조건/반복이 거의 필요 없을 때
- OpenAI/Anthropic의 표준 Tool Calling API와 호환성이 중요할 때
- 외부 시스템이 이미 JSON 기반 Tool 호출 파이프라인을 갖고 있을 때
- 보안 정책상 “코드 실행”이 허용되지 않을 때

### ToolCallingAgent의 한계

- 복잡한 로직은 시스템 쪽에서 처리해야 함
- Action 간 조합성이 낮음
- JSON 파싱 실패, 스키마 불일치 리스크 존재
- 중간 결과를 다루는 데 불리함

## 6. CodeAgent vs ToolCallingAgent: 선택 가이드

정리하면 다음 기준으로 선택할 수 있다.

- 유연한 문제 해결 / 복잡한 멀티스텝 로직 -> CodeAgent
- 단순 Tool 호출 / API 표준 준수 -> ToolCallingAgent

smolagents가 CodeAgent를 “기본(default)”으로 설계한 이유는
Agent의 사고와 행동을 최대한 자연스럽게 확장하기 위함이다.

## 7. 정리

- smolagents에는 두 가지 Action 표현 방식이 존재한다.
- CodeAgent는 Action을 Python 코드로 생성한다.
- ToolCallingAgent는 Action을 JSON Tool Call로 생성한다.
- 두 Agent는 동일한 ReAct 구조를 공유하지만, Action의 표현력과 실행 방식에서 큰 차이가 난다.
- CodeAgent는 강력하고 유연하지만, ToolCallingAgent는 단순하고 표준적이다.

참고자료
Huggingface, agents course, https://huggingface.co/learn