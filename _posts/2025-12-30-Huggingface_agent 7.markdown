---
layout: post
title:  "허깅페이스 에이전트 코스 - Action"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Actions: Enabling the Agent to Engage with Its Environment

앞선 글에서는 **Thought(내부 추론)** 이 어떻게 다음 행동을 결정하는지 살펴보았다.  
이제 그 결정이 **현실 세계(또는 외부 시스템)**에 실제로 반영되는 단계가 바로 **Action**이다.

이 글에서는 다음을 정리한다.

- Action이 무엇인지
- Agent가 Action을 표현하는 방식
- 왜 “stop and parse”가 필수적인 설계 원칙인지
- JSON Agent와 Code Agent의 차이

## 실습 목표

- Action을 “환경과의 인터페이스”로 이해
- LLM이 Action을 **직접 실행하지 않는다**는 구조 명확히 하기
- Stop and Parse 패턴의 필요성 이해
- JSON 기반 Action과 Code 기반 Action의 장단점 비교

## Action이란 무엇인가?

**Action은 Agent가 환경과 상호작용하기 위해 수행하는 구체적인 실행 단계**다.

예를 들면 다음과 같다.

- 웹 검색 수행
- 외부 API 호출
- 데이터베이스 조회
- 계산 실행
- 디지털/물리 장치 제어
- 다른 Agent 또는 사용자와의 통신

중요한 점은 이것이다.

> Action은 “의도”가 아니라 **실제 실행 가능한 명령**이다.

LLM은 오직 텍스트만 생성하지만,  
Agent 시스템은 그 텍스트를 해석해 **현실의 동작으로 변환**한다.

## Agent Action의 유형

### Action을 생성하는 Agent의 형태

Agent는 Action을 표현하는 방식에 따라 다음과 같이 나뉜다.


| Agent 유형 | 설명 |
|---|---|
| JSON Agent | Action을 JSON 구조로 출력 |
| Code Agent | 실행 가능한 코드 블록을 출력 |
| Function-calling Agent | JSON Agent의 특화 형태(함수 호출에 최적화) |


### Action이 수행하는 목적별 분류

Action 자체는 다음과 같은 목적을 가질 수 있다.

| Action 목적 | 설명 |
|---|---|
| Information Gathering | 검색, DB 조회, 문서 검색 |
| Tool Usage | API 호출, 계산 실행 |
| Environment Interaction | UI 조작, 디바이스 제어 |
| Communication | 사용자 응답, 다른 Agent와 협업 |


## 중요한 전제: LLM은 Action을 실행하지 않는다

반드시 기억해야 할 구조적 사실이 있다.

> **LLM은 Action을 “설명”할 뿐,  
> Action을 “실행”하지 않는다.**

즉,

- LLM -> “이 Tool을 이 인자로 호출하라”는 텍스트 생성
- Agent Runtime -> 그 텍스트를 파싱해서 실제 실행

이 분리가 무너지면:

- 출력이 파싱 불가능해지고
- Tool 오작동이 발생하며
- Agent 전체가 불안정해진다

그래서 등장하는 것이 **Stop and Parse** 접근법이다.

## Stop and Parse Approach

**Stop and Parse**는 Agent Action 구현의 핵심 설계 원칙이다.

### 핵심 아이디어

1. Action을 **엄격히 구조화된 포맷**으로 출력
2. Action이 끝나면 **LLM의 출력을 즉시 중단(stop)**
3. 외부 파서가 그 결과를 **안전하게 파싱(parse)**

### 단계별 구조

#### 구조화된 출력

Agent는 Action을 다음처럼 명확한 포맷(JSON 또는 code)으로 출력한다.

```json
{
  "action": "get_weather",
  "action_input": {
    "location": "New York"
  }
}
```

#### 생성 중단 (Stop)

LLM은 Action 정의가 끝난 시점에서
추가 토큰을 생성하지 않아야 한다.

- 설명 섞임 X
- 자연어 덧붙임 X
- 주석성 텍스트 X

#### 파싱 (Parse)

Agent Runtime은 다음을 수행한다.

- 어떤 Action인지 식별
- 어떤 Tool을 호출해야 하는지 결정
- 인자를 안전하게 추출
- 실제 Tool 실행

이 방식의 장점은 명확하다.

- 출력 예측 가능
- 에러 감소
- 보안 강화
- Tool 연동 안정성 확보

## JSON Agent

JSON Agent는 Action을 JSON 객체로 표현한다.

### 예시: 날씨 조회 Action

```json
Thought: I need to check the current weather for New York.
Action:
{
  "action": "get_weather",
  "action_input": {
    "location": "New York"
  }
}
```

- action: 호출할 Tool 이름
- action_input: Tool에 전달할 인자

Function-calling Agent는 이 방식을
모델 학습 단계에서 더 강하게 고정한 형태라고 보면 된다.

### Code Agent

Code Agent는 JSON 대신
실행 가능한 코드 블록을 생성한다.

#### Code Agent의 개념

- Action = 코드 실행
- 코드 안에서:
  - API 호출
  - 데이터 처리
  - 조건 분기
  - 반복 처리
  - 결과 출력

#### Code Agent 예시

```python
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)
```

이 코드 블록은 다음을 수행한다.

- 외부 API 호출
- 결과 처리
- 최종 응답 생성

여기서도 중요한 점은:

- 코드 블록의 시작과 끝이 명확해야 한다.

그래야 Runtime이:

- 어디까지 실행해야 하는지
- 어떤 출력을 Observation으로 삼아야 하는지

를 정확히 판단할 수 있다.

#### Code Agent의 장점과 주의점

장점

- 표현력 높음 (조건문, 반복문, 함수)
- 복잡한 로직에 유리
- 외부 라이브러리와 직접 통합 가능
- 디버깅 용이

주의점

- LLM 생성 코드 실행은 보안 리스크가 큼
- Prompt Injection 위험
- 악성 코드 실행 가능성

따라서 실전에서는:

- 샌드박스 실행
- 권한 제한
- 검증 레이어

가 기본으로 필요하다.

## Action의 본질적 역할

Action은 다음을 연결한다.

- Thought (의사결정)
- Observation (환경 피드백)

즉, Action은 Agent의 “생각”을
현실 세계에 실제로 반영하는 유일한 수단이다.

이 단계가 없다면 Agent는
아무리 똑똑해도 말만 하는 시스템에 그친다.

## 정리

- Action은 Agent가 환경과 상호작용하는 실행 단계다
- LLM은 Action을 실행하지 않고, “Action 정의 텍스트”를 생성한다
- Stop and Parse는 안정적인 Agent 설계를 위한 필수 패턴이다
- JSON Agent는 단순·안정적이고
- Code Agent는 강력하지만 보안 설계가 필수다

참고자료
Huggingface, agents course, https://huggingface.co/learn