---
layout: post
title:  "허깅페이스 에이전트 코스 - Observe"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Observe: Integrating Feedback to Reflect and Adapt

앞선 단계에서 우리는 **Action**이 어떻게 Agent의 생각을 현실 세계에 반영하는지 살펴보았다.  
이제 그 결과를 다시 받아들이는 단계가 바로 **Observation**이다.

Observation은 Agent가 **자신의 행동 결과를 인지하고, 이를 다음 Thought에 반영하기 위한 입력 단계**다.  
즉, Agent가 “보고, 배우고, 조정하는” 메커니즘의 핵심이다.

## 실습 목표

- Observation의 역할을 “환경 피드백 입력”으로 정확히 이해
- Observation이 왜 반드시 **컨텍스트에 추가되어야 하는지** 파악
- Observation의 다양한 형태 정리
- Thought–Action–Observation 사이클이 왜 수렴 구조를 가지는지 이해

## Observation이란 무엇인가?

**Observation은 Agent가 수행한 Action의 결과로 환경에서 돌아오는 모든 신호(signal)**를 의미한다.

이 신호는 다음과 같은 형태일 수 있다.

- API 응답 데이터
- 성공/실패 메시지
- 에러 로그
- 시스템 상태 변화
- 센서 값

핵심 정의는 다음과 같다.

> Observation은  
> **Agent가 자신의 행동이 어떤 결과를 낳았는지 “인지(perceive)”하는 단계**다.

## Observation 단계에서 Agent가 하는 일

Observation 단계에서 Agent는 다음 과정을 거친다.

### Feedback 수집

- Action이 성공했는지, 실패했는지 확인
- 어떤 데이터가 반환되었는지 수신

예:
- “partly cloudy, 15°C, 60% humidity”
- “HTTP 403 Forbidden”
- “Database connection timeout”

### 컨텍스트에 결과 추가

Observation은 단순히 읽고 버려지는 값이 아니다.

> **Observation은 새로운 메시지로 프롬프트 끝에 추가된다.**

즉, Agent의 “기억”은 다음처럼 확장된다.

- 이전 대화
- Thought
- Action
- **Observation (새로 추가)**

이 구조 덕분에 LLM은  
“방금 무슨 일이 일어났는지”를 다음 Thought에서 참고할 수 있다.

### 전략 조정

Observation을 바탕으로 Agent는 판단을 갱신한다.

- 목표 달성 여부 판단
- 추가 정보가 필요한지 결정
- 다른 Tool을 써야 하는지 판단
- 오류 복구 전략 선택

이로써 Agent는 **정적 규칙 실행기**가 아니라  
**피드백 기반 적응 시스템**이 된다.

## Observation 예시: 날씨 Agent

Action 결과로 다음과 같은 API 응답이 왔다고 가정하자.

```
partly cloudy, 15°C, 60% humidity
```

이 값은 Observation 메시지로 추가된다.
그 결과 Agent는 다음을 판단할 수 있다.

- 이미 충분한 정보가 있다 -> 바로 최종 응답 생성
- 또는
- 바람, 강수 확률 등이 필요하다 -> 추가 Tool 호출

즉,

> Observation은 “다음 Thought의 재료”다.


## Observation의 다양한 형태

Observation은 단순 텍스트 응답에 국한되지 않는다.


| Observation 유형 | 예시 |
|---|---|
| System Feedback | 에러 메시지, 성공 알림, 상태 코드 |
| Data Changes | DB 업데이트 결과, 파일 생성/삭제 |
| Environmental Data | 센서 값, 시스템 메트릭, 리소스 사용량 |
| Response Analysis | API 응답, 쿼리 결과, 계산 출력 |
| Time-based Events | 타임아웃 발생, 스케줄 작업 완료 |


중요한 점은 다음이다.

> **모든 Observation은 “텍스트로 표현된 환경 로그”로 취급된다.**

그래야 LLM이 이를 읽고 추론에 활용할 수 있다.

## Observation은 어떻게 프롬프트에 추가되는가?

Agent 프레임워크는 Action 이후 다음 순서를 따른다.

1. **Action 파싱**
   - 어떤 Tool을 호출할지
   - 어떤 인자를 사용할지 결정

2. **Action 실행**
   - 실제 함수 / API / 코드 실행

3. **Observation 추가**
   - 실행 결과를 Observation 메시지로 기록
   - 다음 LLM 호출 시 프롬프트에 포함

이 흐름은 항상 **자동**으로 이루어진다.

LLM은 Observation을 “생성”하지 않고,  
Agent Runtime이 Observation을 **주입**한다.

## 왜 Observation이 중요한가?

Observation이 없다면 Agent는 다음 상태가 된다.

- 자신의 행동 결과를 모름
- 실패/성공 구분 불가
- 같은 Action을 반복
- 환경 변화에 대응 불가

즉,

> **Observation 없는 Agent는  
> 피드백 없는 무한 루프에 빠진다.**

Observation이 있기 때문에 Agent는:

- 오류를 인지하고 복구
- 계획을 수정
- 목표 달성 시 종료

할 수 있다.

## Thought–Action–Observation Cycle의 완성

이제 세 단계를 다시 연결해보면 다음과 같다.

1. **Thought**
   - 다음 행동을 결정
2. **Action**
   - 환경에 실제 개입
3. **Observation**
   - 결과를 받아 컨텍스트 갱신

이 루프는 목표가 달성될 때까지 반복된다.

> 이 반복 구조가  
> Agent를 “지능형 시스템”으로 만든다.


## 정리

- Observation은 Action의 결과로 돌아오는 **환경 피드백**이다
- Observation은 프롬프트에 추가되어 Agent의 “기억”을 갱신한다
- 이를 통해 Agent는 실패를 인지하고 전략을 수정할 수 있다
- Thought–Action–Observation 사이클은 **피드백 기반 수렴 구조**다

이제 모든 조각이 맞춰졌다.

- Thought: 무엇을 할지 결정
- Action: 실제로 실행
- Observation: 결과를 보고 조정

참고자료
Huggingface, agents course, https://huggingface.co/learn