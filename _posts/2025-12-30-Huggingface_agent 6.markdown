---
layout: post
title:  "허깅페이스 에이전트 코스 - Thought"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# Thought: Internal Reasoning and the ReAct Approach

이 글에서는 Agent Workflow(Thought–Action–Observation) 중 첫 단계인 **Thought**를 정리한다.  
Thought는 단순히 “생각을 많이 한다”가 아니라, **다음 행동을 결정하기 위한 내부 추론, 계획 단계**다.

또한 Thought를 유도하는 대표 프롬프팅 기법인

- **Chain-of-Thought(CoT)**
- **ReAct(Reasoning + Acting)**

을 비교 관점으로 정리한다.

## 실습 목표

- Thought의 역할을 “내부 추론/계획 모듈”로 이해
- Thought의 대표 유형(계획, 분석, 우선순위 등) 정리
- CoT와 ReAct의 차이와 적용 시나리오 구분
- “프롬프팅 기법”과 “학습(훈련) 기반 추론”의 차이 이해

## Thought란 무엇인가?

**Thought는 Agent의 내부 추론(Reasoning)과 계획(Planning)**을 의미한다.

- LLM이 프롬프트에 주어진 정보(대화 히스토리, Observation, 정책, 툴 스펙)를 바탕으로
- 현재 상황을 해석하고
- 다음에 어떤 행동(Action)을 할지 결정하는 과정이다

핵심은 다음이다.

> Thought는 “최종 답변을 쓰는 단계”가 아니라  
> “다음 스텝을 결정하는 제어 단계”다.

이 과정 덕분에 Agent는 다음을 수행할 수 있다.

- 복잡한 문제를 단계로 분해
- 과거 관찰(Observation)과 경험을 반영
- 새 정보가 들어오면 계획을 갱신
- 목표 달성을 위해 반복적으로 수렴

## Thought가 수행하는 일 (Agent 관점)

Thought 단계에서 실제로 일어나는 대표 작업은 다음과 같다.

- 문제를 서브태스크로 분해 (task decomposition)
- 현재 상태/제약 조건 확인
- 필요한 정보가 무엇인지 판단
- Tool 사용 여부 판단
- 다음 Action 선택 및 인자 구성 계획
- 실패 시 대안 전략 수립

## Examples: Common Thought Types

실무적으로 Thought는 다음과 같은 유형으로 나타난다.


| Thought 유형 | 예시 |
|---|---|
| Planning | “이 작업은 1) 데이터 수집 2) 분석 3) 리포트 생성으로 나누자.” |
| Analysis | “에러 메시지로 보아 DB 커넥션 파라미터 문제다.” |
| Decision Making | “예산이 제한적이니 미드티어 옵션을 추천하는 게 합리적이다.” |
| Problem Solving | “최적화 전, 먼저 프로파일링으로 병목을 찾아야 한다.” |
| Memory Integration | “사용자는 Python을 선호한다고 했으니 예시를 Python으로 제시하자.” |
| Self-Reflection | “방금 접근은 실패했다. 다른 전략으로 전환해야 한다.” |
| Goal Setting | “완료 조건(acceptance criteria)을 먼저 정의하자.” |
| Prioritization | “기능 추가 전에 보안 취약점을 먼저 처리해야 한다.” |


> 참고: Function-calling에 특화된 모델(또는 프레임워크)은  
> Thought를 **생략하거나 최소화**해도 동작할 수 있다.  
> (이 부분은 Actions 설계에서 더 중요해진다.)


## Chain-of-Thought(CoT): 단계적 내부 추론

**Chain-of-Thought(CoT)**는 모델이 최종 답을 내기 전에  
**문제를 단계적으로 풀도록 유도하는 프롬프팅 기법**이다.

대표적으로 다음과 같은 유도 문장을 사용한다.

- “Let’s think step by step.”

CoT의 특징은 명확하다.

- 단계적 논리 전개
- 외부 Tool 사용 없음 (내부 계산/추론 중심)
- 논리/수학/추론 문제에 특히 유효

### CoT 예시

```
Question: What is 15% of 200?
Thought: Let's think step by step. 10% of 200 is 20, and 5% of 200 is 10, so 15% is 30.
Answer: 30
```

## ReAct: Reasoning + Acting (추론과 행동의 결합)

ReAct는 CoT의 “생각”에 더해,
생각 중간에 행동(Action)을 끼워 넣는 방식이다.

즉, 다음을 반복한다.

- Thought: 다음에 무엇을 할지 판단
- Action: Tool 실행
- Observation: 결과 수신
- Thought: 결과 반영 후 다음 행동 결정

이 구조는 동적, 다단계 문제에 특히 강하다.

- 정보 탐색(검색)
- 최신 데이터 조회
- API 기반 작업
- 여러 단계로 쪼개지는 업무 자동화

### ReAct 예시

```python
Thought: I need to find the latest weather in Paris.
Action: Search["weather in Paris"]
Observation: It's 18°C and cloudy.
Thought: Now that I know the weather...
Action: Finish["It's 18°C and cloudy in Paris."]
```

여기서 핵심은, “생각만으로 끝내지 않고,
중간에 행동을 실행해 외부 피드백으로 추론을 갱신한다”는 점이다.

## ReAct vs CoT 비교


| 구분         | CoT (Chain-of-Thought) | ReAct (Reasoning + Acting)  |
| ---------- | ---------------------- | --------------------------- |
| 단계적 추론     | 가능                   | 가능                        |
| 외부 Tool 사용 | 없음                   | 있음 (Action + Observation) |
| 적합한 작업     | 논리/수학/내부 추론            | 정보탐색/최신데이터/다단계 업무           |
| 실패 시 복구    | 내부 논리로만                | Observation 기반으로 전략 전환 가능   |


“프롬프팅” vs “훈련 기반 추론”의 차이
CoT와 ReAct는 어디까지나 프롬프팅 기법이다.
즉, 모델에게 “이렇게 생각해라”라고 입력으로 유도하는 방식이다.

반면, 최근 일부 모델은 “생각 후 답변”을 학습 단계에서 내재화한다.

- 예시: DeepSeek R1, OpenAI o1 계열(추론 중심 계열)
- 특징:
  - reasoning 구간과 final answer를 분리하는 구조 토큰을 사용하기도 함
  - 이는 프롬프팅이 아니라 훈련 데이터로 습관화된 추론 패턴에 가깝다

정리하면
- CoT / ReAct: 프롬프트 설계로 추론을 유도
- 추론 특화 모델: 훈련으로 “생각 -> 답변” 패턴 자체를 내재화

## 정리

- Thought는 Agent의 내부 추론, 계획 단계이며, 다음 Action을 결정한다
- CoT는 “단계적으로 생각하고 답변”하는 프롬프팅 기법(내부 추론 중심)이다
- ReAct는 “생각 + 행동 + 관찰”을 번갈아 수행하는 Agent 친화적 패턴이다
- 최근 모델 일부는 프롬프팅이 아니라 “훈련”으로 think-before-answer를 내재화한다.

참고자료
Huggingface, agents course, https://huggingface.co/learn