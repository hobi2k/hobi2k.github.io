---
layout: post
title:  "허깅페이스 에이전트 코스 - AI Agent Observability & Evaluation 정리"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

## AI Agent Observability & Evaluation 정리

이 섹션에서는 **AI Agent를 실제 서비스 수준으로 끌어올리기 위해 반드시 필요한 두 축**,  
즉 **Observability(관측 가능성)** 와 **Evaluation(평가)** 를 다룬다.

단순히 “에이전트가 답을 낸다”에서 끝나는 것이 아니라,  
**왜 그렇게 행동했는지**, **어디서 문제가 발생했는지**,  
**지금도 잘 동작하고 있는지**를 지속적으로 확인하는 단계다.

---

## 1. Observability란 무엇인가?

**Observability(관측 가능성)** 는  
에이전트 내부에서 일어나는 일을 **외부 신호를 통해 이해하는 능력**을 의미한다.

AI Agent의 관측 대상은 다음과 같다.

- LLM 호출 내역
- Tool 사용 여부와 순서
- 실행 시간(latency)
- 실패 지점
- 출력 결과

즉, Agent를 **블랙박스가 아닌, 추적 가능한 시스템**으로 만드는 것이 목적이다.

> Observability = “에이전트가 왜 그렇게 행동했는지 설명할 수 있는 능력”

---

## 2. 왜 Agent Observability가 중요한가?

Observability가 없는 에이전트는 다음과 같은 문제를 가진다.

- 응답이 느려도 이유를 모른다
- 비용이 폭증해도 원인을 모른다
- 잘못된 답이 나와도 재현이 안 된다
- 프롬프트 인젝션이나 유해 발화 탐지가 어렵다

반대로 Observability를 도입하면 다음이 가능해진다.

- 비용 ↔ 정확도 트레이드오프 분석
- Latency 병목 지점 파악
- Tool 루프 / API 오류 감지
- 사용자 피드백 기반 개선

👉 **데모 에이전트 → 프로덕션 에이전트로 가는 필수 단계**

---

## 3. 대표적인 Observability 도구들

대표적인 AI Agent Observability 도구는 다음과 같다.

- **Langfuse**
- **Arize**
- 기타 LLMOps 플랫폼

이들 도구는 공통적으로 다음을 제공한다.

- Agent 실행 전체를 하나의 **Trace**로 기록
- 내부 단계를 **Span** 단위로 분해
- 대시보드 기반 시각화

많은 Agent 프레임워크(예: smolagents, LangGraph)는  
**OpenTelemetry 표준**을 통해 이러한 도구들과 연동된다.

---

## 4. Trace와 Span 개념

Observability에서 가장 중요한 개념은 **Trace**와 **Span**이다.

### Trace
- 하나의 사용자 요청에 대한 **에이전트 전체 실행 흐름**
- 예: “질문 수신 → LLM 호출 → Tool 사용 → 최종 답변”

### Span
- Trace 내부의 **개별 단계**
- 예:
  - LLM 호출 1
  - Tool 호출
  - LLM 호출 2

LangGraph 기준으로 보면:
- **Graph 전체 실행 = Trace**
- **각 Node 실행 = Span**

---

## 5. Agent에서 반드시 모니터링해야 할 핵심 지표

### 5.1 Latency (응답 시간)
- 전체 응답 시간
- 단계별(LLM, Tool) 시간
- 병렬화 / 모델 교체 판단 근거

### 5.2 Cost (비용)
- 토큰 사용량
- Tool/API 호출 횟수
- 불필요한 LLM 재호출 감지

### 5.3 Request Errors
- LLM API 실패
- Tool 호출 실패
- 재시도 / fallback 설계 근거

### 5.4 User Feedback (명시적)
- 👍 / 👎
- 별점
- 코멘트

### 5.5 Implicit Feedback (암묵적)
- 같은 질문 반복
- 재질문
- Retry 버튼 사용

### 5.6 Accuracy
- 성공/실패 플래그
- Task 완료 여부
- GAIA 같은 벤치마크 점수

### 5.7 Automated Evaluation
- LLM 기반 평가 (helpfulness, correctness)
- RAG 특화: RAGAS
- 안전성: LLM Guard

👉 실무에서는 **여러 지표를 조합**해서 판단한다.

---

## 6. Evaluation이란 무엇인가?

**Evaluation(평가)** 는  
Observability로 수집한 데이터를 기반으로  
**에이전트가 잘 동작하는지 판단하고 개선하는 과정**이다.

에이전트는:
- 비결정적(non-deterministic)
- 모델 업데이트, 데이터 변화로 성능이 흔들림

👉 **정기적인 평가 없이는 성능 퇴화를 인지할 수 없다**

---

## 7. Offline Evaluation (오프라인 평가)

- 테스트 데이터셋 기반
- 정답이 이미 존재
- 반복 가능 / 재현 가능

예:
- GAIA validation set
- GSM8K 같은 수학 문제

장점:
- 명확한 정확도 측정
- CI/CD에 적합

단점:
- 실제 사용자 질문과 괴리 가능

👉 **개발 단계의 최소 필수 평가**

---

## 8. Online Evaluation (온라인 평가)

- 실제 사용자 트래픽 기반
- 실시간 모니터링

예:
- 사용자 만족도
- 성공률
- 행동 로그

장점:
- 실제 환경 반영
- 모델 드리프트 감지 가능

단점:
- 정답 라벨 확보 어려움

---

## 9. 실전에서는 둘을 어떻게 쓰는가?

성공적인 팀들은 다음 루프를 사용한다.

1. Offline 평가로 기본 성능 확인
2. 배포
3. Online 지표 모니터링
4. 실패 사례 수집
5. 실패 사례를 Offline 데이터셋에 추가
6. 재학습 / 개선

이 루프가 반복되면서 에이전트는 점점 강해진다.

---

## 10. LangGraph & GAIA 맥락에서의 의미

- LangGraph:
  - Node = Span
  - Graph 실행 = Trace
  - Observability 연동 매우 자연스러움

- GAIA:
  - Offline evaluation의 대표 사례
  - EXACT MATCH 기준 → 자동 평가 최적

👉 **이번 과제는 “에이전트 성능 평가의 출발점”**

---

## 정리

- Observability는 “보는 능력”
- Evaluation은 “판단하고 개선하는 능력”
- 둘 다 없으면 Agent는 실험용 장난감에 머문다
- 둘 다 있으면 **프로덕션 AI 시스템**이 된다


참고자료
Huggingface, agents course, https://huggingface.co/learn