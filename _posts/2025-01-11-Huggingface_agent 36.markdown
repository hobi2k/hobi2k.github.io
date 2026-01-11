---
layout: post
title:  "허깅페이스 에이전트 코스 - 실습 2"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# Hands-On: GAIA 리더보드 제출 가이드

이제 최종 에이전트를 직접 만들어 **GAIA 기반 평가에 제출**하는 단계다.  
이 글은 “어떻게 점수를 매기는가?”가 아니라  
**“실제 에이전트를 어떻게 검증하고 제출하는가”** 에 초점을 둔다.

## 1. 평가용 데이터셋 개요

이번 실습에서 사용하는 데이터셋은 다음과 같다.

- **출처**: GAIA 벤치마크
- **구성**:  
  - GAIA **validation set – Level 1** 문제 중
  - 총 **20문항** 선별
- **선별 기준**:
  - 필요한 도구 수
  - 필요한 단계 수 (과도하게 복잡하지 않은 문제)

### 목표 점수 기준

- Level 1 문제 기준
- **30% 정답률**을 현실적인 목표로 설정

> 이는 “완벽한 에이전트”가 아니라  
> **Agentic 시스템을 실제로 작동시키는 경험**에 초점을 둔 기준이다.

## 2. 제출 프로세스 개요

가장 중요한 질문은 이것이다.

> “내 에이전트를 어떻게 채점 시스템에 연결하지?”

이를 위해 **전용 API**가 제공된다.

**API 문서**  
https://agents-course-unit4-scoring.hf.space/docs

## 3. 제공되는 API 엔드포인트

GAIA 과제 제출은 아래 4개의 엔드포인트를 중심으로 이루어진다.

### `GET /questions`
- 필터링된 전체 평가 질문 목록 반환

### `GET /random-question`
- 질문 하나를 무작위로 반환
- 디버깅 및 테스트용으로 유용

### `GET /files/{task_id}`
- 특정 질문(task)에 연결된 파일 다운로드
- 이미지, 문서 등 멀티모달 입력 처리 시 필요

### `POST /submit`
- 에이전트의 답변 제출
- 자동 채점
- 리더보드 업데이트

## 4. 채점 방식 주의사항 (중요)

GAIA 채점은 다음 조건을 따른다.

- **EXACT MATCH**
  - 공백, 순서, 복수형까지 정확히 일치해야 정답
- 불필요한 설명 금지
- `"FINAL ANSWER"` 같은 접두어 포함 금지

**에이전트 출력은 반드시 “정답만”** 반환해야 한다.

GAIA 팀이 공유한 **프롬프트 예시**를 참고하는 것이 강력히 권장된다.  
(단, 이 과제에서는 `FINAL ANSWER` 문구는 제외)

## 5. 제출용 템플릿 사용 안내

과제 이해를 돕기 위해 **기본 템플릿 Space**가 제공된다.

템플릿  
https://huggingface.co/spaces/agents-course/Final_Assignment_Template

### 템플릿 사용 원칙

- 그대로 써도 됨
- 마음껏 수정 가능
- 구조를 완전히 바꿔도 됨
- 새로운 도구·에이전트 구조 추가 가능

---

## 6. API 제출에 필요한 3가지 정보

`POST /submit` 호출 시 반드시 아래 3가지를 제공해야 한다.

### Username
- Hugging Face 사용자 이름
- Gradio 로그인으로 자동 획득 가능
- 리더보드 식별용

### Code Link (`agent_code`)
- 본인의 Hugging Face Space 코드 링크
- 형식 예시:
  ```
  https://huggingface.co/spaces/username/space_name/tree/main
  ```
- **Space는 반드시 public**

### Answers (`answers`)
- 에이전트가 생성한 답변 목록
- 형식:
  ```json
  [
    {
      "task_id": "...",
      "submitted_answer": "..."
    }
  ]
  ```

## 7. 제출 절차 요약

1. 템플릿 Space 복제
2. 에이전트 구현 (LangGraph / Agentic RAG 권장)
3. API로 질문 수집
4. 에이전트로 답변 생성
5. `/submit`으로 결과 제출
6. 리더보드 확인

**학생 리더보드**  
https://huggingface.co/spaces/agents-course/Students_leaderboard

## 8. 리더보드 운영 원칙 (중요)

- 이 리더보드는 **학습 목적**
- 점수는 재미를 위한 참고 지표

## 핵심 요약

- 이 과제는 “점수 경쟁”이 아니라
- **Agentic 시스템을 실제 평가 파이프라인에 연결하는 경험**이다
- GAIA Level 1에서 30%만 넘어도 충분히 의미 있다
- 정확한 출력 제어가 성능을 좌우한다


참고자료
Huggingface, agents course, https://huggingface.co/learn