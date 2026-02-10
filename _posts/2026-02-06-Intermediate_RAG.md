---
layout: post
title:  "입찰/공고 분석 AI - 협업일지 2026-02-06"
date:   2026-02-06 00:10:22 +0900
categories: Intermediate_RAG
---

# 2월 6일 협업일지

날짜: 2026-02-06  
이름: 안호성  
팀명: 3팀

## 오늘 맡은 역할 및 작업 내용

오늘은 **RAG 실행 그래프 구조를 비선형으로 개편**하고,  
**세션 기반 기억(메모리) + 피드백 저장**까지 확장한 날이었다.  
또한 **Gradio 자동 음성 재생** 문제를 해결하기 위해  
브라우저 정책과 DOM 교체 타이밍까지 고려한 로직을 넣었다.

1. LangGraph 구조 개편 (선형 -> 비선형)

- 기존 상태
    - LangGraph는 사실상 **선형** 흐름
    - `analyze -> retrieve -> generate` 한 경로만 존재
    - 질문 유형(single/multi/followup)은  
      내부에서 처리되지만 노드 분기는 없었다
- 오늘 변경
    - **질문 유형별 분기 노드 도입**
    - 라우팅 기준을 명시적으로 분리
    - 분기 결과에 따라  
      **single, multi, single_followup, multi_followup**  
      네 가지 경로로 나눔
- 실제 분기 의도
    - single: 단일 문서/일반 질의
    - multi: 다문서 비교/목록형 질의
    - single_followup: 단일 문서의 후속 질문
    - multi_followup: 다문서 맥락의 후속 질문
- 구현 방향
    - `analyze` 단계에서 `question_type` 결정
    - `route` 노드가 `single | multi | followup`로 분기
    - followup은 세션 메모리에서 문맥 복구
    - followup 중에서도
        - 직전 질문이 multi이면 `multi_followup`
        - 직전 질문이 single이면 `single_followup`
    - retrieve 단계는 분기별로 전략을 다르게 사용

2. 분기별 검색 전략 정리 (single / multi / followup)

- single
    - 기본 similarity 검색
    - 필요 시 MMR 또는 rerank 적용
    - 문서 범위를 좁혀 속도와 정확성 확보
- multi
    - RRF 기반 하이브리드 검색
    - 입력 리스트: similarity + MMR + BM25
    - 다문서 비교를 위해 컨텍스트 폭을 넓힘
- single_followup
    - 직전 세션 문서 범위를 우선 재사용
    - doc_id 기반 필터 적용
    - 단일 문서 내 연속 질문을 가정
- multi_followup
    - 직전 다문서 세션 범위를 재사용
    - doc_id 리스트를 유지해 문서 풀 고정
    - 다문서 비교 맥락에서  
      followup 질문을 바로 처리 가능

3. SQLite 기반 기억 모델 추가

- 목표
    - "1턴 이후 맥락 소실" 문제를 해결
    - 프로세스 재시작에도 세션 기록 유지
- 핵심 저장 항목
    - session_id
    - question
    - answer
    - doc_ids (검색 범위 유지용)
    - question_type (single/multi/followup 분기 판단용)
- 저장 흐름
    1) 질문 처리 후 `answer` 생성
    2) 해당 세션에 대해  
       question/answer/doc_ids/question_type 저장
    3) 다음 질문에서:
        - last_question / last_answer / last_question_type 조회
        - followup 여부 판단
        - doc_ids를 가져와 문서 범위 복구

4. 피드백 저장 기능 추가 (좋아요/싫어요)

- 목표
    - 사용자 반응을 저장해  
      품질 개선 루프의 기반 데이터 확보
- 저장 항목
    - session_id
    - question
    - answer
    - rating (1: 👍, -1: 👎, NULL: 미평가)
    - created_at (timestamp)
- 저장 흐름
    - UI에서 “좋아요/싫어요” 버튼 클릭
    - 직전 질문/답변을 추출
    - SQLite 테이블에 rating 기록
- 중요 포인트
    - 피드백은 **“답변 단위”**로 매칭
    - 동일 세션 내 다른 질문과 혼동되지 않도록  
      직전 turn 기준으로 저장

5. 맥락 변경(Reset) 규칙 정리

- 문제
    - followup 판단이 과도하면  
      엉뚱한 문서 범위를 재사용
- 대응 규칙
    - 사용자가 **명시적 리셋 키워드** 입력 시  
      세션 doc_ids를 초기화
    - 새 질문에서 파일명/기관명/사업명 등이  
      명확하면 "새 문맥"으로 전환
    - followup은 **세션 내 연속성**이 확인될 때만 적용
- 효과
    - 이전 문서 맥락을 고집하는 오류 감소
    - 다른 문서로 전환할 때 자연스럽게 reset

6. Gradio 음성 자동 재생 로직 상세

- 문제
    - 첫 턴 이후부터 자동 재생 실패
    - 브라우저 자동재생 정책 +  
      Gradio가 audio DOM을 매 턴 교체하는 타이밍 문제
- 해결 접근
    1) 사용자의 첫 클릭/키다운을  
       “autoplay 허용 플래그”로 등록
    2) `loadeddata / canplay / loadedmetadata / durationchange`
       이벤트에서 `audio.play()` 재시도
    3) `MutationObserver`로 `src` 변경 감지 후  
       즉시 재생 재시도
    4) `setInterval`로 주기적 재시도  
       (DOM 교체 타이밍 흡수)
- 구현 핵심
    - `send` 클릭 시 강제로  
      userInteracted 플래그 활성화
    - `audio.paused && audio.src` 상태를  
      반복 체크하면서 재생 시도
    - DOM 교체 시점이 늦어도  
      최대 수 회 재시도하여 자동 재생 성공률 상승

## 오늘 작업 현황

- LangGraph 분기 구조를 실제로 사용 가능하게 변경
- single/multi + followup을 분기하여  
  retrieval 전략과 문서 범위를 명확히 분리
- SQLite 세션 기억 + 피드백 저장 기능 도입
- Gradio 자동 재생 문제를  
  사용자 제스처 + DOM 감지 기반으로 해결

## 오늘 협업 중 제안하거나 피드백한 내용

- followup 판단을 “세션 저장 + 질문 유형” 기반으로 강화
- 피드백은 UI에서 즉시 저장하고  
  DB 조회로 확인 가능하도록 설계
- Gradio는 DOM 교체가 잦아  
  재생 이벤트 트리거를 반복해야 안정적이라는 점 공유

## 오늘 분석/실험 중 얻은 인사이트나 발견한 문제점

- RAG 구조에서 **랭그래프가 의미 있게 사용되려면 분기 구조가 있어야 함**
- 단순히 분기만 추가하면 의미가 없고,  
  각 분기마다 전략과 doc scope가 달라야 함
- 자동 재생은 서버 문제가 아니라  
  **브라우저 정책/DOM 교체**가 핵심 원인

## 일정 지연이나 협업 중 어려웠던 점

- 분기 구조 도입 후  
  followup 로직이 더 복잡해짐
- 피드백 저장은 단순하지만  
  직전 질문/답변을 정확히 추출하는 로직이 필요했음
- Gradio의 DOM 교체 특성 때문에  
  단일 이벤트 기반 재생이 실패

## 오늘 발표 준비나 커뮤니케이션에서 기여한 부분

- LangGraph 분기 구조 변경안을 문서로 정리
- SQLite 메모리 + 피드백 설계를  
  팀원에게 공유
- Gradio 자동 재생 문제 해결 로직을  
  팀 내 실험 공유

## 내일 목표 / 할 일

- 베이스라인 결정 후 TTS 연동
- SQLite 기반 데이터베이스 연동