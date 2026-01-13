---
layout: post
title:  "허깅페이스 MCP 코스 - MCP Capability 이해하기: Tools · Resources · Prompts · Sampling"
date:   2025-01-13 00:10:22 +0900
categories: Huggingface_mcp
---

# MCP Capability 이해하기: Tools · Resources · Prompts · Sampling

이 글은 MCP(Model Context Protocol)에서 **Server가 Client에게 제공할 수 있는 기능(Capabilities)**을 체계적으로 정리한다.  
MCP의 핵심은 “AI 모델이 외부 세계와 상호작용하는 방식을 명확히 분리된 원시 개념(primitives)으로 정의한다”는 점이며, 이 네 가지 Capability가 그 기반을 이룬다.

## 1. MCP Capability의 설계 철학

MCP는 단일한 “툴 호출 API”를 제공하지 않는다. 대신,  
**누가 통제하는가**, **어느 방향으로 흐르는가**, **부작용이 있는가**, **사용자 승인이 필요한가**를 기준으로  
외부 기능을 네 가지 범주로 명확히 나눈다.

이 구분의 목적은 다음과 같다.

- LLM의 자율성과 사용자 통제 간 균형 유지
- 위험한 작업과 안전한 작업의 명확한 경계 설정
- 단순 조회부터 복잡한 에이전트 워크플로우까지 포괄

## 2. Tools: 실행 가능한 액션

### 개념
**Tools**는 AI 모델이 MCP를 통해 호출할 수 있는 **실행형 함수(executable function)**다.  
외부 시스템에 영향을 미치거나, 계산·변환·요청 같은 “행동”을 수행한다.

### 핵심 특징
- **통제 주체**: 모델(LLM)
  - LLM이 컨텍스트를 보고 “이 시점에 이 툴을 호출해야 한다”고 판단
- **방향**: Client -> Server
- **부작용**: 있음 (잠재적으로 위험)
- **보안**: 일반적으로 사용자 명시적 승인 필요

### 전형적인 사용 사례
- 메시지 전송
- 티켓 생성
- 외부 API 호출
- 계산 및 데이터 가공

### 의미 정리
Tools는 MCP에서 **“행동(Action)”**에 해당한다.  
그래서 강력하지만, 가장 엄격한 안전 통제가 요구된다.

## 3. Resources: 읽기 전용 컨텍스트

### 개념
**Resources**는 AI 모델이 참고할 수 있는 **읽기 전용 데이터 소스**다.  
복잡한 연산이나 상태 변경 없이, 컨텍스트 제공이 목적이다.

### 핵심 특징
- **통제 주체**: 애플리케이션(Host)
  - 언제 어떤 리소스를 쓸지는 Host가 결정
- **방향**: Client -> Server
- **부작용**: 없음 (read-only)
- **보안**: 상대적으로 낮은 위험

### 전형적인 사용 사례
- 파일 내용 조회
- 데이터베이스 레코드 읽기
- 설정 정보, 문서 접근

### 의미 정리
Resources는 MCP에서 **“정보(Context)”**에 해당한다.  
REST API의 GET 엔드포인트와 유사한 성격을 가진다.

## 4. Prompts: 사전 정의된 워크플로우

### 개념
**Prompts**는 사용자–모델–Server Capability의 상호작용을 구조화하는  
**사전 정의된 템플릿 또는 워크플로우**다.

### 핵심 특징
- **통제 주체**: 사용자
  - 보통 Host UI에서 사용자가 직접 선택
- **방향**: Server -> Client
- **부작용**: 없음
- **역할**: LLM이 “어떤 방식으로 사고하고 출력할지”를 규정

### 전형적인 사용 사례
- 코드 리뷰 템플릿
- 요약 / 분석 / 변환 전용 프롬프트
- 반복되는 업무 흐름의 표준화

### 의미 정리
Prompts는 MCP에서 **“상호작용의 형식(Form)”**을 담당한다.  
즉, 무엇을 할지가 아니라 **어떻게 할지**를 정의한다.

## 5. Sampling: 서버 주도 LLM 호출

### 개념
**Sampling**은 Server가 Client(정확히는 Host)에게  
**LLM 호출을 수행해 달라고 요청**할 수 있게 하는 Capability다.

이는 MCP에서 가장 에이전트적인 기능이다.

### 핵심 특징
- **통제 주체**: Server (하지만 Client/Host가 중재)
- **방향**: Server -> Client -> Server
- **부작용**: 간접적으로 있음
- **보안**: Tools와 마찬가지로 사용자 승인 필요

### Sampling 동작 흐름
1. Server가 `sampling/createMessage` 요청 전송
2. Client/Host가 요청을 검토 및 수정 가능
3. Client가 LLM 샘플링 수행
4. 결과를 다시 검토
5. Server로 결과 반환

### 전형적인 사용 사례
- 다단계 문제 해결
- 에이전트형 워크플로우
- 서버 주도 분석·재검토·반복 처리

### 의미 정리
Sampling은 MCP에서 **“재귀적 사고(Agentic Loop)”**를 가능하게 하는 장치다.  
human-in-the-loop 설계를 통해 통제력을 유지하는 것이 핵심이다.

## 6. Capability 비교 요약


| Capability | 통제 주체 | 흐름 방향 | 부작용 | 승인 필요 | 핵심 역할 |
|-----------|----------|----------|--------|-----------|-----------|
| Tools | 모델(LLM) | Client -> Server | 있음 | 예 | 행동 수행 |
| Resources | 애플리케이션 | Client -> Server | 없음 | 보통 없음 | 정보 제공 |
| Prompts | 사용자 | Server -> Client | 없음 | 없음 | 상호작용 구조 |
| Sampling | Server | Server -> Client → Server | 간접적 | 예 | 에이전트적 사고 |



## 7. Capability는 어떻게 함께 작동하는가

MCP의 강점은 이 네 가지를 **조합**할 수 있다는 점이다.

일반적인 흐름 예시는 다음과 같다.

1. 사용자가 **Prompt**를 선택해 작업 흐름을 정의
2. Prompt가 **Resources**를 통해 필요한 컨텍스트를 포함
3. LLM이 작업 중 **Tools**를 호출해 실제 행동 수행
4. 복잡한 경우, Server가 **Sampling**을 통해 추가 LLM 사고 요청

이 조합 덕분에 MCP는:
- 단순 조회형 앱부터
- 고도화된 자율 에이전트까지
하나의 프로토콜로 포괄한다.

## 8. Capability Discovery (동적 탐색)

MCP는 Client가 Server의 기능을 **사전에 하드코딩하지 않도록** 설계되었다.

Client는 다음 메서드를 통해 Server의 기능을 동적으로 탐색한다.

- `tools/list`
- `resources/list`
- `prompts/list`

이 메커니즘을 통해:
- Server마다 다른 기능 구성을
- 런타임에 유연하게 대응할 수 있다.

## 9. 정리

- MCP Capability는 네 가지 원시 개념으로 명확히 구분된다.
- 이 구분은 “기능”이 아니라 **통제 경계와 책임 분리**를 기준으로 한다.
- Tools, Resources, Prompts, Sampling은 서로 보완적으로 작동한다.
- 이 구조 덕분에 MCP는 강력한 기능을 제공하면서도 안전성과 통제력을 유지한다.

이 Capability 모델이 MCP를 단순한 “툴 연결 API”가 아니라,  
**에이전트 친화적 표준 프로토콜**로 만드는 핵심이다.

참고자료
Huggingface, agents course, https://huggingface.co/learn