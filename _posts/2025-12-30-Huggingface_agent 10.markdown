---
layout: post
title:  "허깅페이스 에이전트 코스 - smolagents 맛보기"
date:   2025-12-30 00:10:22 +0900
categories: Huggingface_agent
---

# smolagents로 첫 번째 Agent 만들기

앞선 글에서는 **Python 코드만으로 Agent를 직접 구현하는 과정이 얼마나 번거로운지** 확인했다.  
이를 해결하기 위해 등장하는 것이 바로 **Agent 라이브러리**이며, 이 글에서는 그중 하나인 **smolagents**를 사용해
**첫 번째 실제 Agent를 만드는 전체 흐름**을 정리한다.

이 섹션의 목적은 “코드를 완벽히 이해하는 것”이 아니라,  
**Agent가 어떻게 구성되고, 어디를 확장하면 되는지 감을 잡는 것**이다.

## 실습 목표

- smolagents가 어떤 문제를 대신 해결해주는지 이해
- CodeAgent 구조 파악
- Tool을 추가해 Agent의 능력을 확장하는 방법 이해
- Hugging Face Space에 Agent를 배포하는 흐름 파악

## smolagents란 무엇인가?

**smolagents**는 Agent 개발을 단순화하기 위한 경량 프레임워크다.

이 라이브러리의 핵심 특징은 다음과 같다.

- Agent의 **Thought–Action–Observation 루프를 내부적으로 관리**
- Stop & Parse, Tool 실행, Observation 주입을 자동 처리
- 사용자는 **행동(behavior)과 Tool 설계에만 집중**하면 됨

특히 smolagents는 **CodeAgent** 중심 설계를 채택한다.

> CodeAgent란  
> - Action을 JSON이 아니라 **코드 블록으로 생성**
> - 코드를 실행해 결과를 Observation으로 사용
> - 다시 Thought로 돌아가는 Agent

즉, 우리가 앞에서 Dummy Agent로 직접 구현했던 구조를  
**프레임워크 차원에서 자동화**해준다.

## 전체 실습 흐름 요약

1. Space 템플릿 복제
2. Hugging Face API 토큰 등록
3. `app.py` 수정
4. Tool 추가
5. Agent 실행 및 테스트
6. Space로 공유

## 1. Space 템플릿 복제

아래 Space를 복제해 시작한다.

https://huggingface.co/spaces/agents-course/First_agent_template


복제란 원본 Space를 **내 Hugging Face 계정 아래에 그대로 복사**하는 것


## 2. Hugging Face 토큰 등록

Agent가 LLM API에 접근하려면 토큰이 필요하다.

절차 요약:

1. https://hf.co/settings/tokens 에서 토큰 생성 (inference 권한)
2. 복제한 Space의 **Settings** 탭 이동
3. **Variables and Secrets** -> New Secret
4. 이름: `HF_TOKEN`
5. 값: 발급받은 토큰
6. 저장

이 토큰은 코드에 직접 쓰이지 않고,  
환경 변수로 안전하게 주입된다.

## 3. app.py의 역할

이 실습에서 **수정해야 할 유일한 파일은 `app.py`**다.
`app.py`는 다음을 담당한다.

- Tool 정의
- LLM 설정
- Agent 생성
- UI 실행

## 4. 기본 import 구조

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
```

여기서 중요한 것은:

- CodeAgent: Agent 본체
- tool: Python 함수를 Tool로 등록하기 위한 데코레이터
- InferenceClientModel: Serverless API 기반 LLM 래퍼

## 5. Tool 정의 방식

Tool은 Agent가 할 수 있는 행동의 집합이다.

### Tool 정의 예시 1: 더미 Tool

```python
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """
    A tool that does nothing yet
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"
```

Tool 정의 시 반드시 지켜야 할 규칙

1. 입력 타입 명시
2. 반환 타입 명시
3. docstring에 인자 설명 포함

이 정보는 smolagents가
Tool 스펙을 자동으로 System Prompt에 삽입하는 데 사용된다.

### Tool 정의 예시 2: 실제 동작하는 Tool

```python
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    tz = pytz.timezone(timezone)
    local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return f"The current local time in {timezone} is: {local_time}"
```

이 Tool은:

- 외부 API 없이
- 로컬 Python 라이브러리만으로 동작
- 어떤 환경에서도 테스트 가능

## 6. LLM 설정

```python
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
)
```

이 부분은:

- Serverless API를 통해
- 대형 instruct 모델을 호출
- chat template, token 처리 등을 내부에서 자동 처리

## 7. System Prompt 관리 방식

System Prompt는 코드에 하드코딩되지 않고
prompts.yaml 파일에 분리되어 있다.

장점:

- Prompt 수정 시 코드 변경 불필요
- 여러 Agent에서 재사용 가능
- 역할, 규칙, 포맷 관리가 쉬움
- Agent 생성 시 다음처럼 전달된다.

```python
with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)
```

## 8. CodeAgent 생성

```python
agent = CodeAgent(
    model=model,
    tools=[final_answer],
    max_steps=6,
    verbosity_level=1,
    prompt_templates=prompt_templates
)
```

여기서 핵심은 tools다.

tools 리스트에 무엇을 넣느냐가
곧 “Agent가 무엇을 할 수 있는가”를 결정한다.

- 기본적으로 FinalAnswerTool은 반드시 포함
- 여기에:
    - 검색 Tool
    - 이미지 생성 Tool
    - 사용자 정의 Tool
- 를 계속 추가 가능

## 9. UI 실행

```python
GradioUI(agent).launch()
```

- Gradio 기반 UI 자동 생성
- 웹에서 바로 Agent와 대화 가능
- Space에 배포되면 누구나 접근 가능

## 이 실습의 핵심 포인트

1. Agent의 복잡한 내부는 모두 숨겨져 있다

- Stop & Parse
- Action 실행
- Observation 주입
- 루프 제어

smolagents가 전부 처리

2. 사용자는 두 가지만 신경 쓰면 된다

- Tool을 어떻게 설계할 것인가
- System Prompt를 어떻게 쓸 것인가

3. Agent는 “Tool이 곧 능력”이다

- Tool을 추가하면 -> Agent 능력 증가
- Tool이 없으면 -> 아무 것도 못 함

## 정리

- smolagents는 Agent 구현의 번거로움을 제거해준다
- CodeAgent는 코드 기반 Action을 중심으로 동작한다
- Tool 정의가 Agent 설계의 핵심이다
- System Prompt는 YAML로 분리 관리한다
- Hugging Face Space를 통해 Agent를 바로 배포할 수 있다

참고자료
Huggingface, agents course, https://huggingface.co/learn