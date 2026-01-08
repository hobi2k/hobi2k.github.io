---
layout: post
title:  "허깅페이스 에이전트 코스 - Why use smolagents"
date:   2025-01-07 00:10:22 +0900
categories: Huggingface_agent
---

# Building Agents That Use Code

이 글에서는 `smolagents`의 기본 Agent 타입인 **CodeAgent**를 “개념 → 내부 동작 → 실전 사용 → 배포/관측” 순서로 매우 구체적으로 정리한다.  
핵심은 단순히 “코드를 생성한다”가 아니라, **왜 코드 기반 Action이 성능·안정성·표현력 측면에서 유리한지**, 그리고 **ReAct(Thought–Action–Observation) 루프가 실제 라이브러리 구현에서 어떻게 굴러가는지**를 정확히 이해하는 것이다.

---

## 0. 전제: CodeAgent는 왜 중요한가

`smolagents`에서 CodeAgent는 사실상 “표준”이다.  
일반적인 Agent 프레임워크들이 JSON 기반 Tool 호출에 집중한 것과 달리, `smolagents`는 기본 철학이 **Action을 코드로 표현하고 실행한다**에 맞춰져 있다.

- JSON Action: “어떤 Tool을 어떤 인자로 호출할지”를 문자열 포맷으로 기술
- Code Action: “실제로 실행될 Python 코드”를 생성

이 차이는 단순한 취향이 아니라, **에이전트의 에러율, 표현력, 멀티스텝 효율**에 직접적으로 영향을 준다.

---

## 1. CodeAgent란 무엇인가

CodeAgent는 다음을 수행하는 Agent다.

1) LLM이 **다음 Action을 Python 코드 스니펫으로 작성**한다.  
2) 실행기(executor)가 해당 코드를 **샌드박스 환경에서 실행**한다.  
3) 실행 결과를 **Observation으로 기록**하고, 그 Observation을 기반으로 다음 Thought/Action을 이어간다.  

즉, CodeAgent는 “LLM이 코드 생성 → 시스템이 실행 → 결과를 다시 LLM에 반영”하는 구조를 갖는다.

이 구조는 다음 특성을 가진다.

- **효율(Efficient)**: 한 번의 코드 블록에서 여러 Tool 호출과 후처리를 묶을 수 있다.
- **표현력(Expressive)**: 조건/반복/자료구조 조작이 자연스럽다.
- **정확성(Accurate)**: JSON 파싱/스키마 불일치/문자열 인자 오류를 크게 줄인다.

---

## 2. 왜 Code Agents인가 (JSON 대비 기술적 이유)

멀티스텝 Agent에서 LLM은 “생각 → 행동(툴 호출) → 결과 반영”을 반복한다.  
전통적인 방식은 보통 다음 흐름이다.

### 2.1 JSON 기반 Tool Calling의 흐름(전통적)
1. LLM 출력:  
   - `{ "tool": "web_search", "args": {"q": "..."} }`
2. 시스템이 JSON을 **파싱**  
3. tool registry에서 `"web_search"` 찾기  
4. 인자 검증/변환(타입 캐스팅, 누락 체크)  
5. 실제 함수 호출  
6. 결과를 Observation으로 추가

여기서 문제는 **(a) JSON이 깨지거나**, **(b) tool명이 틀리거나**, **(c) args 구조가 스키마와 불일치하거나**, **(d) 문자열로 표현된 값이 타입 문제를 일으키는** 케이스가 빈번하다는 점이다.

### 2.2 코드 기반 Action의 핵심 이점

#### (1) Composability(조합성)
코드에서는 여러 Tool 호출을 “하나의 프로그램 조각”으로 조합할 수 있다.

- 검색 결과를 받아
- 상위 3개만 추리고
- URL을 방문해 본문을 읽고
- 요약한 뒤
- 최종 답변을 구성

이 모든 과정을 **하나의 코드 블록**에서 수행할 수 있다.

JSON 기반에서는 보통 Tool 호출이 “한 번에 하나”로 강제되며, 후처리는 다음 스텝으로 넘겨야 해서 스텝 수가 늘어난다.

#### (2) Object Management(객체 직접 처리)
이미지/테이블/딕셔너리 같은 복합 객체는 JSON으로 표현하면 결국 문자열 인코딩/디코딩 문제가 발생한다.  
반면 코드에서는 객체를 직접 변수로 들고 다닌다.

- `img = image_generation(prompt=...)`
- `metadata = extract(img)`
- `scores = rank(metadata)`

#### (3) Generality(표현 가능한 작업의 범위)
코드는 계산적으로 가능한 모든 로직을 표현할 수 있다.

- 분기(조건)
- 반복(while/for)
- 예외 처리(try/except)
- 자료구조 처리
- 임시 캐싱, 중간 결과 저장

#### (4) LLM에게 자연스러운 출력 형태
LLM은 코드 데이터를 매우 많이 학습했다.  
특히 “도구를 코드로 호출하는 패턴”은 학습 데이터에서 매우 흔하다.

따라서 동일한 목표라도,
- JSON 스키마를 엄격히 지키며 문자열을 맞추는 것보다
- 익숙한 Python 코드 호출 형태를 출력하는 것이
성공률이 높다는 보고가 있다(관련 도식이 바로 “Code vs JSON Actions”).

---

## 3. CodeAgent는 내부적으로 어떻게 동작하나

CodeAgent는 `smolagents`의 추상 클래스인 `MultiStepAgent`의 구현체다.  
즉, CodeAgent는 **멀티스텝 루프**를 전제로 설계되어 있다.

아래는 실행 흐름을 “프레임워크 관점”에서 그대로 풀어쓴 것이다.

### 3.1 Step 0: 로그/메모리 구조 생성
- 시스템 프롬프트는 `SystemPromptStep`에 기록된다.
- 사용자 요청은 `TaskStep`에 기록된다.

이 단계에서 Agent는 “무엇을 할지”가 아니라, **앞으로 루프를 돌기 위한 상태(memory/log)**를 초기화한다.

### 3.2 while-loop 실행(멀티스텝 루프)
각 루프에서 다음이 반복된다.

#### (1) `agent.write_memory_to_messages()`
Agent 내부 로그(시스템 프롬프트, 이전 Thought/Action/Observation)를  
LLM이 읽을 수 있는 **chat messages** 형태로 변환한다.

- System: 행동 규칙/툴 목록/출력 포맷
- User: 요청
- Assistant: 이전 스텝의 코드와 관측 결과

이 과정을 통해 LLM은 “지금까지 무슨 일이 있었는지”를 매번 전체 컨텍스트로 다시 읽는다.

#### (2) Model 호출
`Model`(예: `InferenceClientModel`)에 messages를 보내 completion을 생성한다.

#### (3) Completion 파싱
CodeAgent에서는 completion에서 **실행 가능한 Python 코드 조각**을 추출해야 한다.  
여기서 “코드 블록”이 명확하지 않으면 실행이 실패할 수 있으므로, CodeAgent는 특정 포맷을 강하게 유도한다(프롬프트 템플릿이 중요).

#### (4) 코드 실행
추출한 코드를 샌드박스 환경에서 실행한다.

- 허용된 import만 가능
- Tool 호출은 wrapper로 제공
- 출력/반환/예외를 수집

#### (5) 결과를 Observation으로 기록
실행 결과(출력/반환값/에러)를 **ActionStep**에 기록하고, 다음 루프에서 다시 LLM에게 제공한다.

이 구조가 곧 ReAct의 구현이다.

---

## 4. 실습 예제 1: 웹 검색으로 파티 플레이리스트 선정

### 4.1 설치

```bash
pip install smolagents -U
```

`-U`는 최신 버전으로 업그레이드 설치한다는 뜻이다.

### 4.2 Hugging Face Hub 로그인

```python
from huggingface_hub import login

login()
```

Serverless Inference API 등을 쓰려면 토큰이 필요하다.
login()은 토큰을 로컬 환경에 저장해 이후 호출에서 자동 사용한다.

### 4.3 DuckDuckGo 검색 Tool을 가진 CodeAgent

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=InferenceClientModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
```

코드 해설

- DuckDuckGoSearchTool()
  - 웹 검색을 수행하는 Tool이다.
  - CodeAgent가 코드로 web_search(...) 같은 형태로 호출할 수 있도록 제공된다.

- InferenceClientModel()
  - Hugging Face Serverless Inference를 통해 LLM 호출을 수행한다.
  - 기본 모델이 정해져 있지만, 옵션으로 변경 가능하다.

- agent.run(...)
  - 내부적으로 멀티스텝 while-loop를 돌며,
  - 필요한 검색을 수행하고,
  - 결과를 조합해 최종 답을 낸다.

실행 중 다음과 같은 로그가 뜰 수 있다.

```python
 ─ Executing parsed code: ──────────────────────────────────────────────────────────────────────────────────────── 
  results = web_search(query="best music for a Batman party")                                                      
  print(results)                                                                                                   
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
```

이 로그가 의미하는 것

- LLM이 실제로 생성한 코드 조각이 “파싱된 뒤 실행”되었다는 뜻이다.
- web_search(...)는 DuckDuckGoSearchTool의 wrapper 함수로 연결된다.
- 이 결과가 Observation으로 기록되고, 다음 스텝에서 LLM이 이를 읽고 최종 플레이리스트를 구성한다.

## 5. 실습 예제 2: 커스텀 Tool로 메뉴 준비하기

이번에는 “이미 있는 Tool”이 아니라 “내가 정의한 함수”를 Tool로 등록한다.

### 5.1 @tool로 Tool 만들기

```python
from smolagents import CodeAgent, tool, InferenceClientModel

# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."
```

핵심 포인트

- 타입 힌트: occasion: str -> str
  - LLM에게 인자/반환 타입 힌트를 준다.
- docstring에 Args 설명
  - smolagents는 docstring에서 “이 Tool이 뭘 하는지/인자가 뭔지”를 읽어 시스템 프롬프트에 반영한다.
- Allowed values를 명시
  - LLM이 존재하지 않는 값을 상상해 넣는(환각) 가능성을 줄인다.

### 5.2 Tool을 tools 리스트에 넣고 실행

```python
# Alfred, the butler, preparing the menu for the party
agent = CodeAgent(tools=[suggest_menu], model=InferenceClientModel())

# Preparing the menu for the party
agent.run("Prepare a formal menu for the party.")
```

이제 Agent는 내부적으로 다음과 같은 코드를 생성할 가능성이 높다.

- menu = suggest_menu("formal")
- print(menu) 혹은 최종 답변에 반영

Allowed values를 정확히 줬기 때문에 "formal"을 선택할 확률이 크게 오른다.

## 6. 실습 예제 3: Agent 코드에서 import 쓰기(보안 모델)

CodeAgent는 “코드를 실행”하므로 보안이 핵심이다.
따라서 기본적으로 import는 제한된다.

### 6.1 추가 import 허용

```python
from smolagents import CodeAgent, InferenceClientModel
import numpy as np
import time
import datetime

agent = CodeAgent(tools=[], model=InferenceClientModel(), additional_authorized_imports=['datetime'])

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)
```

해설

- additional_authorized_imports=['datetime']
  - 샌드박스에서 import datetime을 허용한다.
- 반면 numpy, time은 이 예제에서는 “파이썬 파일 레벨 import”로 보이지만,
  - 실제 실행 샌드박스에서 해당 import가 허용되는지 여부는 정책에 따라 다르다.
  - 중요한 건 “Agent가 생성하는 코드 내 import”가 통제된다는 점이다.

Agent는 tasks의 분 단위를 모두 합산해 datetime.now()에 더한 뒤 준비 완료 시간을 계산한다.

## 7. Agent를 Hub에 공유하기

### 7.1 업로드

```python
# Change to your username and repo name
agent.push_to_hub('sergiopaniego/AlfredAgent')
```

의미

- Agent 구성(프롬프트/툴/설정)을 Hub 리포지토리로 올린다.
- 다른 사람은 같은 Agent를 재현할 수 있다.

### 7.2 다운로드 및 실행

```python
# Change to your username and repo name
alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent', trust_remote_code=True)

alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")  
```

trust_remote_code=True 주의
- 원격 리포지토리의 코드를 신뢰하고 실행한다는 뜻이다.
- 내부적으로 Tool/클래스 구현이 포함될 수 있으므로, 프로덕션에서는 검증이 필요하다.

## 8. “완성형” 파티 준비 Agent 예시(여러 Tool 결합)

아래 코드는 다양한 Tool을 결합해 “실제 활용 가능한” 수준의 Agent를 만든다.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, Tool, tool, VisitWebpageTool

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.
    
    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    
    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)
    
    return best_service

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""
    
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }
    
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        
        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")

# Alfred, the butler, preparing the menu for the party
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(), 
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool(),
        FinalAnswerTool()
    ], 
    model=InferenceClientModel(),
    max_steps=10,
    verbosity_level=2
)

agent.run("Give me the best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")
```

이 코드의 구조적 의미

- DuckDuckGoSearchTool()
  - “외부 정보 수집”을 담당
- VisitWebpageTool()
  - 검색 결과 URL을 실제로 열어 본문을 추출하는 역할(웹페이지 관측 강화)
- suggest_menu / catering_service_tool
  - 도메인 로직(메뉴/케이터링)을 Tool로 캡슐화
- SuperheroPartyThemeTool
  - Tool 클래스를 상속해 “스키마 기반 Tool”을 정의(입력 정의/설명/출력 타입 포함)
- FinalAnswerTool()
  - 최종 답변을 “종료 신호”와 함께 반환하도록 설계된 Tool (프레임워크가 completion 종료를 안정적으로 처리하게 도움)

## 9. 실행 추적: OpenTelemetry + Langfuse로 관측 가능성 확보

멀티스텝 Agent는 디버깅이 어렵다.
따라서 “실행 추적(Tracing)”이 매우 중요하다.

### 9.1 설치

```python
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents langfuse
```

### 9.2 Langfuse 환경 변수 설정

```python
import os

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..." 
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
```

### 9.3 Langfuse 클라이언트 인증 확인

```python
from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
```

### 9.4 smolagents Instrumentation 활성화

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

SmolagentsInstrumentor().instrument()
```

이후부터는 smolagents 실행이 자동으로 추적되어,
각 Step의 Thought/Action/Observation 흐름을 외부에서 재현 가능해진다.

### 9.5 Hub Agent 실행 예시(트레이싱 대상)

```python
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(tools=[], model=InferenceClientModel())
alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent', trust_remote_code=True)
alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")  
```

## 결론

- CodeAgent는 “코드 생성”이 아니라 ReAct 루프의 실행형 구현이다.
- JSON 대비 코드 기반 Action은 조합성/객체처리/표현력/LLM 친화성에서 유리하다.
- tools=[...]는 단순 리스트가 아니라, LLM이 호출 가능한 실행 환경을 구성하는 것이다.
- 보안(샌드박스, import 허용 목록)은 “코드 실행 Agent”의 필수 설계 요소다.
- 프로덕션에서는 관측 가능성(Tracing)이 곧 안정성이다.

참고자료
Huggingface, agents course, https://huggingface.co/learn