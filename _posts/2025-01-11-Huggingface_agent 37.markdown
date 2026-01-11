---
layout: post
title:  "허깅페이스 에이전트 코스 - 실습 3"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# GAIA 제출용 Agent Runner (LangGraph 버전)

이 글에서는 **GAIA Level 1 평가용 에이전트**를 Hugging Face Space에서 실행하고,  
질문을 받아 **자동으로 답변 -> 제출 -> 점수 확인**까지 수행하는 **제출 파이프라인 템플릿**을 정리한다.

기본 제공 코드는 `BasicAgent`라는 **더미 에이전트**만 포함하고 있으므로,  
과제에서는 이 부분을 **LangGraph 기반 Agent**로 교체해야 한다.

아래에서는 다음을 정리한다.

1. 전체 코드의 역할과 흐름
2. 과제에서 반드시 수정해야 하는 지점
3. **LangGraph 기반 Agent 구현 코드 (추가해야 할 코드)**

## 1. 전체 구조 개요

이 Space는 크게 3부분으로 구성된다.

### Agent 정의 영역
```python
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer
```

- 현재는 항상 `"This is a default answer."`만 반환
- **과제에서 반드시 구축해야 하는 핵심 영역**

### 평가 및 제출 파이프라인 (`run_and_submit_all`)

```python
def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
```

이 함수는 다음을 순서대로 수행한다.

1. Hugging Face 로그인 확인
2. GAIA 서버에서 질문 목록 가져오기 (`GET /questions`)
3. 모든 질문에 대해 Agent 실행
4. 결과를 `{task_id, submitted_answer}` 형식으로 수집
5. `POST /submit`으로 일괄 제출
6. 점수 및 결과를 UI에 표시

**이 함수는 수정할 필요 없음**  
Agent만 제대로 만들면 그대로 사용 가능

### Gradio UI

```python
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
```

- Hugging Face 로그인 버튼
- “Run Evaluation & Submit All Answers” 버튼
- 실행 상태 + 결과 테이블 출력

UI 역시 **수정 불필요**

## 2. 과제에서 반드시 해야 할 일

### 해야 할 것

- `BasicAgent`를 **LangGraph 기반 Agent**로 교체
- Agent가 질문 문자열을 받아 **정답 문자열만** 반환하도록 구현
- 불필요한 설명, 접두어, 포맷 제거 (EXACT MATCH 필수)

### 하지 말아야 할 것

- `"FINAL ANSWER"` 같은 문구 출력
- JSON, 마크다운, 설명 포함
- 여러 줄 출력

## 3. LangGraph 기반 Agent 구현

아래 코드는 **과제용으로 바로 사용할 수 있는 LangGraph Agent 최소 구현 예시**다.  
(※ Level 1 기준 / 외부 검색·도구 없이도 구조적으로 합격점)

### 3.1 LangGraph Agent 정의 코드

```python
# ===== LangGraph 기반 Agent 구현 =====

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 1. State 정의
class AgentState(TypedDict):
    question: str
    answer: str

# 2. LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Space 환경에 맞게 조정 가능
    temperature=0
)

# 3. Node 정의
def solve_question(state: AgentState) -> dict:
    """
    GAIA 질문을 받아 정답만 생성하는 노드
    """
    question = state["question"]

    prompt = f"""
You are solving a GAIA benchmark question.

Rules:
- Answer with ONLY the final answer
- No explanation
- No formatting
- No extra text

Question:
{question}
""".strip()

    response = llm.invoke([HumanMessage(content=prompt)])

    # EXACT MATCH를 위해 strip
    return {
        "answer": response.content.strip()
    }

# 4. Graph 구성
builder = StateGraph(AgentState)
builder.add_node("solve", solve_question)

builder.add_edge(START, "solve")
builder.add_edge("solve", END)

langgraph_agent = builder.compile()
```

---

### 3.2 BasicAgent를 LangGraph Agent로 교체

기존 코드의 `BasicAgent`를 **아래처럼 교체**한다.

```python
# 기존 BasicAgent 대체

class BasicAgent:
    def __init__(self):
        print("LangGraphAgent initialized.")

    def __call__(self, question: str) -> str:
        """
        question: GAIA 질문 (string)
        return: 정답만 포함된 string
        """
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        result = langgraph_agent.invoke({
            "question": question,
            "answer": ""
        })

        final_answer = result["answer"]
        print(f"Agent returning answer: {final_answer}")

        return final_answer
```

## 4. 이 구조가 과제에 적합한 이유

- LangGraph 사용 (과제 요구 충족)
- State / Node / Flow 명확
- EXACT MATCH 제약을 프롬프트에서 강제
- 추후 확장 용이
  - Web search
  - File download (`/files/{task_id}`)
  - ToolNode 추가
  - 멀티스텝 reasoning


## 5. 최종 코드

```python
# LangGraph 기반 GAIA Level 1 Agent 구현

"""
목적
- GAIA Benchmark Level 1 문제를 풀기 위한 최소 Agent 구조
- LangGraph를 사용하여 "에이전트의 사고 흐름"을 그래프로 명시
- Hugging Face Space + 자동 제출 파이프라인과 결합 가능

전제
- GAIA는 EXACT MATCH 평가이므로 출력 형식이 매우 중요
- Agent는 반드시 "정답 문자열만" 반환해야 함
- 설명, 포맷, 접두어, 줄바꿈은 전부 감점 요인
"""

# 1. 필수 라이브러리 import
from typing import TypedDict

# LangGraph의 핵심 구성 요소
from langgraph.graph import StateGraph, START, END

# LangChain 메시지 타입 (LLM 호출 시 사용)
from langchain_core.messages import HumanMessage

# 실제 LLM 호출용 (OpenAI 계열 예시)
from langchain_openai import ChatOpenAI


# 2. AgentState 정의
"""
LangGraph의 핵심 개념: State

- LangGraph는 "함수 호출 체인"이 아니라
  "상태(State)가 노드를 거치며 변화하는 그래프"임
- 모든 노드는 "State -> State 일부 수정" 구조를 가짐

즉,
    입력 State  ->  Node  ->  출력 State
"""

class AgentState(TypedDict):
    """
    Agent가 문제를 푸는 동안 유지하는 최소 상태

    - GAIA Level 1은 복잡한 계획/도구 없이도 해결 가능

    필드 설명
    question : str
        GAIA 서버에서 받은 원본 질문 문자열

    answer : str
        Agent가 계산/추론을 거쳐 최종적으로 도출한 정답
        (EXACT MATCH 대상)
    """
    question: str
    answer: str


# 3. LLM 초기화
"""
LLM은 LangGraph 바깥에서 한 번만 생성하는 것이 좋다.

이유:
- 매 Node 호출마다 LLM을 새로 만들면 비용/지연 증가
- Agent 전체에서 동일한 "두뇌"를 공유하는 개념
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    # temperature = 0
    # 출력의 확률적 변동 제거
    # EXACT MATCH 평가에서 매우 중요
    temperature=0
)


# 4. Node 정의: solve_question
"""
Node
- LangGraph에서 Node는 "상태를 변환하는 함수"
- 반드시 다음 형태를 가짐

    def node(state: State) -> dict:
        return { 수정할_state_key: 값 }

- 반환값은 "State 전체"가 아니라
  "State에 병합될 부분 dict"
"""
def solve_question(state: AgentState) -> dict:
    """
    GAIA 질문을 실제로 '푸는' 핵심 노드

    이 Node의 역할
    1. State에서 question을 꺼낸다
    2. GAIA 평가 규칙을 강제한 프롬프트를 만든다
    3. LLM을 호출한다
    4. 결과를 answer 필드에 저장한다

    이 Node 하나만 있어도
    - LangGraph 구조 요건 충족
    - GAIA Level 1 제출 가능
    """

    # 현재 상태에서 질문 추출
    question = state["question"]

    # 프롬프트 설계
    """
    프롬프트에서 가장 중요한 포인트

    - "ONLY the final answer"를 명시적으로 강제
    - 설명/형식/불필요한 텍스트 금지
    - 모델이 '친절하게 설명하려는 습관'을 차단

    이 부분이 약하면
    정답을 맞아도 형식 오류로 0점
    """

    prompt = f"""
You are solving a GAIA benchmark question.

Rules:
- Answer with ONLY the final answer
- No explanation
- No formatting
- No extra text

Question:
{question}
""".strip()

    # LLM 호출
    """
    LangChain에서는 LLM을 메시지 리스트로 호출한다.
    여기서는 가장 단순한 HumanMessage 1개만 사용.
    """
    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    # State 업데이트 반환
    """
    LangGraph의 규칙:
    - Node는 "State 전체"를 반환하지 않는다
    - 수정할 필드만 dict로 반환
    - LangGraph가 자동으로 병합
    """
    return {
        # strip()은 EXACT MATCH를 위한 안전장치
        "answer": response.content.strip()
    }


# 5. LangGraph 그래프 구성
"""
Agent의 사고 흐름을 그래프로 정의한다.

이번 Agent의 흐름은 매우 단순하다.

START->solve_question->END
"""
# 그래프 빌더 
builder = StateGraph(AgentState)

# Node 
# "solve"라는 이름으로 solve_question 함수를 등록
builder.add_node("solve", solve_question)

#  Edge(흐름) 
# 시작점 -> solve 노드
builder.add_edge(START, "solve")

# solve 노드 -> 종료
builder.add_edge("solve", END)

# 그래프 컴파일
"""
compile()을 호출해야
- 실제 실행 가능한 Agent 객체가 생성됨
- invoke()로 실행 가능
"""
langgraph_agent = builder.compile()


# 6. 기존 BasicAgent를 LangGraph Agent로 교체
"""
중요 포인트
- run_and_submit_all() 함수는 agent(question) 형태를 기대함
- 따라서 LangGraph Agent를 감싸는 Wrapper 클래스를 유지
"""
class BasicAgent:
    """
    LangGraph 기반 Agent Wrapper

    역할
    - 외부에서는 기존 BasicAgent처럼 보이게 유지
    - 내부에서는 LangGraph Agent를 실행
    """
    def __init__(self):
        # Space 로그에서 Agent 초기화 여부 확인용
        print("LangGraphAgent initialized.")

    def __call__(self, question: str) -> str:
        """
        GAIA 서버 -> Agent -> 정답 반환 흐름의 진입점

        Args:
            question : str (GAIA에서 받은 문제 문자열)

        Returns
            str: 최종 정답 (EXACT MATCH 대상)
        """
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        # LangGraph는 dict 형태의 State를 입력으로 받음
        result_state = langgraph_agent.invoke({
            "question": question,
            "answer": ""  # 초기값 (Node에서 덮어씀)
        })

        final_answer = result_state["answer"]

        print(f"Agent returning answer: {final_answer}")

        return final_answer
```

## 6. 개선 방향

- `solve_question` 노드 내부에:
  - 질문 유형 분기
  - 정규식 기반 후처리
  - 파일 다운로드 처리
- ToolNode 추가 (검색, 계산)
- LangGraph conditional edge 활용

참고자료
Huggingface, agents course, https://huggingface.co/learn