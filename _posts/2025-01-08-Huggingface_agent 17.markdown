---
layout: post
title:  "허깅페이스 에이전트 코스 - Multi-Agent Systems"
date:   2025-01-08 00:10:22 +0900
categories: Huggingface_agent
---

# Multi-Agent Systems

## 1. 왜 Multi-Agent Systems가 필요한가

**Multi-Agent System(MAS)**은  
하나의 에이전트가 모든 일을 처리하는 대신,  
**서로 다른 역할을 가진 여러 에이전트가 협력**하여 문제를 해결하는 구조다.

단일 에이전트 방식의 한계는 명확하다.

- 컨텍스트 윈도우가 빠르게 소모됨
- 복잡한 문제일수록 추론 품질 저하
- 검색 + 계산 + 시각화 + 검증을 동시에 수행하면 불안정
- 모든 책임이 하나의 사고 흐름에 집중됨

Multi-Agent 구조는 이를 다음 방식으로 해결한다.

- 역할 분리 (Separation of Concerns)
- 메모리 분리
- 병렬적 사고 구조
- 상위 에이전트가 하위 에이전트를 관리

## 2. smolagents에서의 Multi-Agent 개념

`smolagents`에서 Multi-Agent 시스템은 다음 구조를 가진다.

- **Manager / Orchestrator Agent**
  - 전체 문제를 분해
  - 하위 에이전트에게 작업 위임
  - 결과를 종합하고 최종 출력 생성

- **Worker Agents**
  - 특정 작업에 특화
  - 예: 웹 검색, 계산, 문서 분석, 이미지 생성

각 Agent는 **독립적인 메모리와 도구 집합**을 가진다.

## 3. 예제 시나리오 설명

### 문제 설정

> 전 세계의 배트맨 영화 촬영지를 찾고  
> 고담(뉴욕 기준)에서 화물기로 이동하는 시간을 계산한 뒤  
> 같은 이동 시간을 가진 슈퍼카 공장들도 함께  
> 세계 지도 위에 시각화하라

이 문제는 단일 에이전트에 매우 불리하다.

- 대규모 웹 검색
- 좌표 수집
- 거리 계산
- 데이터프레임 생성
- 지도 시각화
- 결과 검증

따라서 **Multi-Agent 구조가 필수**다.

## 4. 사전 패키지 설치

```bash
pip install 'smolagents[litellm]' plotly geopandas shapely kaleido -q
```

## 5. 핵심 Tool: 화물기 이동 시간 계산

이 Tool은 모든 Agent가 공통으로 사용할 수 있다.

```python
import math
from typing import Optional, Tuple
from smolagents import tool

@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth.

    Args:
        origin_coords: (latitude, longitude)
        destination_coords: (latitude, longitude)
        cruising_speed_kmh: cruising speed (km/h)

    Returns:
        Estimated travel time in hours
    """

    def to_radians(deg: float) -> float:
        return deg * math.pi / 180

    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    EARTH_RADIUS_KM = 6371.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    actual_distance = distance * 1.1
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    return round(flight_time, 2)
```

## 6. 단일 에이전트 기준선(Baseline)

먼저 단일 에이전트로 시도한다.

```python
from smolagents import CodeAgent, GoogleSearchTool, InferenceClientModel, VisitWebpageTool

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    provider="together"
)

task = """
Find all Batman filming locations in the world,
calculate the time to transfer via cargo plane to Gotham (40.7128° N, 74.0060° W),
and return them as a pandas dataframe.
Also give me some supercar factories with the same travel time.
"""

agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool("serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    additional_authorized_imports=["pandas"],
    max_steps=20,
)

result = agent.run(task)
print(result)
```

문제점

- 검색 결과가 과도하게 많아짐
- 컨텍스트 포화
- 느려짐
- 시각화 단계로 확장 불가

## 7. planning_interval로 사고 능력 개선

```python
agent.planning_interval = 4

detailed_report = agent.run(f"""
You're an expert analyst.
Search extensively and verify data by visiting source URLs.

{task}
""")

print(detailed_report)
```

한계

- 여전히 모든 책임이 단일 메모리에 집중
- 확장 불가능

## 8. Multi-Agent 구조로 분리

구조 설계

- Web Agent
    - 검색 전용
    - 웹 탐색 + 좌표 수집

- Manager Agent
    - 계획 수립
    - Web Agent 호출
    - 결과 통합
    - 지도 시각화
    - 최종 검증

## 9. Web Agent 생성

```python
web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(provider="serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)
```

### 10. Manager Agent 생성 (고급 추론)

Manager는 더 강한 모델과 더 많은 권한을 가진다.

```python
from smolagents import OpenAIServerModel
from PIL import Image
import os

def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath)

    image = Image.open(filepath)

    prompt = (
        f"Here is the agent reasoning: {agent_memory.get_succinct_steps()}."
        "Check whether the plot answers the task correctly."
        "Return PASS or FAIL."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": filepath}},
        ],
    }]

    output = multimodal_model(messages).content
    if "FAIL" in output:
        raise Exception(output)

    return True

manager_agent = CodeAgent(
    model=InferenceClientModel(
        "deepseek-ai/DeepSeek-R1",
        provider="together",
        max_tokens=8096
    ),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)
```

## 11. 구조 시각화

```python
manager_agent.visualize()
```

이 시각화는 다음을 보여준다.

- Agent 계층 구조
- 각 Agent의 도구
- 권한(import)
- 책임 분리 상태

12. 최종 실행

```python
manager_agent.run("""
Find all Batman filming locations in the world,
calculate cargo plane transfer time to Gotham,
add supercar factories with similar transfer time,
plot everything on a world map using px.scatter_map,
save it to saved_map.png,
and return the figure.
""")
```

## 13. 결과 확인

```python
manager_agent.python_executor.state["fig"]
```

이 결과는:

- 전 세계 좌표 시각화
- 이동 시간에 따른 색상 변화
- 배트맨 촬영지 + 슈퍼카 공장 통합

## 14. Multi-Agent 구조의 핵심 장점 요약


| 항목    | 단일 Agent | Multi-Agent |
| ----- | -------- | ----------- |
| 확장성   | 낮음       | 높음          |
| 비용    | 높음       | 낮음          |
| 안정성   | 불안정      | 안정          |
| 디버깅   | 어려움      | 쉬움          |
| 역할 분리 | 없음       | 명확          |


## 15. 핵심 설계 원칙

- 검색은 검색 에이전트
- 계획은 관리자
- 계산은 Tool
- 검증은 별도 체크
- 메모리는 분리

## 결론

Multi-Agent 시스템은
“에이전트를 여러 개 쓰는 것”이 아니라
사고 구조를 분산하는 아키텍처다.

복잡한 문제일수록
smolagents의 managed_agents 구조는
사실상 필수에 가깝다.

참고자료
Huggingface, agents course, https://huggingface.co/learn