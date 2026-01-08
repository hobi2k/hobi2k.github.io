---
layout: post
title:  "허깅페이스 에이전트 코스 - Vision Agents with smolagents"
date:   2025-01-08 00:10:22 +0900
categories: Huggingface_agent
---

# Vision Agents with smolagents

## 1. 왜 Vision Agent가 필요한가

텍스트 기반 LLM만으로는 해결할 수 없는 문제가 매우 많다.

대표적인 예:
- 웹 페이지 탐색 (레이아웃, 버튼, 팝업)
- 신분 확인, 얼굴/의상/외형 판별
- 문서·이미지 기반 정보 검증
- UI 상태 인식 (스크린샷 기반)

이러한 문제를 해결하려면 **Vision-Language Model(VLM)** 이 필요하다.  
`smolagents`는 이러한 VLM을 **에이전트 구조 안에 자연스럽게 통합**할 수 있도록 설계되어 있다.

## 2. smolagents에서 Vision Agent의 핵심 개념

Vision Agent는 본질적으로 **멀티모달 입력을 처리하는 CodeAgent**다.

- 텍스트 + 이미지 입력을 동시에 받음
- 이미지가 에이전트 메모리(step log)에 저장됨
- ReAct 사이클(Thought-Action-Observation)에 이미지가 포함됨

즉, Vision Agent는 새로운 Agent 타입이 아니라  
**이미지를 이해할 수 있는 모델을 사용하는 Agent 구성 방식**이다.

## 3. Vision Agent 사용 방식 개요

smolagents에서 Vision Agent를 구성하는 방식은 크게 두 가지다.

### 실행 시작 시 이미지 제공 (Static Vision Input)
- 이미지가 처음부터 주어짐
- 얼굴 인식, 의상 비교, 이미지 설명에 적합

### 실행 중 이미지 동적 획득 (Dynamic Vision Retrieval)
- 웹 탐색 중 스크린샷을 수집
- 브라우저 에이전트, UI 분석에 적합

## 4. 방식: 실행 시작 시 이미지 제공

### 4.1 시나리오 설명

알프레드는 파티 입장객의 신원을 확인해야 한다.  
손님이 원더우먼이라고 주장하지만, 조커일 가능성이 있다.

알프레드는:
- 기존 데이터셋(이미지)을 보유
- 새로운 방문자의 외형을 비교
- 신원을 판단

### 4.2 이미지 로드 코드

```python
from PIL import Image
import requests
from io import BytesIO

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg"
]

images = []
for url in image_urls:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)
```

설명

- PIL을 사용해 이미지를 RGB 포맷으로 로드
- images 리스트는 이후 Agent 실행 시 전달됨
- 실제 환경에서는 내부 DB 이미지, CCTV 프레임 등이 될 수 있음

### 4.3 Vision Agent 생성 및 실행

```python
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(model_id="gpt-4o")

agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=20,
    verbosity_level=2
)

response = agent.run(
    """
    Describe the costume and makeup that the comic character
    in these photos is wearing.
    Tell me if the guest is The Joker or Wonder Woman.
    """,
    images=images
)
```

설명

- OpenAIServerModel(gpt-4o)는 VLM
- images 인자는 task_images로 저장됨
- 에이전트는 이미지 + 텍스트를 동시에 분석

### 4.4 출력 예시 해석

```python
{
    'Costume and Makeup - First Image': (
        'Purple coat...',
        'White face paint...'
    ),
    'Costume and Makeup - Second Image': (
        'Dark suit...',
        'Green hair...'
    ),
    'Character Identity': 'This character resembles The Joker'
}
```

이 결과로:

- 외형 분석
- 분장 특징 추출
- 캐릭터 정체 판별

즉, Vision 기반 신원 검증이 완료된다.

## 5. 방식: 실행 중 이미지 동적 획득 (Dynamic Vision)

### 5.1 왜 동적 Vision이 필요한가

정적 이미지가 없는 경우:

- 처음 보는 인물
- 웹에서 정보 탐색 필요
- UI 상태 기반 판단 필요

이 경우 Agent는:

- 웹을 탐색
- 스크린샷을 촬영
- 그 이미지를 다시 분석

## 6. smolagents 내부 구조 관점에서의 Vision

Vision Agent는 다음 step log를 활용한다.

- SystemPromptStep
- TaskStep
- ActionStep
    - 여기서 스크린샷이 observation_images로 저장됨

즉, 이미지도 Observation의 일부다.

## 7. 동적 Vision을 위한 도구 설치

```bash
pip install "smolagents[all]" helium selenium python-dotenv
```

## 8. 브라우저 제어용 Tool 정의

### 8.1 Ctrl+F 검색 Tool

```python
@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F
    and jumps to the nth occurrence.
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(
            f"Match n°{nth_result} not found "
            f"(only {len(elements)} matches found)"
        )
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    return f"Focused on match {nth_result}"
```

### 8.2 페이지 이동 Tool

```python
@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()
```

### 8.3 팝업 닫기 Tool

```python
@tool
def close_popups() -> str:
    """
    Closes visible modal or pop-up windows.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    return "Popups closed"
```

## 9. 스크린샷 저장 로직

```python
def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)
    driver = helium.get_driver()
    if driver is not None:
        image = Image.fromarray(driver.get_screenshot_as_png())
        step_log.observation_images = [image.copy()]
```

설명

- 스크린샷을 Observation에 저장
- 이후 VLM이 해당 이미지를 인식 가능
- Agent 메모리에 시각 정보가 누적됨

## 10. 최종 결과 예시

```python
Final answer:
Wonder Woman is typically depicted wearing a red and gold bustier,
blue shorts with white stars, a golden tiara, silver bracelets,
and the Lasso of Truth.
```

이 결과는:

- 웹 탐색
- 시각 정보 수집
- VLM 분석
- 최종 판단

이라는 전체 파이프라인의 결과다.

## 11. 정적 Vision vs 동적 Vision 비교


| 항목          | 정적 Vision | 동적 Vision |
| ----------- | --------- | --------- |
| 이미지 제공 시점   | 시작 시      | 실행 중      |
| 사용 사례       | 얼굴/의상 비교  | 웹 브라우징    |
| 구현 난이도      | 낮음        | 높음        |
| 자동화 가능성     | 제한적       | 매우 높음     |
| 실제 Agent 활용 | 중간        | 핵심        |


## 12. 핵심 정리

- Vision Agent는 별도 타입이 아니다
- VLM + CodeAgent 조합이다
- 이미지도 Observation의 일부다
- 브라우저 + 스크린샷 + VLM 조합은 강력하다
- 실제 UI 기반 Agent는 거의 항상 Vision이 필요하다

## 13. 결론

Vision Agents는
smolagents를 텍스트 도우미에서 실제 세계 상호작용 시스템으로
격상시키는 핵심 기능이다.

웹 브라우저, 문서, UI, 이미지가 개입되는 순간
Vision Agent는 선택이 아니라 필수 아키텍처다.

참고자료
Huggingface, agents course, https://huggingface.co/learn