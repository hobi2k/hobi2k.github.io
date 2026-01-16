---
layout: post
title:  "허깅페이스 MCP 코스 - Module 3: Slack Notification"
date:   2025-01-15 00:10:22 +0900
categories: Huggingface_mcp
---

# Module 3: Slack Notification
## Completing the MCP Automation Pipeline with Team Communication

Module 3에서는 Module 1·2에서 구축한 자동화 시스템을 완성한다.  
목표는 단순하다.

> **“중요한 일이 발생했을 때, 사람이 말하지 않아도 팀이 알게 하자.”**

이를 위해 MCP Tools와 MCP Prompts를 결합하여  
**GitHub Actions -> Claude -> Slack**으로 이어지는 자동 알림 파이프라인을 구축한다.

## 문제 배경: Communication Gap

Module 1과 2로 다음 문제는 해결되었다.

- PR 설명 품질 개선
- CI/CD 실패의 실시간 감지

그러나 새로운 문제가 드러난다.

- 이미 해결된 이슈를 다른 팀이 다시 디버깅
- 준비된 작업 결과가 공유되지 않아 방치
- “누가, 언제, 무엇을 했는지” 팀 전체가 모름

이는 기술 문제가 아니라 **정보 전달 문제**다.

## Module 3의 목표

Module 3의 목표는 다음과 같다.

- GitHub Actions 이벤트를 **팀 채널에 자동 공유**
- 실패/성공에 따라 **다른 메시지 포맷 제공**
- Claude가 상황을 이해하고 **적절한 알림을 선택**
- 모든 과정이 MCP 서버 내부에서 자동화

결과적으로 MCP 서버는 **팀 커뮤니케이션 허브**가 된다.

## Module 3에서 추가되는 구성 요소

Module 3에서 새로 추가되는 요소:

- Slack Incoming Webhook 연동 Tool
- Slack 메시지 포맷 전용 MCP Prompts
- Module 2의 CI 이벤트 Tools와 결합
- 모든 MCP primitive (Tools + Prompts + Integration) 완성

## 핵심 개념

### 1. MCP 통합 패턴 (End-to-End)

Module 3는 MCP의 이상적인 통합 흐름을 보여준다.

1. **Events**  
   GitHub Actions -> Webhook (Module 2)

2. **Prompts**  
   CI 이벤트를 사람이 읽기 좋은 메시지로 변환

3. **Tools**  
   Slack Webhook으로 외부 시스템 호출

4. **Result**  
   팀 채널에 자동 알림 전송

### 2. Slack 메시지 포맷

Slack은 GitHub Markdown과 다르다.  
반드시 Slack 전용 마크업을 사용해야 한다.

- `*bold*` (`**bold**` 아님)
- `_italic_`
- `` `inline code` ``
- `> block quote`
- Emoji (`✅ ❌ 🚨 ⚠️`)
- 링크: `<URL|TEXT>`

## 프로젝트 구조

```
slack-notification/
├── starter/
│ ├── server.py # Module 1+2 + TODO
│ ├── webhook_server.py # Module 2
│ ├── pyproject.toml
│ └── README.md
└── solution/
├── server.py # Slack 통합 포함
├── webhook_server.py
└── README.md
```


## 구현 단계

## Step 1. Slack Incoming Webhook 설정

1. Slack App 생성  
   https://api.slack.com/apps

2. Incoming Webhooks 활성화  
   - Features → Incoming Webhooks
   - 채널 선택 후 Webhook URL 복사

3. Webhook 테스트

```bash
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Hello from MCP Course!"}' \
  YOUR_WEBHOOK_URL
```

4. 환경 변수 설정

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```
Webhook URL은 비밀 키다.
절대 코드에 하드코딩하거나 커밋하지 않는다.

## Step 2. Slack 알림 MCP Tool 추가
Module 2의 server.py에 Tool을 추가한다.

```python
import os
import requests

@mcp.tool()
def send_slack_notification(message: str) -> str:
    """Send a formatted notification to the team Slack channel."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return "Error: SLACK_WEBHOOK_URL environment variable not set"
    
    try:
        response = requests.post(
            webhook_url,
            json={
                "text": message,
                "mrkdwn": True
            },
            timeout=5,
        )
        if response.status_code != 200:
            return f"Slack error: {response.text}"
        return "Slack notification sent successfully"
    except Exception as e:
        return f"Error sending message: {str(e)}"
```

이 Tool은:

- 외부 API 호출
- 환경 변수 기반 보안 처리
- 실패 시 Claude가 이해할 수 있는 메시지 반환

## Step 3. Slack 메시지 포맷 MCP Prompts
Prompts는 “어떻게 말할지”를 표준화한다.

### CI 실패 알림 Prompt
```python
@mcp.prompt()
def format_ci_failure_alert() -> str:
    """Create a Slack alert for CI/CD failures."""
    return """Format this GitHub Actions failure as a Slack message:

Use this template:
:rotating_light: *CI Failure Alert* :rotating_light:

A CI workflow has failed:
*Workflow*: workflow_name
*Branch*: branch_name
*Status*: Failed
*View Details*: <LOGS_LINK|View Logs>

Please check the logs and address any issues.

Use Slack markdown formatting and keep it concise."""
```

### CI 성공 요약 Prompt
```python
@mcp.prompt()
def format_ci_success_summary() -> str:
    """Create a Slack message celebrating successful deployments."""
    return """Format this successful GitHub Actions run as a Slack message:

Use this template:
:white_check_mark: *Deployment Successful* :white_check_mark:

Deployment completed successfully for [Repository Name]

*Changes:*
- Key feature or fix 1
- Key feature or fix 2

*Links:*
<PR_LINK|View Changes>

Keep it celebratory but informative."""
```

### Step 4. 전체 시스템 테스트
모든 서비스를 동시에 실행한다.

```bash
# Terminal 1
python webhook_server.py

# Terminal 2
uv run server.py

# Terminal 3
cloudflared tunnel --url http://localhost:8080
```

이제 실제 또는 가짜 GitHub 이벤트를 통해 전체 흐름을 검증할 수 있다.

## Claude Code에서의 실제 워크플로우

```
User: Check recent CI events and notify the team about any failures

Claude:
1. get_recent_actions_events 호출
2. 실패 이벤트 탐지
3. format_ci_failure_alert Prompt 사용
4. send_slack_notification Tool 호출
5. 결과 보고
```

## Slack 메시지 예시
실패 알림

```yaml
🚨 *CI Failure Alert* 🚨

A CI workflow has failed:
*Workflow*: CI (Run #42)
*Branch*: feature/slack-integration
*Status*: Failed
*View Details*: <https://github.com/user/repo/actions/runs/123|View Logs>
```

성공 알림
```
✅ *Deployment Successful* ✅

Deployment completed successfully for mcp-course

*Changes:*
- Added Slack notifications
- Integrated MCP Prompts

*Links:*
<https://github.com/user/repo/pull/42|View Changes>
```

## 자주 발생하는 문제

Slack 메시지가 안 보일 때

- Webhook URL 환경 변수 확인
- curl 테스트 선행
- mrkdwn: true 여부 확인

포맷이 깨질 때

- Slack은 *bold*만 지원
- 특수 문자 escape 필요
- 커밋 메시지 그대로 넣지 말 것

## Module 3에서 완성된 것

- 외부 API 연동 MCP Tool
- 포맷 표준화를 위한 MCP Prompts
- 이벤트 -> 분석 -> 알림의 완전 자동화
- 실제 팀에서 바로 쓸 수 있는 시스템

참고자료
Huggingface, mcp course, https://huggingface.co/learn