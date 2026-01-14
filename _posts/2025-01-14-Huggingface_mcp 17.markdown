---
layout: post
title:  "허깅페이스 MCP 코스 - Module 1: Build MCP Server"
date:   2025-01-14 00:10:22 +0900
categories: Huggingface_mcp
---

# Module 1: Build MCP Server
## Intelligent PR Agent with MCP Tools

Module 1에서는 MCP Tools를 사용해 **Pull Request 작성 과정을 자동화하는 MCP 서버**를 구축한다.  
이 서버는 Claude Code와 연동되어, 코드 변경 내용을 분석하고 **적절한 PR 템플릿과 설명을 제안하는 PR Agent**의 역할을 수행한다.

## 문제 배경: PR Chaos

실제 개발 현장에서 자주 발생하는 문제는 다음과 같다.

- PR 제목과 설명이 의미를 전달하지 못함  
  - 예: `fix`, `update things`, `various improvements`
- 리뷰어가 변경 의도를 파악하기 위해 직접 diff를 해석해야 함
- 변경 파일 수가 많을수록 리뷰 비용과 리스크 증가

이 문제의 본질은 **“변경은 존재하지만 설명이 없다”**는 점이다.

## 목표

이 Module의 목표는 다음과 같다.

- Git 변경 내용을 구조화된 데이터로 제공하는 MCP Tool 구현
- PR 템플릿 목록을 Claude에게 제공
- **Claude가 변경 맥락을 이해하고 가장 적절한 PR 템플릿과 설명을 생성하도록 지원**

중요한 설계 원칙은 **서버가 판단하지 않고, Claude가 판단하게 하는 것**이다.

## MCP 설계 철학

이 Module은 MCP의 핵심 철학을 그대로 따른다.

- 서버가 규칙 기반으로 PR 유형을 결정하지 않음
- 서버는 **풍부한 원시 데이터 제공**
- **Claude가 분석 · 판단 · 서술**

즉,
> MCP Tool은 “결론”이 아니라  
> **LLM이 결론을 낼 수 있는 재료를 제공하는 역할**을 한다.-

## 구현할 MCP Tools

### 1. analyze_file_changes
- Git diff, 변경 파일, 통계 정보 수집
- 대규모 diff에 대비한 출력 제한 필수

```python
@mcp.tool()
async def analyze_file_changes(
    base_branch: str = "main",
    include_diff: bool = True,
    max_diff_lines: int = 500
) -> str:
    """Analyze git changes with output limiting.

    Args:
        base_branch: Branch to compare against
        include_diff: Whether to include diff content
        max_diff_lines: Maximum number of diff lines
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True
        )

        diff_lines = result.stdout.split("\n")

        if len(diff_lines) > max_diff_lines:
            diff_output = "\n".join(diff_lines[:max_diff_lines])
            diff_output += (
                f"\n\n... Output truncated. "
                f"Showing {max_diff_lines} of {len(diff_lines)} lines ..."
            )
        else:
            diff_output = result.stdout

        stats = subprocess.run(
            ["git", "diff", "--stat", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True
        )

        return json.dumps({
            "stats": stats.stdout,
            "total_lines": len(diff_lines),
            "diff": diff_output if include_diff else "Diff omitted",
        })

    except Exception as e:
        return json.dumps({"error": str(e)})
```

## 2. get_pr_templates
- 사용 가능한 PR 템플릿 목록 제공
- Claude의 선택을 위한 리소스 역할

```python
@mcp.tool()
async def get_pr_templates() -> str:
    """Return available PR templates."""
    templates = [
        "Feature",
        "Bugfix",
        "Refactor",
        "Security",
        "Documentation",
        "Chore",
        "Test"
    ]
    return json.dumps({"templates": templates})
```

## 3. suggest_template
- 서버는 로직을 강제하지 않음
- Claude가 analyze 결과를 바탕으로 판단

```python
@mcp.tool()
async def suggest_template(analysis_summary: str) -> str:
    """Allow Claude to recommend the best PR template."""
    return json.dumps({
        "instruction": (
            "Based on the provided analysis summary, "
            "recommend the most appropriate PR template "
            "and explain your reasoning."
        ),
        "analysis": analysis_summary
    })
```

**출력 크기 제한**
MCP Tool 응답에는 25,000 토큰 제한이 있다.
Git diff는 이 제한을 매우 쉽게 초과한다.

```
Error: MCP tool response (262521 tokens) exceeds maximum allowed tokens (25000)
```

**교훈**
- Tool은 모든 데이터를 반환하지 않는다
- “판단에 충분한 정보”만 반환한다

**적용 전략**
- diff 라인 수 제한
- stat 요약 우선 제공
- 추가 정보는 재요청 방식

**실행 디렉토리 문제와 해결**
MCP 서버는 기본적으로 자신의 설치 디렉토리에서 실행된다.
Claude Code의 작업 디렉토리와 다를 수 있다.

**해결: MCP Roots 사용**
```python
context = mcp.get_context()
roots_result = await context.session.list_roots()
working_dir = roots_result.roots[0].uri.path

subprocess.run(
    ["git", "diff"],
    cwd=working_dir,
    capture_output=True,
    text=True
)
```
이렇게 해야 Claude가 작업 중인 저장소를 정확히 분석한다.

### 테스트 절차
1. 정적 검증
```bash
uv run python validate_starter.py
```

2. 유닛 테스트
```bash
uv run pytest test_server.py -v
```

3. Claude Code 연동
```bash
claude mcp add pr-agent -- uv --directory /absolute/path/to/starter run server.py
claude mcp list
```

Claude에게 질문:

“내 변경 사항을 분석해서 PR 템플릿을 추천해줘”

## Module 1에서 얻는 것
성과
- 실사용 가능한 MCP Tool 서버
- PR 자동 분석 파이프라인
- 출력 제한·에러 처리 실전 경험

재사용 가능한 패턴
- 외부 시스템 데이터 수집 Tool
- LLM 중심 의사결정 구조
- 대용량 출력 관리
- JSON 기반 에러 반환
- 테스트 중심 MCP 개발 흐름

참고자료
Huggingface, agents course, https://huggingface.co/learn