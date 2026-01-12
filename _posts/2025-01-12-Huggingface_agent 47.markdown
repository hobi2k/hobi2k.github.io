---
layout: post
title:  "허깅페이스 에이전트 코스 - Pokémon Battle LLM Agent 1"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# Pokémon Battle LLM Agent 1

이 글은 poke-env 기반 Pokémon Showdown Agent에
LLM(OpenAI / Gemini / Mistral)를 **의사결정 엔진**으로
결합하기 위한 공통 베이스 + 구현체를 독해하는 데 목적을 두고 있다.

## agents.py

```python
# 1. 기본 라이브러리
import os
import json
import asyncio
import random

# 2. LLM Provider SDK
# OpenAI
# AsyncOpenAI:
#   - 비동기 호출을 지원하는 OpenAI 클라이언트
# APIError:
#   - OpenAI API 호출 실패 시 예외 타입
from openai import AsyncOpenAI, APIError

# Google Gemini
# genai.Client:
#   - Gemini API 메인 클라이언트
# types:
#   - Tool / Function Calling 관련 설정 타입
from google import genai
from google.genai import types

# Mistral AI
# Mistral:
#   - Mistral API 메인 클라이언트
from mistralai import Mistral

# 3. poke-env (Pokémon Showdown 인터페이스)

# Player:
#   - Pokémon Showdown과 실제로 통신하는 클래스
#   - choose_move 메서드를 override 하면 "플레이어"가 된다
from poke_env.player import Player

# 타입 힌트용
from typing import Optional, Dict, Any, Union

# 4. 공통 유틸 함수
def normalize_name(name: str) -> str:
    """
    문자열 정규화 함수

    목적:
    - LLM은 동일한 기술/포켓몬을 다양한 표기로 출력할 수 있다.
      예: "Thunder Bolt", "thunderbolt", "Thunderbolt"
    - poke-env 내부에서는 id 기반(lowercase, alnum)으로 관리됨
    - 비교 전 반드시 정규화 필요

    동작:
    - 소문자로 변환
    - 알파벳/숫자 외 문자 제거
    """
    return "".join(filter(str.isalnum, name)).lower()

# 5. 표준 Tool(Function) 스키마
"""
STANDARD_TOOL_SCHEMA는 "이 에이전트가 할 수 있는 행동의 전부"다.

LLM은:
- 여기 정의된 함수만 호출할 수 있고
- 정의되지 않은 행동은 절대 실행되지 않는다

즉, 이 스키마가 곧 Action Space 이다.
"""
STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        "name": "choose_move",
        "description": "Selects and executes an available attacking or status move.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_name": {
                    "type": "string",
                    "description": (
                        "The exact name or ID of the move to use. "
                        "Must be one of the available moves."
                    ),
                },
            },
            "required": ["move_name"],
        },
    },
    "choose_switch": {
        "name": "choose_switch",
        "description": "Selects an available Pokémon from the bench to switch into.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon_name": {
                    "type": "string",
                    "description": (
                        "The exact species name of the Pokémon to switch to. "
                        "Must be one of the available switches."
                    ),
                },
            },
            "required": ["pokemon_name"],
        },
    },
}

# 6. OpenAI 전용 Tool Schema
"""
OpenAI API는 tool 정의 시 반드시
"type": "function" 래퍼를 요구한다.

논리적으로는 STANDARD_TOOL_SCHEMA와 동일하며,
형식만 OpenAI 요구사항에 맞게 변환한 것이다.
"""
OPENAI_TOOL_SCHEMA = {
    "choose_move": {
        "type": "function",
        "function": STANDARD_TOOL_SCHEMA["choose_move"],
    },
    "choose_switch": {
        "type": "function",
        "function": STANDARD_TOOL_SCHEMA["choose_switch"],
    },
}

# 7. LLMAgentBase
"""
LLMAgentBase는 이 파일의 **핵심 클래스**다.

역할:
- poke-env Player를 상속
- "LLM 의사결정"과 "게임 실행"을 완전히 분리
- 모든 LLM(OpenAI/Gemini/Mistral)은 이 클래스를 상속

핵심:
- LLM은 '결정'만 한다
- 실행, 검증, 폴백은 전부 코드가 담당
"""
class LLMAgentBase(Player):

    def __init__(self, *args, **kwargs):
        # Player 초기화 (Showdown 연결 포함)
        super().__init__(*args, **kwargs)

        # LLM에게 노출할 표준 툴
        self.standard_tools = STANDARD_TOOL_SCHEMA

        # 확장용: 이전 턴 기록 저장 가능
        self.battle_history = []

    # Battle 상태를 LLM 입력용 문자열로 변환
    def _format_battle_state(self, battle) -> str:
        """
        Battle 객체는 매우 복잡한 Python 객체다.
        LLM은 이를 이해할 수 없으므로,
        '결정에 필요한 정보만' 요약해 문자열로 만든다.
        """

        # 내 포켓몬 정보
        active = battle.active_pokemon
        active_info = (
            f"Your active Pokemon: {active.species} "
            f"(Type: {'/'.join(map(str, active.types))}) "
            f"HP: {active.current_hp_fraction * 100:.1f}% "
            f"Status: {active.status.name if active.status else 'None'} "
            f"Boosts: {active.boosts}"
        )

        # 상대 포켓몬 정보
        opp = battle.opponent_active_pokemon
        if opp:
            opp_info = (
                f"{opp.species} "
                f"(Type: {'/'.join(map(str, opp.types))}) "
                f"HP: {opp.current_hp_fraction * 100:.1f}% "
                f"Status: {opp.status.name if opp.status else 'None'} "
                f"Boosts: {opp.boosts}"
            )
        else:
            opp_info = "Unknown"

        # 사용 가능한 기술
        if battle.available_moves:
            moves = "\n".join(
                f"- {m.id} (Type:{m.type}, BP:{m.base_power}, Acc:{m.accuracy})"
                for m in battle.available_moves
            )
        else:
            moves = "- None (Must switch or Struggle)"

        # 교체 가능 포켓몬
        if battle.available_switches:
            switches = "\n".join(
                f"- {p.species} (HP:{p.current_hp_fraction*100:.1f}%)"
                for p in battle.available_switches
            )
        else:
            switches = "- None"

        # 최종 문자열
        return f"""
{active_info}
Opponent's active Pokemon: {opp_info}

Available moves:
{moves}

Available switches:
{switches}

Weather: {battle.weather}
Terrains: {battle.fields}
Your Side Conditions: {battle.side_conditions}
Opponent Side Conditions: {battle.opponent_side_conditions}
""".strip()

    # LLM 출력 검증: 기술 이름 -> 실제 Move 객체
    def _find_move_by_name(self, battle, move_name: str):
        normalized = normalize_name(move_name)

        # 1순위: ID 정확 매칭 (가장 안전)
        for move in battle.available_moves:
            if move.id == normalized:
                return move

        # 2순위: 표시 이름 매칭 (경고)
        for move in battle.available_moves:
            if move.name.lower() == move_name.lower():
                print(
                    f"[WARN] Matched by display name: {move.name} "
                    f"instead of id {move.id}"
                )
                return move

        return None

    # LLM 출력 검증: 교체 포켓몬
    def _find_pokemon_by_name(self, battle, pokemon_name: str):
        normalized = normalize_name(pokemon_name)
        for p in battle.available_switches:
            if normalize_name(p.species) == normalized:
                return p
        return None

    # 핵심 메서드: 한 턴의 행동 결정
    async def choose_move(self, battle) -> str:
        """
        이 함수는 Pokémon Showdown이 매 턴 호출한다.

        흐름:
        1. Battle -> 문자열 변환
        2. LLM 호출 (_get_llm_decision)
        3. Tool Call 파싱
        4. 검증
        5. 실패 시 폴백 (무조건 행동 보장)
        """

        state = self._format_battle_state(battle)
        result = await self._get_llm_decision(state)

        decision = result.get("decision")
        error = result.get("error")

        # LLM이 정상적인 Tool Call을 준 경우
        if decision:
            name = decision.get("name")
            args = decision.get("arguments", {})

            # 기술 사용
            if name == "choose_move":
                move = self._find_move_by_name(battle, args.get("move_name", ""))
                if move:
                    return self.create_order(move)

            # 교체
            if name == "choose_switch":
                pkmn = self._find_pokemon_by_name(battle, args.get("pokemon_name", ""))
                if pkmn:
                    return self.create_order(pkmn)

        # 여기 도달 = LLM 실패
        print("[FALLBACK]", error or "Invalid LLM decision")

        # 무작위 행동으로 게임 지속 보장
        if battle.available_moves or battle.available_switches:
            return self.choose_random_move(battle)

        # 최후 수단 (Struggle)
        return self.choose_default_move(battle)

    # LLM 호출부 (하위 클래스에서 구현)
    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "LLM 호출 로직은 하위 클래스에서 구현해야 한다."
        )
"""
1. 세 에이전트는 **구조적으로 완전히 동일**하다.
2. 차이는 오직:
   - API SDK
   - Tool(Function Calling) 포맷
   - 응답 파싱 방식
3. choose_move() 로직은 절대 중복 구현하지 않는다.
   -> 모든 검증/폴백은 LLMAgentBase에 있음
4. 각 Agent는 반드시 아래 형식으로 반환해야 한다.

반환 규약 (절대 변경 금지):
{
    "decision": {
        "name": "choose_move" | "choose_switch",
        "arguments": {...}
    }
}
또는
{
    "error": "에러 메시지"
}
"""

# 1. GeminiAgent (Google Gemini API)
class GeminiAgent(LLMAgentBase):
    """
    Google Gemini API를 사용하는 Pokémon Battle Agent

    특징:
    - Google genai SDK 사용
    - Gemini의 native function calling 사용
    - Automatic function calling은 비활성화
      (LLM이 명시적으로 함수 호출하도록 강제)
    """
    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.5-pro-preview-03-25",
        avatar: str = "steven",
        *args,
        **kwargs
    ):
        # poke-env Player 옵션
        kwargs["avatar"] = avatar
        kwargs["start_timer_on_battle_start"] = True
        super().__init__(*args, **kwargs)

        self.model_name = model

        # API Key 처리
        used_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not used_api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        # Gemini Client 초기화
        self.genai_client = genai.Client(api_key=used_api_key)

        # Gemini는 function_declarations 형식 사용
        self.function_declarations = list(self.standard_tools.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        """
        Gemini에 Battle State를 보내고
        Function Call 결과만 파싱한다.
        """

        prompt = (
            "You are a skilled Pokémon battle AI.\n"
            "Decide the best action using ONLY the provided functions.\n\n"
            f"Current Battle State:\n{battle_state}\n\n"
            "Call either 'choose_move' or 'choose_switch'."
        )

        try:
            # Tool 설정
            tools = genai.types.Tool(
                function_declarations=self.function_declarations
            )

            # 자동 함수 호출 비활성화
            config = genai.types.GenerateContentConfig(
                tools=[tools],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )

            # Gemini 호출
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            # Gemini는 function_calls 배열로 반환
            calls = response.function_calls
            if calls:
                return {
                    "decision": {
                        "name": calls[0].name,
                        "arguments": calls[0].args,
                    }
                }

            return {"error": "Gemini did not return a function call."}

        except Exception as e:
            return {"error": f"Gemini error: {e}"}


# 2. OpenAIAgent (OpenAI Chat Completions)
class OpenAIAgent(LLMAgentBase):
    """
    OpenAI Chat Completions + Tool Calling 기반 Agent

    특징:
    - AsyncOpenAI 사용
    - tools + tool_choice="auto"
    - tool_calls 필드에서 함수 호출 추출
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4.1",
        avatar: str = "giovanni",
        *args,
        **kwargs
    ):
        kwargs["avatar"] = avatar
        kwargs["start_timer_on_battle_start"] = True
        super().__init__(*args, **kwargs)

        self.model = model

        used_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not used_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.openai_client = AsyncOpenAI(api_key=used_api_key)

        # OpenAI는 type="function" 래퍼 필요
        self.openai_tools = list(OPENAI_TOOL_SCHEMA.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a Pokémon battle AI.\n"
            "Always choose the best action using function calls."
        )

        user_prompt = (
            f"Current Battle State:\n{battle_state}\n\n"
            "Call 'choose_move' or 'choose_switch'."
        )

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=self.openai_tools,
                tool_choice="auto",
                temperature=0.5,
            )

            message = response.choices[0].message

            # OpenAI는 tool_calls 필드에 결과 반환
            if message.tool_calls:
                call = message.tool_calls[0]
                return {
                    "decision": {
                        "name": call.function.name,
                        "arguments": json.loads(call.function.arguments or "{}"),
                    }
                }

            return {"error": "OpenAI did not return a tool call."}

        except APIError as e:
            return {"error": f"OpenAI API error: {e.message}"}
        except Exception as e:
            return {"error": f"OpenAI unexpected error: {e}"}


# 3. MistralAgent (Mistral AI API)
class MistralAgent(LLMAgentBase):
    """
    Mistral Chat Completion 기반 Agent

    특징:
    - tool_choice="any" 로 함수 호출 강제
    - Mistral 고유 tool_calls 구조 사용
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-large-latest",
        avatar: str = "alder",
        *args,
        **kwargs
    ):
        kwargs["avatar"] = avatar
        kwargs["start_timer_on_battle_start"] = True
        super().__init__(*args, **kwargs)

        self.model = model

        used_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not used_api_key:
            raise ValueError("MISTRAL_API_KEY not set")

        self.mistral_client = Mistral(api_key=used_api_key)

        # Mistral은 OpenAI 유사하지만 직접 변환 필요
        self.mistral_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self.standard_tools.values()
        ]

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a Pokémon battle AI.\n"
            "Use the provided tools to choose an action."
        )

        user_prompt = (
            f"Current Battle State:\n{battle_state}\n\n"
            "Call 'choose_move' or 'choose_switch'."
        )

        try:
            response = self.mistral_client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=self.mistral_tools,
                tool_choice="any",
                temperature=0.3,
            )

            message = response.choices[0].message

            if message.tool_calls:
                call = message.tool_calls[0]
                return {
                    "decision": {
                        "name": call.function.name,
                        "arguments": json.loads(call.function.arguments or "{}"),
                    }
                }

            return {"error": "Mistral did not return a tool call."}

        except Exception as e:
            return {"error": f"Mistral error: {e}"}
```

## Qwen 추가 코드

```python
"""
LangGraph 기반 Pokémon Battle Agent (Qwen3-4B 버전)

목표
- 기존 Gemini/OpenAI/Mistral 구현은 "모델 자체의 function calling" 기능을 이용했다.
- 하지만 Qwen3-4B(특히 Hugging Face Inference / vLLM 서빙)는 환경에 따라
tool-calling 스키마가 제각각이거나 아예 없을 수 있다.

따라서 여기서는 "LangGraph로 에이전트 플로우를 구성"하고,
Qwen3-4B에는 다음 규칙을 강제하는 방식으로 동작시킨다.

핵심 아이디어
1) LangGraph는 '상태(State) + 노드(Node) + 엣지(Edge)'로 구성된 실행 그래프
2) 매 턴 choose_move() 호출 시:
   - battle_state 문자열 생성(LLMAgentBase가 이미 함)
   - LangGraph 실행: (plan -> llm_call -> parse -> validate -> finalize)
   - 최종적으로 {"decision": {...}} 또는 {"error": "..."} 반환
3) Qwen3-4B에는 "오직 JSON 한 덩어리"만 출력하도록 프롬프트로 강제
   - 모델의 function calling이 아니라 "JSON을 function-call로 간주"하는 방식

의존성
- langgraph
- langchain-core
- langchain-huggingface  (또는 langchain-community + HuggingFaceEndpoint)
- huggingface_hub

모델 호출 방식
아래 예시는 Hugging Face Inference Endpoint / HF_TOKEN 기반 호출을 가정한다.
- Inference Endpoint: https://... (user endpoint) 또는
- HF Serverless: HuggingFaceEndpoint가 지원하는 endpoint_url로 호출

서버/SDK 버전 차이가 있을 수 있으므로,
"Qwen3-4B 호출부"는 한 군데(llm 객체 생성 부분)만 수정하면 되게 작성한다.
"""

import os
import json
from typing import TypedDict, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, START, END

# langchain 메시지 타입(LLM 입력 포맷 표준화)
from langchain_core.messages import SystemMessage, HumanMessage

# Qwen3-4B 호출(권장: langchain-huggingface)
# 설치 예:
#   pip install langgraph langchain-core langchain-huggingface huggingface_hub
#
# 참고:
# - 환경에 따라 HuggingFaceEndpoint의 import 경로가 달라질 수 있다.
# - 아래 import가 안 되면:
#   - (대안1) langchain_community.llms import HuggingFaceEndpoint
#   - (대안2) huggingface_hub.InferenceClient로 직접 호출하는 래퍼 작성
try:
    from langchain_huggingface import HuggingFaceEndpoint
except Exception:
    HuggingFaceEndpoint = None


# 0) "LLMAgentBase"에서 요구하는 반환 규약(절대 변경 금지)
# _get_llm_decision()는 반드시 다음 중 하나를 반환해야 한다:
#  (성공) {"decision": {"name": "...", "arguments": {...}}}
#  (실패) {"error": "..."}

# 1) LangGraph 상태 정의
class BattleAgentState(TypedDict, total=False):
    """
    LangGraph는 상태를 딕셔너리처럼 흘려보낸다.
    노드는 이 상태를 읽고, 부분 업데이트(dict)를 반환한다.
    LangGraph가 이를 합쳐서 다음 노드로 전달한다.

    battle_state: LLMAgentBase가 만들어준 현재 턴 정보(문자열)
    system_prompt: 시스템 메시지(행동 규칙)
    user_prompt: 유저 메시지(현재 상태 + 출력 규칙)
    raw_model_output: LLM이 출력한 원문 텍스트
    parsed: JSON 파싱 결과(dict) 또는 None
    decision: 표준 decision 형태(LLMAgentBase 반환 규약)
    error: 에러 메시지
    """
    battle_state: str
    system_prompt: str
    user_prompt: str
    raw_model_output: str
    parsed: Optional[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]
    error: Optional[str]


# 2) Qwen3-4B LLM 클라이언트 초기화
def build_qwen_llm():
    """
    Qwen3-4B 호출부를 한 곳에 모아둔다.
    환경에 맞게 이 함수만 고치면 전체가 동작한다.

    (A) Hugging Face Inference Endpoint 사용(권장)
      - HF_TOKEN 필요
      - endpoint_url: 본인 endpoint 주소(또는 서버리스)

    Qwen3-4B 모델명은 공개 레포 이름과 다를 수 있다.
    사용자가 말한 "qwen3-4b"를 그대로 쓰기보다
    실제 배포/endpoint에 연결된 model을 사용해야 한다.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN이 필요합니다. (Hugging Face 토큰)")

    if HuggingFaceEndpoint is None:
        raise ImportError(
            "HuggingFaceEndpoint import 실패. "
            "pip install langchain-huggingface 또는 대체 경로(langchain-community)로 수정하세요."
        )

    # 예시 1) Serverless / Model ID 방식(환경별로 동작이 다를 수 있음)
    # repo_id는 실제 모델 레포로 교체해야 할 수 있음.
    #
    # 예: "Qwen/Qwen3-4B-Instruct" 같은 형태 (실제 존재 여부는 본인 환경에 맞춰 조정)
    repo_id = os.environ.get("QWEN_REPO_ID", "Qwen/Qwen3-4B-Instruct")

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,

        # 생성 파라미터(턴 기반 전투는 안정성이 중요 -> temperature 낮게)
        temperature=0.2,
        max_new_tokens=128,

        # stop를 강제하면 JSON이 끊길 수 있어 보통 비권장.
        # 대신 "JSON만 출력"을 강하게 지시하고, 파서에서 복구한다.
    )
    return llm


# 전역 1회 초기화(매 턴마다 새로 만들지 않도록)
QWEN_LLM = None


# 3) 프롬프트 정책(중요)
def build_system_prompt() -> str:
    """
    모델이 반드시 따라야 할 정책을 system 메시지로 고정한다.
    핵심은:
      - JSON만 출력
      - 허용된 action은 choose_move / choose_switch 뿐
      - arguments key 강제(move_name or pokemon_name)
    """
    return (
        "You are a Pokémon battle decision engine.\n"
        "You MUST output ONLY a single JSON object and nothing else.\n"
        "Do NOT include markdown fences, explanations, or extra text.\n\n"
        "Allowed actions:\n"
        '1) {"name":"choose_move","arguments":{"move_name":"<available_move_id>"}}\n'
        '2) {"name":"choose_switch","arguments":{"pokemon_name":"<available_species_name>"}}\n\n'
        "Rules:\n"
        "- Choose ONLY from the AVAILABLE moves/switches shown in the battle state.\n"
        "- For moves, use the exact move.id (normalized, e.g., thunderbolt, swordsdance).\n"
        "- For switches, use the exact Pokémon species name as displayed.\n"
        "- If no moves are available, you should usually switch.\n"
        "- Output must be valid JSON.\n"
    )


def build_user_prompt(battle_state: str) -> str:
    """
    user 메시지에는 "현재 battle_state"를 그대로 넣고
    마지막에 출력 형식을 재강조한다.
    """
    return (
        "Current battle state:\n"
        f"{battle_state}\n\n"
        "Return ONLY the JSON object with the best action."
    )


# 4) LangGraph 노드 구현
def node_prepare_prompts(state: BattleAgentState) -> Dict[str, Any]:
    """
    [역할]
    - battle_state를 받아 system/user 프롬프트를 구성해 상태에 저장한다.

    [LangGraph 관점]
    - 입력 state: battle_state 필수
    - 출력 업데이트: system_prompt, user_prompt
    """
    battle_state = state["battle_state"]
    return {
        "system_prompt": build_system_prompt(),
        "user_prompt": build_user_prompt(battle_state),
        "error": None,  # 이전 턴 에러가 남아있을 수 있으니 초기화
        "decision": None,
        "parsed": None,
        "raw_model_output": "",
    }


async def node_call_llm(state: BattleAgentState) -> Dict[str, Any]:
    """
    [역할]
    - Qwen3-4B를 호출하여 raw text 응답을 받아온다.

    [중요]
    - langchain LLM은 보통 invoke()가 sync일 수 있다.
      endpoint 구현에 따라 await가 안 먹을 수 있으므로
      여기서는 안전하게 "가능하면 async", 아니면 sync fallback을 둔다.

    [출력]
    - raw_model_output: 모델 응답 원문 텍스트
    """
    global QWEN_LLM
    if QWEN_LLM is None:
        QWEN_LLM = build_qwen_llm()

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=state["user_prompt"]),
    ]

    # HuggingFaceEndpoint는 보통 문자열 입력을 받는 LLM 형태인 경우가 많다.
    # 하지만 우리는 메시지 구조를 유지하고 싶으므로 간단히 합친다.
    # (이 부분은 사용하는 LLM 래퍼에 따라 조정 가능)
    flat_prompt = (
        f"[SYSTEM]\n{state['system_prompt']}\n\n"
        f"[USER]\n{state['user_prompt']}\n"
    )

    try:
        # sync invoke (대부분의 HuggingFaceEndpoint는 sync)
        text = QWEN_LLM.invoke(flat_prompt)
        # invoke 결과가 문자열이 아닐 수 있으니 안전하게 캐스팅
        return {"raw_model_output": str(text).strip()}
    except Exception as e:
        return {"error": f"LLM call failed: {e}", "raw_model_output": ""}


def _extract_json_object(text: str) -> Optional[str]:
    """
    모델이 JSON만 출력하라고 해도 가끔 앞뒤로 군더더기가 붙는다.
    이 함수는 text에서 "가장 그럴듯한 JSON object {...}"를 추출한다.

    전략:
    - 첫 '{'부터 마지막 '}'까지를 우선 잘라 시도
    - 그래도 실패하면 None
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def node_parse_decision(state: BattleAgentState) -> Dict[str, Any]:
    """
    [역할]
    - raw_model_output을 JSON으로 파싱하여 parsed에 저장한다.
    - 파싱 실패 시 error 설정

    [출력 예]
    - parsed: {"name":"choose_move","arguments":{"move_name":"thunderbolt"}}
    """
    raw = state.get("raw_model_output", "")
    json_str = _extract_json_object(raw) or raw

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return {"error": "Model output JSON is not an object.", "parsed": None}
        return {"parsed": parsed}
    except Exception as e:
        return {"error": f"JSON parse failed: {e}. Raw: {raw[:200]}", "parsed": None}


def node_validate_and_finalize(state: BattleAgentState) -> Dict[str, Any]:
    """
    [역할]
    - parsed dict를 LLMAgentBase 규약 형태로 정규화한다.
    - name/arguments 존재 여부, 허용 함수 여부를 검증한다.

    여기서는 "가용 move/switch인지"까지는 검증하지 않는다.
    왜냐하면 실제 가용성 검증은 LLMAgentBase.choose_move()가
    battle.available_moves / available_switches 기준으로 수행하고,
    실패 시 랜덤 폴백까지 담당하기 때문이다.

    즉, 이 노드는:
    - 형식 검증(함수명/arguments key)까지만 책임진다.
    """
    if state.get("error"):
        # 이전 노드에서 이미 실패한 경우 그대로 종료
        return {}

    parsed = state.get("parsed")
    if not parsed:
        return {"error": "No parsed decision."}

    name = parsed.get("name")
    arguments = parsed.get("arguments")

    if name not in ("choose_move", "choose_switch"):
        return {"error": f"Invalid action name: {name}"}

    if not isinstance(arguments, dict):
        return {"error": "arguments must be an object."}

    if name == "choose_move":
        if "move_name" not in arguments or not isinstance(arguments["move_name"], str) or not arguments["move_name"].strip():
            return {"error": "choose_move requires non-empty 'move_name'."}

    if name == "choose_switch":
        if "pokemon_name" not in arguments or not isinstance(arguments["pokemon_name"], str) or not arguments["pokemon_name"].strip():
            return {"error": "choose_switch requires non-empty 'pokemon_name'."}

    # LLMAgentBase가 원하는 규약으로 래핑
    return {"decision": {"name": name, "arguments": arguments}}


# 5) LangGraph 구성 및 컴파일
def build_langgraph_qwen_agent():
    """
    LangGraph 실행 그래프 정의

    흐름:
    START
      -> prepare_prompts
      -> call_llm
      -> parse
      -> validate_finalize
      -> END
    """
    g = StateGraph(BattleAgentState)

    g.add_node("prepare_prompts", node_prepare_prompts)
    g.add_node("call_llm", node_call_llm)
    g.add_node("parse", node_parse_decision)
    g.add_node("validate_finalize", node_validate_and_finalize)

    g.add_edge(START, "prepare_prompts")
    g.add_edge("prepare_prompts", "call_llm")
    g.add_edge("call_llm", "parse")
    g.add_edge("parse", "validate_finalize")
    g.add_edge("validate_finalize", END)

    return g.compile()


LANGGRAPH_QWEN_AGENT = build_langgraph_qwen_agent()


# 6) LLMAgentBase 하위 클래스로 연결
class QwenLangGraphAgent(LLMAgentBase):
    """
    poke-env가 매 턴 호출하는 choose_move()는 LLMAgentBase에 이미 구현되어 있다.
    우리는 _get_llm_decision()만 LangGraph로 구현하면 된다.

    즉, "전투 의사결정 로직"은 LangGraph로 옮기되,
    "검증/폴백/실제 create_order 반환"은 기존 LLMAgentBase 로직을 그대로 사용한다.
    """

    def __init__(self, avatar: str = "qwen", *args, **kwargs):
        kwargs["avatar"] = avatar
        kwargs["start_timer_on_battle_start"] = True
        super().__init__(*args, **kwargs)

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        """
        LLMAgentBase.choose_move()가 호출하는 추상 메서드 구현

        입력: battle_state(문자열)
        출력: {"decision": ...} 또는 {"error": ...}

        LangGraph를 "invoke"해서 결과 state를 받고,
        그 안의 decision/error를 꺼내 반환한다.
        """
        final_state = await LANGGRAPH_QWEN_AGENT.ainvoke(
            {"battle_state": battle_state}
        )

        # 최종 state에서 decision 우선 반환
        if final_state.get("decision"):
            return {"decision": final_state["decision"]}

        # decision이 없으면 error 반환
        if final_state.get("error"):
            return {"error": final_state["error"]}

        # 둘 다 없으면 방어적으로 에러 처리
        return {"error": "LangGraph produced neither decision nor error."}


"""
=========================================================
사용 예시(개념)
---------------------------------------------------------
from poke_env.player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle

agent = QwenLangGraphAgent(
    battle_format="gen9randombattle",
    username="mybot",
    password="...",
)

# poke-env 루프가 매 턴 agent.choose_move() 호출
=========================================================

적용 팁
---------------------------------------------------------
1) QWEN_REPO_ID 환경변수로 모델 레포를 명확히 지정
   export QWEN_REPO_ID="Qwen/Qwen3-4B-Instruct"

2) HF Inference Endpoint를 쓰는 경우 HuggingFaceEndpoint 설정을
   endpoint_url 기반으로 바꾸는 편이 더 안정적이다.
   (서버리스는 가끔 속도/쿼터/모델 로딩 이슈가 있다)

3) JSON 파싱 안정화가 점수/승률에 직결된다.
   - _extract_json_object() 로 복구
   - arguments 누락 시 에러 처리
   - name 오탈자 방지 위해 system prompt를 강하게 유지

4) 승률을 올리고 싶으면 LangGraph 노드를 추가:
   - "상대 타입 기반 간이 점수화 노드"
   - "스위치 우선순위 노드"
   - "상태(예: 최근 3턴 행동) 메모리 노드"
=========================================================
"""
```


참고자료
Huggingface, agents course, https://huggingface.co/learn