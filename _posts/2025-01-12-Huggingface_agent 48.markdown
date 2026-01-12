---
layout: post
title:  "허깅페이스 에이전트 코스 - Pokémon Battle LLM Agent 2"
date:   2025-01-11 00:10:22 +0900
categories: Huggingface_agent
---

# Pokémon Battle LLM Agent 2

이 글은 poke-env 기반 Pokémon Showdown Agent에서 서버 역할을 하는 코드를 정리한다.

## main.py

```python
"""
main.py
Pokemon Battle Livestream Server (FastAPI + WebSocket + poke_env Agents)

이 파일의 목적
- 여러 LLM 기반 포켓몬 배틀 에이전트를 “순환(cycle)”시키며,
- 특정 쇼다운 서버(Custom Showdown Server)에서 배틀을 자동으로 받게 하고,
- 웹 프론트(iframe)로 배틀 화면을 스트리밍하며,
- OBS 오버레이용으로 /last_action 엔드포인트에서 “마지막 행동 텍스트”를 제공한다.

전체 구성(큰 그림)
1) FastAPI 서버
   - GET "/" : 메인 화면 (WebSocket으로 HTML fragment를 받아서 표시)
   - WS  "/ws": 서버가 HTML fragment를 push (클라이언트는 거의 send 안 함)
   - GET "/last_action": OBS Browser Source용 “마지막 행동” 텍스트 페이지

2) Agent Lifecycle Manager (백그라운드 태스크)
   - 활성 에이전트가 없으면: 랜덤으로 에이전트를 하나 선택 → accept_challenges(1회) 실행
   - 배틀이 시작되면: battle_tag 감지 → iframe 표시
   - 배틀이 끝나면: 에이전트/태스크 정리 → 다음 에이전트로 교체
   - private/suffixed battle로 보이면: 즉시 forfeit 후 교체(서버 정책/보안/공개 배틀 제약 대응)

3) ConnectionManager
   - 현재 표시해야 할 “HTML fragment”를 상태로 보관
   - WebSocket 연결된 모든 클라이언트에게 fragment를 브로드캐스트

운영 상 유의사항
- 환경변수로 각 에이전트 계정 비밀번호가 반드시 있어야 한다.
- poke_env Player는 내부에 _battles 딕셔너리를 유지하며, battle_tag가 “즉시” 준비되지 않을 수 있다.
- 비동기(AsyncIO) 환경에서 전역 상태(active_agent_instance 등) 관리가 핵심이며,
  race condition을 줄이기 위해 “전역 상태를 먼저 clear”하는 패턴을 사용한다.
"""

import asyncio
import os
import random
import time
import traceback
import logging
from typing import List, Dict, Optional, Set
import html  # ADDED FOR /last_action (HTML escaping)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Imports for poke_env and agents
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration
# from poke_env.environment.battle import Battle

# Import the actual agent classes
from agents import OpenAIAgent, GeminiAgent, MistralAgent

# Configuration
"""
[Configuration 섹션]
- CUSTOM_SERVER_URL / CUSTOM_ACTION_URL:
  참가자들이 공통으로 사용하는 “특정 쇼다운 서버” 엔드포인트.
  ServerConfiguration에 주입되어 poke_env Player가 이 서버로 접속한다.

- CUSTOM_BATTLE_VIEW_URL_TEMPLATE:
  battle_id를 붙여서 웹에서 배틀 화면을 띄울 때 사용 가능한 템플릿(여기선 참고용).
  실제 iframe은 create_battle_iframe()에서 custom testclient URL을 사용한다.

- DEFAULT_BATTLE_FORMAT:
  랜덤배틀 룰(gen9randombattle). poke_env의 battle_format으로 들어간다.

- LAST_ACTION_FILE:
  OBS overlay로 쓰는 /last_action에서 읽어오는 파일.
  “에이전트가 직전에 어떤 행동(기술/교체)을 했는지”를 별도 프로세스나 에이전트 코드에서
  이 파일에 기록한다고 가정한다(이 파일에는 기록 로직이 없다. 읽기/표시만 한다).
"""
CUSTOM_SERVER_URL = "wss://jofthomas.com/showdown/websocket"
CUSTOM_ACTION_URL = 'https://play.pokemonshowdown.com/action.php?'
CUSTOM_BATTLE_VIEW_URL_TEMPLATE = "https://jofthomas.com/play.pokemonshowdown.com/testclient.html#{battle_id}"
custom_config = ServerConfiguration(CUSTOM_SERVER_URL, CUSTOM_ACTION_URL)
DEFAULT_BATTLE_FORMAT = "gen9randombattle"
LAST_ACTION_FILE = "last_action.txt"  # --- ADDED FOR /last_action --- (Filename)

# Define available agents with their corresponding classes
"""
[AGENT_CONFIGS]
- key: “에이전트 계정 username”
- value:
  - class: 실제 Player(LLMAgentBase) 서브클래스
  - password_env_var: 이 에이전트 계정 비밀번호를 담는 환경변수 이름

이 구조의 장점
- 에이전트 타입을 추가/삭제할 때 설정 한 곳만 수정하면 된다.
- lifecycle manager가 공통 로직으로 어떤 에이전트든 동일하게 다룰 수 있다.
"""
AGENT_CONFIGS = {
    "OpenAIAgent": {"class": OpenAIAgent, "password_env_var": "OPENAI_AGENT_PASSWORD"},
    "GeminiAgent": {"class": GeminiAgent, "password_env_var": "GEMINI_AGENT_PASSWORD"},
    "MistralAgent": {"class": MistralAgent, "password_env_var": "MISTRAL_AGENT_PASSWORD"},
}

# Filter out agents with missing passwords
"""
[AVAILABLE_AGENT_NAMES]
- AGENT_CONFIGS 중에서, password_env_var 환경변수가 실제로 설정된 에이전트만 남긴다.
- 즉, 비밀번호 없는 에이전트는 런타임에서 사용하지 않는다.

이 처리가 중요한 이유
- poke_env Player는 서버 로그인에 계정 비밀번호가 필요
- 비밀번호 없는 계정을 활성화하면 즉시 예외/실패가 나므로 사전 필터링이 안전하다.
"""
AVAILABLE_AGENT_NAMES = [
    name for name, cfg in AGENT_CONFIGS.items()
    if os.environ.get(cfg.get("password_env_var", ""))
]

if not AVAILABLE_AGENT_NAMES:
    print("FATAL ERROR: No agent configurations have their required password environment variables set. Exiting.")
    exit(1)

# --- Global State Variables ---
"""
[전역 상태(서버 런타임 FSM 상태)]
- active_agent_name:
  현재 “대기/배틀” 상태인 에이전트의 이름(AGENT_CONFIGS 키)

- active_agent_instance:
  실제 poke_env Player 인스턴스. accept_challenges/forfeit/disconnect 등을 수행한다.

- active_agent_task:
  agent.accept_challenges(None, 1) 를 돌리는 백그라운드 태스크 핸들.
  배틀이 시작되면 더 이상 챌린지를 받지 않도록 cancel한다.

- current_battle_instance:
  현재 진행 중인 Battle 객체(또는 그에 준하는 포인터).
  battle_tag 기반으로 추적하며, finished 여부를 주기적으로 확인한다.

- background_task_handle:
  manage_agent_lifecycle() 를 도는 “메인 FSM 루프” 태스크 핸들.
"""
active_agent_name: Optional[str] = None
active_agent_instance: Optional[Player] = None
active_agent_task: Optional[asyncio.Task] = None
current_battle_instance = None
background_task_handle: Optional[asyncio.Task] = None

# --- Create FastAPI app ---
app = FastAPI(title="Pokemon Battle Livestream")

# --- Helper Functions ---
def get_active_battle(agent: Player):
    """
    [역할]
    - 특정 agent(Player)가 현재 갖고 있는 battle들 중,
      finished=False 인 “진행 중 배틀”을 하나 찾아서 반환한다.

    [왜 battle_tag 체크가 까다로운가]
    - poke_env 내부에서 Battle 객체는 먼저 생기고, battle_tag가 나중에 세팅되는 경우가 있다.
    - 이 코드에서는 battle_tag가 “battle-”로 시작하는 정상 포맷인지까지 확인한다.
      (battle_tag 미완성/비정상 포맷인 경우 iframe 표시 등 UI 동작이 꼬일 수 있어 방어)

    [반환]
    - 조건을 만족하는 Battle 객체 1개 또는 None
    """
    if agent and agent._battles:
        active_battles = [b for b in agent._battles.values() if not b.finished]
        if active_battles:
            # Ensure the battle object has a battle_tag before returning
            if hasattr(active_battles[0], 'battle_tag') and active_battles[0].battle_tag:
                # Check if the battle_tag has the expected format (starts with 'battle-')
                if active_battles[0].battle_tag.startswith("battle-"):
                    return active_battles[0]
                else:
                    # This handles cases where the battle object might exist but tag isn't ready
                    # print(f"DEBUG: Found active battle for {agent.username} but tag '{active_battles[0].battle_tag}' not ready.")
                    return None
            else:
                # print(f"DEBUG: Found active battle for {agent.username} but it has no battle_tag attribute yet.")
                return None
    return None


def create_battle_iframe(battle_id: str) -> str:
    """
    [역할]
    - 현재 배틀을 브라우저에 표시하기 위한 iframe HTML fragment를 생성한다.
    - “전체 HTML 페이지”가 아니라 iframe 태그만 반환한다.
      (프론트는 WebSocket으로 fragment를 받고 stream-container에 innerHTML로 주입한다.)

    [URL 선택]
    - 기본 공식 클라이언트 URL이 아니라, custom testclient URL을 쓰고 있다.
    - battle_id는 보통 battle_tag 그대로("battle-gen9randombattle-...") 형태가 들어온다.
    """
    print("Creating iframe content for battle ID: ", battle_id)
    # Use the official client URL unless you specifically need the test client
    # battle_url = f"https://play.pokemonshowdown.com/{battle_id}"
    battle_url = f"https://jofthomas.com/play.pokemonshowdown.com/testclient.html#{battle_id}"  # Using your custom URL

    # Return ONLY the iframe tag with a class for styling
    return f"""
    <iframe
        id="battle-iframe"
        class="battle-iframe"
        src="{battle_url}"
        allowfullscreen
    ></iframe>
    """


def create_idle_html(status_message: str, instruction: str) -> str:
    """
    [역할]
    - 배틀이 없을 때(대기/초기화/전환 등) 보여주는 “idle 화면” HTML fragment를 생성한다.
    - status_message와 instruction을 UI에 표시.

    [특징]
    - 배경 이미지는 메인 HTML의 CSS(.idle-container)에서 지정된다.
    - 여기서는 content div만 반환한다(전체 HTML 문서 아님).
    """
    # Returns ONLY the content div, not the full HTML page
    return f"""
    <div class="content-container idle-container">
        <div class="message-box">
            <p class="status">{status_message}</p>
            <p class="instruction">{instruction}</p>
        </div>
    </div>
    """


def create_error_html(error_msg: str) -> str:
    """
    [역할]
    - 에러 발생 시 UI에 표시할 “error 화면” HTML fragment를 생성한다.
    """
    # Returns ONLY the content div, not the full HTML page
    return f"""
    <div class="content-container error-container">
        <div class="message-box">
            <p class="status">🚨 Error 🚨</p>
            <p class="instruction">{error_msg}</p>
        </div>
    </div>
    """


async def update_display_html(new_html_fragment: str) -> None:
    """
    [역할]
    - 현재 표시할 HTML fragment를 갱신하고,
    - 연결된 모든 웹소켓 클라이언트에게 브로드캐스트한다.

    [흐름]
    - manager.update_all() 호출 → manager가 current_html_fragment를 갱신 → send_text()로 push
    """
    # Pass the fragment directly
    await manager.update_all(new_html_fragment)
    print("HTML Display FRAGMENT UPDATED and broadcasted.")


# --- Agent Lifecycle Management ---
async def select_and_activate_new_agent():
    """
    [역할]
    - AVAILABLE_AGENT_NAMES 중 랜덤으로 하나 선택
    - 해당 에이전트(Player) 인스턴스를 생성
    - accept_challenges(None, 1) 태스크를 시작(“딱 1번” 챌린지 받기)
    - 전역 상태(active_agent_*)를 설정
    - UI를 “challenge 요청 대기 화면”으로 갱신

    [핵심 설계]
    - accept_challenges를 “태스크로 따로” 돌리는 이유:
      서버는 FastAPI 이벤트 루프를 유지해야 하며, lifecycle 루프와 독립적으로 챌린지를 받아야 함.
    """
    global active_agent_name, active_agent_instance, active_agent_task

    if not AVAILABLE_AGENT_NAMES:
        print("Lifecycle: No available agents with passwords set.")
        await update_display_html(create_error_html("No agents available. Check server logs/environment variables."))
        return False

    selected_name = random.choice(AVAILABLE_AGENT_NAMES)
    config = AGENT_CONFIGS[selected_name]
    AgentClass = config["class"]
    password_env_var = config["password_env_var"]
    agent_password = os.environ.get(password_env_var)
    print(f"Lifecycle: Activating agent '{selected_name}'...")
    # Use HTML tags for slight emphasis if desired
    await update_display_html(create_idle_html("Selecting Next Agent...", f"Preparing <strong>{selected_name}</strong>..."))

    try:
        account_config = AccountConfiguration(selected_name, agent_password)
        agent = AgentClass(
            account_configuration=account_config,
            server_configuration=custom_config,
            battle_format=DEFAULT_BATTLE_FORMAT,
            log_level=logging.INFO,
            max_concurrent_battles=1
        )

        # Start the task to accept exactly one battle challenge
        # Setting name for easier debugging
        task = asyncio.create_task(agent.accept_challenges(None, 1), name=f"AcceptChallenge_{selected_name}")
        task.add_done_callback(log_task_exception)  # Add callback for errors

        # Update global state
        active_agent_name = selected_name
        active_agent_instance = agent
        active_agent_task = task

        print(f"Lifecycle: Agent '{selected_name}' is active and listening for 1 challenge.")
        # Use HTML tags for slight emphasis
        await update_display_html(create_idle_html(f"Agent Ready: <strong>{selected_name}</strong>",
                                      f"Please challenge <strong>{selected_name}</strong> to a <strong>{DEFAULT_BATTLE_FORMAT}</strong> battle."))
        return True

    except Exception as e:
        error_msg = f"Failed to activate agent '{selected_name}': {e}"
        print(error_msg)
        traceback.print_exc()
        await update_display_html(create_error_html(f"Error activating {selected_name}. Please wait or check logs."))

        # Clear state if activation failed
        active_agent_name = None
        active_agent_instance = None
        active_agent_task = None
        return False


async def check_for_new_battle():
    """
    [역할]
    - active_agent_instance가 현재 “배틀을 시작했는지” 확인한다.
    - get_active_battle()로 battle을 찾고 battle_tag가 준비되었으면 current_battle_instance에 저장.
    - 배틀이 시작되었으면 accept_challenges 태스크를 cancel해서 추가 챌린지 수락을 중단.

    [왜 cancel이 필요한가]
    - max_concurrent_battles=1이라도, accept_challenges는 다음 챌린지를 계속 기다릴 수 있다.
    - 정책상 “1배틀이 시작되면 더 이상 챌린지를 받지 않도록” 명시적으로 끊어준다.
    """
    # --- FIX: Declare intention to use/modify global variables ---
    global active_agent_instance, current_battle_instance, active_agent_name, active_agent_task
    # -------------------------------------------------------------

    if active_agent_instance:
        battle = get_active_battle(active_agent_instance)
        # Check if battle exists AND has a valid battle_tag
        if battle and battle.battle_tag:
            # This line MODIFIES the global variable
            current_battle_instance = battle
            print(f"Lifecycle: Agent '{active_agent_name}' started battle: {battle.battle_tag}")

            # Stop the agent from listening for more challenges once a battle starts
            if active_agent_task and not active_agent_task.done():
                print(f"Lifecycle: Cancelling accept_challenges task for {active_agent_name} as battle started.")
                active_agent_task.cancel()
                # Optional: Wait briefly for cancellation confirmation, but don't block excessively
                # try:
                #     await asyncio.wait_for(active_agent_task, timeout=0.5)
                # except (asyncio.CancelledError, asyncio.TimeoutError):
                #     pass # Expected outcomes
        # else:
            # print(f"DEBUG: get_active_battle returned None or battle without tag.")


async def deactivate_current_agent(reason: str = "cycle"):
    """
    [역할]
    - 현재 active agent 및 관련 task를 안전하게 종료/정리하고,
    - 전역 상태(active_agent_*, current_battle_instance)를 리셋한다.

    [왜 “전역 상태를 먼저 clear” 하는가]
    - manage_agent_lifecycle()가 주기적으로 전역 상태를 검사한다.
    - cleanup 중에 전역을 그대로 두면, lifecycle이 “아직 살아있는 agent”로 착각하고
      _battles 접근/forfeit/disconnect 등으로 race condition을 유발할 수 있다.
    - 그래서 먼저 active_*를 None으로 만들고, 그 뒤에 task cancel / disconnect를 수행한다.

    [reason]
    - "battle_end" / "cycle" / "forfeited_private_battle" / 기타 에러 상황 등을 UI에 반영.
    """
    global active_agent_name, active_agent_instance, active_agent_task, current_battle_instance

    agent_name_to_deactivate = active_agent_name  # Store before clearing
    print(f"Lifecycle: Deactivating agent '{agent_name_to_deactivate}' (Reason: {reason})...")

    # Display appropriate intermediate message
    if reason == "battle_end":
        await update_display_html(create_idle_html("Battle Finished!", f"Agent <strong>{agent_name_to_deactivate}</strong> completed the match."))
    elif reason == "cycle":
        await update_display_html(create_idle_html("Cycling Agents", f"Switching from <strong>{agent_name_to_deactivate}</strong>..."))
    elif reason == "forfeited_private_battle":
        await update_display_html(create_idle_html("Switching Agent", f"Agent <strong>{agent_name_to_deactivate}</strong> forfeited a private battle."))
    else:  # Generic reason or error
        await update_display_html(create_idle_html(f"Resetting Agent ({reason})", f"Cleaning up <strong>{agent_name_to_deactivate}</strong>..."))

    # Give users a moment to see the intermediate message
    await asyncio.sleep(3)  # Adjust duration as needed

    # Show the "preparing next agent" message before lengthy cleanup
    await update_display_html(create_idle_html("Preparing Next Agent...", "Please wait..."))

    agent = active_agent_instance
    task = active_agent_task

    # Store a local copy of the battle instance before clearing it
    # last_battle_instance = current_battle_instance # Not strictly needed now

    # --- Crucial: Clear global state variables FIRST ---
    # This prevents race conditions where the lifecycle loop might try to
    # access the agent while it's being deactivated.
    active_agent_name = None
    active_agent_instance = None
    active_agent_task = None
    current_battle_instance = None
    print(f"Lifecycle: Global state cleared for '{agent_name_to_deactivate}'.")

    # --- Now perform cleanup actions ---
    # Cancel the accept_challenges task if it's still running (it might already be done/cancelled)
    if task and not task.done():
        print(f"Lifecycle: Ensuring task cancellation for {agent_name_to_deactivate} ({task.get_name()})...")
        task.cancel()
        try:
            # Wait briefly for the task to acknowledge cancellation
            await asyncio.wait_for(task, timeout=2.0)
            print(f"Lifecycle: Task cancellation confirmed for {agent_name_to_deactivate}.")
        except asyncio.CancelledError:
            print(f"Lifecycle: Task cancellation confirmation (CancelledError) for {agent_name_to_deactivate}.")
        except asyncio.TimeoutError:
            print(f"Lifecycle: Task did not confirm cancellation within timeout for {agent_name_to_deactivate}.")
        except Exception as e:
            # Catch other potential errors during task cleanup
            print(f"Lifecycle: Error during task cancellation wait for {agent_name_to_deactivate}: {e}")

    # Disconnect the player (ensure agent object exists)
    if agent:
        print(f"Lifecycle: Disconnecting player {agent.username}...")
        try:
            # Check websocket state before attempting disconnection
            if hasattr(agent, '_websocket') and agent._websocket and agent._websocket.open:
                await agent.disconnect()
                print(f"Lifecycle: Player {agent.username} disconnected successfully.")
            else:
                print(f"Lifecycle: Player {agent.username} already disconnected or websocket not available.")
        except Exception as e:
            # Log errors during disconnection but don't halt the process
            print(f"ERROR during agent disconnect ({agent.username}): {e}")
            traceback.print_exc()  # Log full traceback for debugging

    # Add a brief delay AFTER deactivation before the loop potentially selects a new agent
    await asyncio.sleep(2)  # Reduced from 3, adjust as needed
    print(f"Lifecycle: Agent '{agent_name_to_deactivate}' deactivation complete.")


async def manage_agent_lifecycle():
    """
    [역할: 서버의 메인 FSM(상태 머신) 루프]
    - 이 루프가 “에이전트 선택 → 챌린지 대기 → 배틀 감지 → 배틀 모니터링 → 종료/교체”를 반복한다.
    - FastAPI startup 이벤트에서 백그라운드 태스크로 실행된다.

    [상태 정의]
    - State 1: active_agent_instance is None
      → 새 에이전트를 활성화(select_and_activate_new_agent)
    - State 2: active_agent_instance is not None
      → 2a: current_battle_instance is None : 배틀 시작 감지(check_for_new_battle)
      → 2b: current_battle_instance is not None : 배틀 종료 여부 감시

    [private battle 포기 로직]
    - battle_tag를 split('-') 했을 때 특정 패턴(“suffix format”)이면 private로 간주하고 즉시 forfeit.
      이건 서버 운영 정책(공개 배틀만 스트리밍) 또는 iframe 표시/접근 이슈를 회피하려는 휴리스틱.
    """
    # --- FIX: Declare intention to use global variables ---
    global active_agent_name, active_agent_instance, active_agent_task, current_battle_instance
    # ------------------------------------------------------

    print("Background lifecycle manager started.")
    REFRESH_INTERVAL_SECONDS = 3  # How often to check state when idle/in battle
    LOOP_COOLDOWN_SECONDS = 1  # Small delay at end of loop if no other waits occurred
    ERROR_RETRY_DELAY_SECONDS = 10  # Longer delay after errors
    POST_BATTLE_DELAY_SECONDS = 5  # Delay after a battle finishes before selecting next agent

    loop_counter = 0

    while True:
        loop_counter += 1
        loop_start_time = time.monotonic()
        print(f"\n--- Lifecycle Check #{loop_counter} [{time.strftime('%H:%M:%S')}] ---")

        try:
            # ==================================
            # State 1: No agent active
            # ==================================
            # Now Python knows active_agent_instance refers to the global one
            if active_agent_instance is None:
                print(f"[{loop_counter}] State 1: No active agent. Selecting...")
                activated = await select_and_activate_new_agent()
                if not activated:
                    print(f"[{loop_counter}] State 1: Activation failed. Waiting {ERROR_RETRY_DELAY_SECONDS}s before retry.")
                    await asyncio.sleep(ERROR_RETRY_DELAY_SECONDS)
                else:
                    # Now Python knows active_agent_name refers to the global one set by select_and_activate_new_agent
                    print(f"[{loop_counter}] State 1: Agent '{active_agent_name}' activated successfully.")
                    # No sleep here, proceed to next check immediately if needed

            # ==================================
            # State 2: Agent is active
            # ==================================
            else:
                # Now Python knows active_agent_name refers to the global one
                agent_name = active_agent_name  # Cache for logging
                print(f"[{loop_counter}] State 2: Agent '{agent_name}' is active.")

                # --- Sub-state: Check for new battle if none is tracked ---
                # Now Python knows current_battle_instance refers to the global one
                if current_battle_instance is None:
                    print(f"[{loop_counter}] State 2a: Checking for new battle for '{agent_name}'...")
                    await check_for_new_battle()  # This updates global current_battle_instance if found

                    # Now Python knows current_battle_instance refers to the global one
                    if current_battle_instance:
                        battle_tag = current_battle_instance.battle_tag
                        print(f"[{loop_counter}] State 2a: *** NEW BATTLE DETECTED: {battle_tag} for '{agent_name}' ***")

                        # Check for non-public/suffixed format (heuristic: more than 3 parts, 3rd part is number)
                        parts = battle_tag.split('-')
                        is_suffixed_format = len(parts) > 3 and parts[2].isdigit()

                        if is_suffixed_format:
                            # Forfeit immediately if it looks like a private/suffixed battle ID
                            print(f"[{loop_counter}] Detected potentially non-public battle format ({battle_tag}). Forfeiting.")
                            # Don't update display yet, do it before deactivation
                            try:
                                # Now Python knows active_agent_instance refers to the global one
                                if active_agent_instance:  # Ensure agent still exists
                                    await active_agent_instance.forfeit(battle_tag)
                                    # await active_agent_instance.send_message("/forfeit", battle_tag) # Alternative
                                    print(f"[{loop_counter}] Sent forfeit command for {battle_tag}.")
                                    await asyncio.sleep(1.5)  # Give forfeit time to register
                            except Exception as forfeit_err:
                                print(f"[{loop_counter}] ERROR sending forfeit for {battle_tag}: {forfeit_err}")
                            # Deactivate agent after forfeit attempt
                            await deactivate_current_agent(reason="forfeited_private_battle")
                            continue  # Skip rest of the loop for this iteration

                        else:
                            # Public battle format - display the iframe
                            print(f"[{loop_counter}] Public battle format detected. Displaying battle {battle_tag}.")
                            await update_display_html(create_battle_iframe(battle_tag))
                            # Now fall through to monitor this battle in the next section

                    else:
                        # No new battle found, agent remains idle
                        print(f"[{loop_counter}] State 2a: No new battle found. Agent '{agent_name}' remains idle, waiting for challenge.")
                        # Periodically refresh idle screen to ensure consistency
                        idle_html = create_idle_html(f"Agent Ready: <strong>{agent_name}</strong>",
                                                     f"Please challenge <strong>{agent_name}</strong> to a <strong>{DEFAULT_BATTLE_FORMAT}</strong> battle.")
                        await update_display_html(idle_html)
                        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)  # Wait before next check if idle

                # --- Sub-state: Monitor ongoing battle ---
                # Now Python knows current_battle_instance refers to the global one
                if current_battle_instance is not None:
                    battle_tag = current_battle_instance.battle_tag
                    print(f"[{loop_counter}] State 2b: Monitoring battle {battle_tag} for '{agent_name}'")

                    # Ensure agent instance still exists before accessing its battles
                    # Now Python knows active_agent_instance refers to the global one
                    if not active_agent_instance:
                        print(f"[{loop_counter}] WARNING: Agent instance for '{agent_name}' disappeared while monitoring battle {battle_tag}! Deactivating.")
                        await deactivate_current_agent(reason="agent_disappeared_mid_battle")
                        continue

                    # Get potentially updated battle object directly from agent's state
                    # Use .get() for safety
                    # Now Python knows active_agent_instance refers to the global one
                    battle_obj = active_agent_instance._battles.get(battle_tag)

                    if battle_obj and battle_obj.finished:
                        print(f"[{loop_counter}] Battle {battle_tag} is FINISHED. Deactivating agent '{agent_name}'.")
                        await deactivate_current_agent(reason="battle_end")
                        print(f"[{loop_counter}] Waiting {POST_BATTLE_DELAY_SECONDS}s post-battle before selecting next agent.")
                        await asyncio.sleep(POST_BATTLE_DELAY_SECONDS)
                        continue  # Start next loop iteration to select new agent

                    elif not battle_obj:
                        # This can happen briefly during transitions or if battle ends unexpectedly
                        print(f"[{loop_counter}] WARNING: Battle object for {battle_tag} not found in agent's list for '{agent_name}'. Battle might have ended abruptly. Deactivating.")
                        await deactivate_current_agent(reason="battle_object_missing")
                        continue

                    else:
                        # Battle is ongoing, battle object exists, iframe should be displayed
                        print(f"[{loop_counter}] Battle {battle_tag} ongoing for '{agent_name}'.")
                        # Optionally: Could re-send iframe HTML periodically if needed, but usually not necessary
                        # await update_display_html(create_battle_iframe(battle_tag))
                        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)  # Wait before next check

        # --- Global Exception Handling for the main loop ---
        except asyncio.CancelledError:
            print("Lifecycle manager task cancelled.")
            raise  # Re-raise to ensure proper shutdown
        except Exception as e:
            print(f"!!! ERROR in main lifecycle loop #{loop_counter}: {e} !!!")
            traceback.print_exc()
            # Now Python knows active_agent_name refers to the global one
            current_agent_name = active_agent_name  # Cache name before deactivation attempts
            # Now Python knows active_agent_instance refers to the global one
            if active_agent_instance:
                print(f"Attempting to deactivate agent '{current_agent_name}' due to loop error...")
                try:
                    await deactivate_current_agent(reason="main_loop_error")
                except Exception as deactivation_err:
                    print(f"Error during error-handling deactivation: {deactivation_err}")
                    # Ensure state is cleared even if deactivation fails partially
                    active_agent_name = None
                    active_agent_instance = None
                    active_agent_task = None
                    current_battle_instance = None
            else:
                # Error happened potentially before agent activation or after clean deactivation
                print("No active agent instance during loop error.")
                # Show a generic error on the frontend
                await update_display_html(create_error_html(f"A server error occurred in the lifecycle manager. Please wait. ({e})"))

            # Wait longer after a major error before trying again
            print(f"Waiting {ERROR_RETRY_DELAY_SECONDS}s after loop error.")
            await asyncio.sleep(ERROR_RETRY_DELAY_SECONDS)
            continue  # Go to next loop iteration after error handling

        # --- Delay at end of loop if no other significant waits happened ---
        elapsed_time = time.monotonic() - loop_start_time
        if elapsed_time < LOOP_COOLDOWN_SECONDS:
            await asyncio.sleep(LOOP_COOLDOWN_SECONDS - elapsed_time)


def log_task_exception(task: asyncio.Task):
    """
    [역할]
    - asyncio.create_task(...)로 만든 백그라운드 태스크가 예외로 죽는 경우를 놓치지 않기 위한 콜백.
    - task.add_done_callback(log_task_exception) 로 등록한다.

    [왜 필요한가]
    - 백그라운드 태스크 예외는 await하지 않으면 로그에 조용히 묻힐 수 있다.
    - accept_challenges 같은 태스크가 죽었는데도 서버는 계속 도는 상황을 방지하기 위해 로그를 남긴다.
    """
    try:
        if task.cancelled():
            # Don't log cancellation as an error, it's often expected
            print(f"Task '{task.get_name()}' was cancelled.")
            return
        # Accessing result will raise exception if task failed
        task.result()
        print(f"Task '{task.get_name()}' completed successfully.")
    except asyncio.CancelledError:
        print(f"Task '{task.get_name()}' confirmed cancelled (exception caught).")
        pass  # Expected
    except Exception as e:
        # Log actual errors
        print(f"!!! Exception in background task '{task.get_name()}': {e} !!!")
        traceback.print_exc()
        # Optionally: Trigger some recovery or notification here if needed


# --- WebSocket connection manager ---
class ConnectionManager:
    """
    [역할]
    - WebSocket 클라이언트 연결을 관리한다.
    - “현재 화면에 표시할 HTML fragment”를 상태로 유지한다.
    - update_all() 호출 시, 모든 클라이언트에 fragment를 send_text로 push한다.

    [설계 포인트]
    - current_html_fragment를 캐싱해서, 같은 fragment면 브로드캐스트를 스킵(불필요한 트래픽/깜박임 방지)
    - active_connections는 Set[WebSocket]으로 관리한다.
    """
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # Initialize with the idle HTML fragment
        self.current_html_fragment: str = create_idle_html("Initializing...", "Setting up Pokémon Battle Stream")

    async def connect(self, websocket: WebSocket):
        """
        [역할]
        - 신규 WebSocket 연결 수락
        - active_connections에 등록
        - 현재 fragment를 즉시 1회 전송(클라이언트 초기 렌더)
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Client connected. Sending current state. Total clients: {len(self.active_connections)}")
        # Send current state (HTML fragment) to newly connected client
        try:
            await websocket.send_text(self.current_html_fragment)
        except Exception as e:
            print(f"Error sending initial state to new client: {e}")
            # Consider removing the connection if initial send fails
            await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket):
        """
        [역할]
        - WebSocket 연결을 active_connections에서 제거
        - discard() 사용으로 “없어도 안전”하게 처리
        """
        # Use discard() to safely remove even if not present
        self.active_connections.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def update_all(self, html_fragment: str):
        """
        [역할]
        - current_html_fragment 갱신
        - 모든 클라이언트에게 브로드캐스트

        [주의: 구현상 잠재 버그]
        - 아래 코드에서 results를 순회하며 connection = list(self.active_connections)[i] 로 매칭하는데,
          set은 순서가 보장되지 않는다.
          그리고 위에서 send_tasks를 만들 때도 list(self.active_connections)를 새로 만든다.
          결과적으로 results 인덱스와 connection이 1:1로 맞지 않을 수 있다.

        [실무적 개선]
        - send_tasks 만들 때 (connection, task) 쌍을 유지하거나,
          connections_snapshot = list(self.active_connections) 를 한 번만 만들고,
          그 snapshot을 send_tasks/결과 매칭에 같이 써야 안정적이다.
        """
        """Update the current HTML fragment and broadcast to all clients."""
        if self.current_html_fragment == html_fragment:
            # print("Skipping broadcast, HTML fragment unchanged.")
            return  # Avoid unnecessary updates if content is identical

        self.current_html_fragment = html_fragment
        if not self.active_connections:
            # print("No active connections to broadcast update to.")
            return

        print(f"Broadcasting update to {len(self.active_connections)} clients...")

        # Create a list of tasks to send updates concurrently
        # Make a copy of the set for safe iteration during potential disconnects
        send_tasks = [
            connection.send_text(html_fragment)
            for connection in list(self.active_connections)  # Iterate over a copy
        ]

        # Use asyncio.gather to send to all clients, collecting results/exceptions
        results = await asyncio.gather(*send_tasks, return_exceptions=True)

        # Handle potential errors during broadcast (e.g., client disconnected abruptly)
        # Iterate over connections again, checking results
        connections_to_remove = set()
        for i, result in enumerate(results):
            connection = list(self.active_connections)[i]  # Assumes order is maintained
            if isinstance(result, Exception):
                print(f"Error sending update to client: {result}. Marking for removal.")
                connections_to_remove.add(connection)

        # Disconnect clients that failed
        for connection in connections_to_remove:
            await self.disconnect(connection)


manager = ConnectionManager()

# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """
    [역할]
    - 단일 HTML 문서를 반환한다.
    - 이 문서는 WebSocket에 연결한 후, 서버가 보내주는 HTML fragment를
      #stream-container에 innerHTML로 삽입해서 화면을 구성한다.

    [프론트 동작 요약]
    - connectWebSocket() 실행
    - ws.onmessage: streamContainer.innerHTML = event.data
      즉, 서버가 보내는 fragment가 곧 “전체 화면”이다.
    - ws.onclose: 재연결 시도(5초 후)
    """
    # NOTE: Ensure the static path '/static/pokemon_huggingface.png' is correct
    # and the image exists in a 'static' folder next to your main.py
    # (Existing HTML content remains unchanged)
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pokemon Battle Livestream</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Press+Start+2P&display=swap" rel="stylesheet">
        <style>
            /* Basic Reset */
            * {
                box-sizing: border-box;
            }
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
                overflow: hidden; /* Prevent scrollbars on body */
                font-family: 'Poppins', sans-serif; /* Default font */
                color: #ffffff; /* Default text color */
                background-color: #1a1a1a; /* Dark background */
            }
            /* Container for dynamic content */
            #stream-container {
                position: fixed; /* Use fixed to ensure it covers viewport */
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                display: flex; /* Use flexbox for centering content */
                justify-content: center;
                align-items: center;
            }
            /* Iframe Styling */
            .battle-iframe {
                width: 100%;
                height: 100%;
                border: none; /* Remove default border */
                display: block; /* Prevents potential extra space below iframe */
            }
            /* Base Content Container Styling (used by idle/error) */
            .content-container {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 20px;
                text-align: center;
            }
            /* Idle Screen Specific Styling */
            .idle-container {
                 /* Ensure the background covers the entire container */
                background-image: url('/static/pokemon_huggingface.png');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            /* Error Screen Specific Styling */
            .error-container {
                background: linear-gradient(135deg, #4d0000, #1a0000); /* Dark red gradient */
            }
            /* Message Box Styling (shared by idle/error) */
            .message-box {
                background-color: rgba(0, 0, 0, 0.75); /* Darker, more opaque */
                padding: 40px 50px; /* More padding */
                border-radius: 20px; /* More rounded */
                max-width: 70%; /* Max width */
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5); /* Softer shadow */
                border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
            }
            .status {
                font-family: 'Press Start 2P', cursive; /* Pixel font for status */
                font-size: clamp(1.5em, 4vw, 2.5em); /* Responsive font size */
                margin-bottom: 25px;
                color: #ffcb05; /* Pokemon Yellow */
                text-shadow: 3px 3px 0px #3b4cca; /* Pokemon Blue shadow */
                /* Subtle pulse animation for idle status */
                animation: pulse 2s infinite ease-in-out;
            }
            .instruction {
                font-size: clamp(1em, 2.5vw, 1.4em); /* Responsive font size */
                color: #f0f0f0; /* Light grey for readability */
                line-height: 1.6;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
            }
             .instruction strong {
                 color: #ff7f0f; /* A contrasting color like orange */
                 font-weight: 700; /* Ensure Poppins bold is used */
             }
            /* Error Screen Specific Text Styling */
            .error-container .status {
                color: #ff4d4d; /* Bright Red for error status */
                text-shadow: 2px 2px 0px #800000; /* Darker red shadow */
                animation: none; /* No pulse on error */
            }
             .error-container .instruction {
                color: #ffdddd; /* Lighter red for error details */
            }
            /* Pulse Animation */
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.03); }
                100% { transform: scale(1); }
            }
        </style>
    </head>
    <body>
        <div id="stream-container">
             </div>
        <script>
            const streamContainer = document.getElementById('stream-container');
            let ws = null; // WebSocket instance
            function connectWebSocket() {
                // Use wss:// for https:// and ws:// for http://
                const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
                const wsUrl = `${wsProtocol}://${location.host}/ws`;
                ws = new WebSocket(wsUrl);
                console.log('Attempting to connect to WebSocket server...');
                ws.onopen = (event) => {
                    console.log('WebSocket connection established.');
                    // Optional: Clear any 'connecting...' message if you have one
                    // streamContainer.innerHTML = ''; // Clear container only if needed
                };
                ws.onmessage = (event) => {
                    // console.log('Received update from server:', event.data);
                    // Directly set the innerHTML with the fragment received from the server
                    streamContainer.innerHTML = event.data;
                };
                ws.onclose = (event) => {
                    console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason}. Attempting to reconnect in 5 seconds...`);
                    ws = null; // Clear the instance
                    // Clear the display or show a 'disconnected' message
                    streamContainer.innerHTML = createReconnectMessage();
                    setTimeout(connectWebSocket, 5000); // Retry connection after 5 seconds
                };
                ws.onerror = (event) => {
                    console.error('WebSocket error:', event);
                    // The onclose event will usually fire after an error,
                    // so reconnection logic is handled there.
                    // You might want to display an error message here briefly.
                    streamContainer.innerHTML = createErrorMessage("WebSocket connection error. Attempting to reconnect...");
                    // Optionally force close to trigger reconnect logic if onclose doesn't fire
                    if (ws && ws.readyState !== WebSocket.CLOSED) {
                        ws.close();
                    }
                };
            }
             // Helper function to generate reconnecting message HTML (matches error style)
            function createReconnectMessage() {
                 return `
                    <div class="content-container error-container" style="background: #333;">
                        <div class="message-box" style="background-color: rgba(0,0,0,0.6);">
                            <p class="status" style="color: #ffcb05; text-shadow: none; animation: none;">🔌 Disconnected 🔌</p>
                            <p class="instruction" style="color: #eee;">Connection lost. Attempting to reconnect automatically...</p>
                        </div>
                    </div>`;
            }
             // Helper function to generate error message HTML
            function createErrorMessage(message) {
                return `
                    <div class="content-container error-container">
                        <div class="message-box">
                            <p class="status">🚨 Error 🚨</p>
                            <p class="instruction">${message}</p>
                        </div>
                    </div>`;
            }
            // Initial connection attempt when the page loads
            connectWebSocket();
        </script>
    </body>
    </html>
    """


@app.get("/last_action", response_class=HTMLResponse)
async def get_last_action():
    """
    [역할]
    - OBS Browser Source에 넣기 좋은 “텍스트 오버레이 전용 HTML”을 반환한다.
    - LAST_ACTION_FILE 내용을 매 요청마다 읽어서 표시한다.

    [보안]
    - html.escape()로 파일 내용을 이스케이프한다(XSS 방지).
      즉, last_action.txt에 악성 HTML/JS가 들어가도 실행되지 않고 텍스트로 보인다.

    [운영]
    - 파일이 없으면 에러 메시지를 표시한다(서버 로그에도 남김).
    """
    """
    Serves a simple HTML page displaying the content of last_action.txt,
    styled for OBS integration.
    """
    file_content_raw = ""
    error_message = None
    try:
        # Read the file content fresh on each request
        with open(LAST_ACTION_FILE, "r", encoding="utf-8") as f:
            file_content_raw = f.read()
    except FileNotFoundError:
        error_message = f"Error: File '{LAST_ACTION_FILE}' not found."
        print(f"WARN: {error_message}")  # Log server-side
    except Exception as e:
        error_message = f"An unexpected error occurred while reading '{LAST_ACTION_FILE}': {e}"
        print(f"ERROR: {error_message}")  # Log server-side
        traceback.print_exc()  # Log full traceback for debugging

    # Escape the raw content to prevent XSS if the file contains HTML/JS
    display_content = html.escape(file_content_raw) if not error_message else error_message
    # Use a class to differentiate normal content from error messages for styling
    content_class = "error" if error_message else "log-content"

    # Create the simple HTML response with updated styles for OBS
    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Last Action Log</title>
        <style>
            /* Import fonts to potentially match the main stream interface */
            /* You already import Poppins and Press Start 2P in the main '/' route's HTML */
            /* No extra import needed here if loaded by the browser already, but safe to include */
             @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Press+Start+2P&display=swap');
            /* Basic Reset */
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            /* Ensure html and body take full height/width and hide overflow */
            html, body {{
                height: 100%;
                width: 100%;
                overflow: hidden; /* Crucial for OBS Browser Source */
            }}
            body {{
                /* Use a font that matches your stream's text */
                font-family: 'Poppins', sans-serif;
                /* Fallback font */
                /* font-family: Consolas, 'Courier New', monospace; */
                line-height: 1.5; /* Adjust line spacing if needed */
                padding: 15px; /* Add some padding inside the OBS box */
                /* Make background transparent for seamless overlay in OBS */
                background-color: transparent;
                /* Set default text color to white for dark OBS backgrounds */
                color: #FFFFFF;
                /* Center the text block */
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}
            /* Wrapper for content, useful for centering and potential future overflow */
            .content-wrapper {{
                max-width: 100%;
                max-height: 100%;
                /* Optional: Add scroll if content exceeds OBS source size */
                /* overflow-y: auto; */
            }}
            /* Style for the main action text */
            .log-content {{
                 /* Significantly larger font size */
                font-size: 2em; /* Example: Adjust as needed */
                /* Alternative responsive size based on OBS source width: */
                /* font-size: calc(12px + 1.8vw); */
                white-space: pre-wrap; /* Allow text wrapping */
                word-wrap: break-word; /* Break long words */
                color: #EAEAEA; /* Slightly off-white */
                /* Optional: Add subtle shadow for better readability on complex backgrounds */
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
            }}
            /* Style for error messages */
            .error {{
                font-family: 'Poppins', sans-serif; /* Keep font consistent */
                font-size: 1.6em; /* Make errors large but slightly smaller than normal */
                color: #FFBDBD; /* Light red text, visible on dark */
                font-weight: bold;
                background-color: rgba(100, 0, 0, 0.7); /* Dark red, semi-transparent */
                border: 1px solid #FF8080; /* Lighter red border */
                padding: 10px 15px;
                border-radius: 8px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
                white-space: normal; /* Allow errors to wrap normally */
            }}
        </style>
    </head>
    <body>
        <div class="content-wrapper">
             <div class="{content_class}">{display_content}</div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_output)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    [역할]
    - 프론트가 연결하는 WebSocket 엔드포인트.
    - 연결 시 manager.connect()로 등록하고 current_html_fragment를 즉시 전송한다.
    - 현재 설계에서는 클라이언트가 서버로 보내는 메시지는 “없다”가 전제지만,
      코드상 receive_text()를 계속 기다리므로 클라이언트가 아무 것도 안 보내면 여기서 block된다.

    [중요한 점]
    - 이 구현은 “서버→클라 push”가 주 목적이므로,
      실무에서는 receive loop 대신 ping/pong 혹은 단순 sleep loop로 keepalive하는 형태가 흔하다.
    - 다만 FastAPI WebSocket이 내부 ping/pong을 처리하는 경우도 있고,
      여기서는 “혹시라도 클라가 메시지를 보내면 로깅”하려는 형태로 작성되어 있다.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive. Client doesn't send messages in this setup.
            # FastAPI's WebSocket implementation handles ping/pong internally usually.
            # If needed, you could implement explicit keepalive here.
            data = await websocket.receive_text()
            # We don't expect messages from the client in this design,
            # but log if received for debugging.
            print(f"Received unexpected message from client: {data}")
            # Or simply keep listening:
            # await asyncio.sleep(60) # Example keepalive interval if needed
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: Code {e.code}, Reason: {getattr(e, 'reason', 'N/A')}")
        await manager.disconnect(websocket)  # Use await here
    except Exception as e:
        # Catch other potential errors on the connection
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        await manager.disconnect(websocket)  # Ensure disconnect on error


@app.on_event("startup")
async def startup_event():
    """
    [역할]
    - FastAPI 앱이 시작될 때:
      1) static 디렉토리 생성 및 mount
      2) last_action.txt 없으면 기본 파일 생성
      3) manage_agent_lifecycle() 백그라운드 태스크 시작

    [왜 startup에서 lifecycle을 돌리나]
    - uvicorn 실행과 동시에 agent lifecycle을 자동 가동하기 위함.
    - 외부에서 별도 CLI로 lifecycle을 실행할 필요가 없도록 구성.
    """
    """Start background tasks when the application starts."""
    global background_task_handle

    # Mount static files directory (make sure 'static' folder exists)
    # Place your 'pokemon_huggingface.png' inside this 'static' folder
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        print(f"Created static directory at: {os.path.abspath(static_dir)}")
        print("!!! Please add 'pokemon_huggingface.png' to this directory! !!!")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"Mounted static directory '{static_dir}' at '/static'")

    # --- ADDED FOR /last_action --- Check if last_action.txt exists ---
    if not os.path.exists(LAST_ACTION_FILE):
        print(f"WARN: '{LAST_ACTION_FILE}' not found. Creating an empty file.")
        try:
            with open(LAST_ACTION_FILE, "w", encoding="utf-8") as f:
                f.write("No actions recorded yet.")
        except Exception as e:
            print(f"ERROR: Could not create '{LAST_ACTION_FILE}': {e}")
    # --- END ADDED SECTION ---

    print("🚀 Starting background tasks")
    # Start the main lifecycle manager task
    background_task_handle = asyncio.create_task(manage_agent_lifecycle(), name="LifecycleManager")
    # Add the exception logging callback
    background_task_handle.add_done_callback(log_task_exception)
    print("✅ Background tasks started")


@app.on_event("shutdown")
async def shutdown_event():
    """
    [역할]
    - 서버 종료 시 리소스 정리:
      1) lifecycle manager 태스크 cancel
      2) 활성 agent disconnect
      3) 모든 클라이언트 WebSocket 연결 close

    [왜 정리가 필요한가]
    - poke_env / websocket 연결이 열린 상태로 남으면 서버가 종료되지 않거나,
      다음 실행에서 “이미 로그인된 세션” 같은 충돌이 날 수 있다.
    """
    """Clean up tasks when shutting down."""
    global background_task_handle, active_agent_instance

    print("\n🔌 Shutting down application. Cleaning up...")

    # 1. Cancel the main lifecycle manager task
    if background_task_handle and not background_task_handle.done():
        print("Cancelling background task...")
        background_task_handle.cancel()
        try:
            await asyncio.wait_for(background_task_handle, timeout=5.0)
            print("Background task cancelled successfully.")
        except asyncio.CancelledError:
            print("Background task cancellation confirmed (CancelledError).")
        except asyncio.TimeoutError:
            print("Background task did not finish cancelling within timeout.")
        except Exception as e:
            print(f"Error during background task cancellation: {e}")

    # 2. Deactivate and disconnect any currently active agent
    #    Use a copy of the instance in case it gets cleared elsewhere during shutdown.
    agent_to_disconnect = active_agent_instance
    if agent_to_disconnect:
        agent_name = agent_to_disconnect.username if hasattr(agent_to_disconnect, 'username') else 'Unknown Agent'
        print(f"Disconnecting active agent '{agent_name}'...")
        try:
            # Check websocket status before disconnecting
            if hasattr(agent_to_disconnect, '_websocket') and agent_to_disconnect._websocket and agent_to_disconnect._websocket.open:
                await agent_to_disconnect.disconnect()
                print(f"Agent '{agent_name}' disconnected.")
            else:
                print(f"Agent '{agent_name}' already disconnected or websocket not available.")
        except Exception as e:
            print(f"Error during agent disconnect on shutdown for '{agent_name}': {e}")

    # 3. Close all active WebSocket connections cleanly
    print(f"Closing {len(manager.active_connections)} client WebSocket connections...")
    # Create tasks to close all connections concurrently
    close_tasks = [
        conn.close(code=1000, reason="Server shutting down")  # 1000 = Normal Closure
        for conn in list(manager.active_connections)  # Iterate over a copy
    ]
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)  # Log potential errors during close

    print("✅ Cleanup complete. Application shutdown.")


# For direct script execution
if __name__ == "__main__":
    """
    [역할]
    - python main.py 로 직접 실행할 때 uvicorn을 구동한다.
    - 환경변수 점검 및 로그 설정을 수행한다.

    [운영 팁]
    - reload=False는 운영/안정용. 개발 중이면 reload=True를 쓰기도 한다.
    - loggers 레벨을 조정해서 poke_env/websockets 로그 소음을 줄인다.
    """
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Reduce noise from poke_env's default INFO logging if desired
    logging.getLogger('poke_env').setLevel(logging.WARNING)
    logging.getLogger('websockets.client').setLevel(logging.INFO)  # Show websocket connection attempts

    print("Starting Pokemon Battle Livestream Server...")
    print("="*60)

    if not AVAILABLE_AGENT_NAMES:
        print("█████████████████████ FATAL ERROR █████████████████████")
        print(" No agents found with configured passwords!")
        print(" Please set the required environment variables:")
        for name, cfg in AGENT_CONFIGS.items():
            print(f"  - {cfg.get('password_env_var', 'N/A')} (for agent: {name})")
        print("="*60)
        exit("Exiting due to missing agent passwords.")
    else:
        print("✨ Available Agents Found:")
        for name in AVAILABLE_AGENT_NAMES:
            print(f"  - {name}")
    print("="*60)
    print(f"Server will run on http://0.0.0.0:7860")
    print(f"Last action log available at http://0.0.0.0:7860/last_action")  # --- ADDED INFO ---
    print("="*60)

    # Run with uvicorn
    uvicorn.run(
        "main:app",  # Point to the FastAPI app instance
        host="0.0.0.0",
        port=7860,
        reload=False,  # Disable reload for production/stable testing
        log_level="info"  # Uvicorn's log level
    )
```


참고자료
Huggingface, agents course, https://huggingface.co/learn