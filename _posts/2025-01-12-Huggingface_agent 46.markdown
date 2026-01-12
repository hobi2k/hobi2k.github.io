---
layout: post
title:  "허깅페이스 에이전트 코스 - Build Your Own Pokémon Battle Agent"
date:   2025-01-12 00:10:22 +0900
categories: Huggingface_agent
---

# Build Your Own Pokémon Battle Agent

이 글에서는 **Agentic AI를 실제 게임 환경에 적용하는 실습**으로,  
LLM 기반 에이전트가 **포켓몬 턴제 배틀**을 수행하도록 만드는 전체 구조를 다룬다.

핵심 목표는 다음과 같다.

- LLM이 **게임 상태를 이해**
- 가능한 행동(기술 사용 / 교체)을 **도구 호출 형태로 선택**
- 실제 온라인 배틀 환경에서 **자율적으로 턴을 진행**


## 전체 맥락: 이 코드가 하는 일

이 시스템은 Pokémon Showdown(대전 서버)에서 한 턴씩 들어오는 `Battle` 상태를  
1) 사람이 읽을 수 있는 문자열로 “요약”해서 LLM에 주고  
2) LLM이 **함수 호출 형식**으로 `choose_move` 또는 `choose_switch`를 선택하도록 강제하고  
3) 그 결과를 실제 `poke-env`의 `create_order()`로 바꿔 서버에 전송한다.

핵심은 **LLM이 자연어로 “때릴게요”라고 말하는 게 아니라, 반드시 정해진 도구를 호출하게 한다**는 점이다.

## 코드

```python
# STANDARD_TOOL_SCHEMA
# [역할]
# - LLM이 사용할 수 있는 “행동(action)”의 스펙(= 함수 호출 규약)을 정의한다.
# - LLM은 자연어로 행동을 설명하는 대신, 아래 중 하나를 "function call"로 선택해야 한다.
#
# [왜 필요한가]
# - Pokémon 배틀은 매 턴 "기술 선택" 또는 "교체" 같은 명령이 정확한 포맷으로 들어가야 한다.
# - LLM의 자유로운 문장 생성은 오동작 확률이 높다(오타, 여러 문장, 설명 포함 등).
# - 그래서 가능한 행동을 스키마로 제한해 에이전트가 "실행 가능한 출력"만 내도록 한다.
#
# [중요 포인트]
# - choose_move / choose_switch 키 이름과 function name은 반드시 일관되어야 한다.
# - arguments의 필드명(move_name, pokemon_name)이 어긋나면 호출 파싱에서 실패한다.
STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        ...
    },
    "choose_switch": {
        ...
    },
}


# LLMAgentBase
# [역할]
# - poke-env의 Player를 상속하여, "LLM 기반" 플레이어를 만들기 위한 공통 기반 클래스.
# - 이 클래스는:
#   (1) Battle 객체(복잡한 구조)를 LLM 입력으로 바꾸고
#   (2) LLM이 반환한 function call을 해석해
#   (3) poke-env가 이해하는 실제 명령(create_order)로 변환한다.
#
# [설계 의도]
# - “LLM 호출부(_get_llm_decision)”는 하위 클래스(TemplateAgent)가 구현한다.
# - 나머지(상태 포맷/결정 파싱/검증/폴백)는 공통 로직으로 재사용한다.
#
# [핵심 제약]
# - LLM이 내린 결정이 틀릴 수 있으므로:
#   - move_name이 실제 available_moves에 없으면 무조건 폴백
#   - pokemon_name이 실제 available_switches에 없으면 무조건 폴백
# - 폴백 없으면 게임이 멈추거나 invalid command로 패배/에러가 난다.
class LLMAgentBase(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # LLM에게 노출할 도구 스키마를 보관
        self.standard_tools = STANDARD_TOOL_SCHEMA

        # 배틀 히스토리(확장 포인트: 장기 전략/기억/요약)
        self.battle_history = []

  
    # _format_battle_state
  
    # [역할]
    # - Battle 객체는 LLM이 직접 이해하기 어려움 -> 텍스트 상태로 변환한다.
    # - “현재 내 포켓몬 / 상대 포켓몬 / 사용 가능한 기술 / 교체 후보 / 필드 상태”를 요약한다.
    #
    # [왜 이 포맷이 중요한가]
    # - LLM의 판단 품질은 입력 상태 요약 품질에 강하게 의존한다.
    # - 특히 move.id를 정확히 보여줘야 LLM이 "실행 가능한 선택"을 한다.
    #
    # [실전 팁]
    # - 최적화를 하려면:
    #   - 상대 포켓몬이 확정적으로 보이는 정보만 포함
    #   - 기술을 우선순위로 정렬(타입상성, BP 등)
    #   - 히스토리 요약 1~2줄 추가(직전 턴 내가 쓴 기술/상대가 쓴 기술)
  
    def _format_battle_state(self, battle: Battle) -> str:
        active_pkmn = battle.active_pokemon
        active_pkmn_info = f"Your active Pokemon: {active_pkmn.species} " \
                           f"(Type: {'/'.join(map(str, active_pkmn.types))}) " \
                           f"HP: {active_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"Status: {active_pkmn.status.name if active_pkmn.status else 'None'} " \
                           f"Boosts: {active_pkmn.boosts}"

        opponent_pkmn = battle.opponent_active_pokemon
        opp_info_str = "Unknown"
        if opponent_pkmn:
            opp_info_str = f"{opponent_pkmn.species} " \
                           f"(Type: {'/'.join(map(str, opponent_pkmn.types))}) " \
                           f"HP: {opponent_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"Status: {opponent_pkmn.status.name if opponent_pkmn.status else 'None'} " \
                           f"Boosts: {opponent_pkmn.boosts}"
        opponent_pkmn_info = f"Opponent's active Pokemon: {opp_info_str}"

        available_moves_info = "Available moves:\n"
        if battle.available_moves:
            available_moves_info += "\n".join(
                [f"- {move.id} (Type: {move.type}, BP: {move.base_power}, Acc: {move.accuracy}, "
                 f"PP: {move.current_pp}/{move.max_pp}, Cat: {move.category.name})"
                 for move in battle.available_moves]
            )
        else:
             available_moves_info += "- None (Must switch or Struggle)"

        available_switches_info = "Available switches:\n"
        if battle.available_switches:
              available_switches_info += "\n".join(
                  [f"- {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%, "
                   f"Status: {pkmn.status.name if pkmn.status else 'None'})"
                   for pkmn in battle.available_switches]
              )
        else:
            available_switches_info += "- None"

        state_str = f"{active_pkmn_info}\n" \
                    f"{opponent_pkmn_info}\n\n" \
                    f"{available_moves_info}\n\n" \
                    f"{available_switches_info}\n\n" \
                    f"Weather: {battle.weather}\n" \
                    f"Terrains: {battle.fields}\n" \
                    f"Your Side Conditions: {battle.side_conditions}\n" \
                    f"Opponent Side Conditions: {battle.opponent_side_conditions}"
        return state_str.strip()

  
    # _find_move_by_name
  
    # [역할]
    # - LLM이 선택한 move_name을 실제 Move 객체로 매핑한다.
    #
    # [우선순위]
    # 1) move.id 정확 매칭(가장 안전)
    # 2) move.name(디스플레이 이름) 매칭(덜 안전, 경고 출력)
    #
    # [왜 두 단계인가]
    # - LLM이 종종 "Flamethrower"처럼 표시명(name)을 내기도 하고,
    #   시스템은 "flamethrower" 같은 id를 요구할 수 있다.
    # - 2)단계를 둬서 약간의 회복탄력성을 확보한다.
  
    def _find_move_by_name(self, battle: Battle, move_name: str) -> Optional[Move]:
        normalized_name = normalize_name(move_name)
        for move in battle.available_moves:
            if move.id == normalized_name:
                return move
        for move in battle.available_moves:
            if move.name.lower() == move_name.lower():
                print(f"Warning: Matched move by display name '{move.name}' instead of ID '{move.id}'. "
                      f"Input was '{move_name}'.")
                return move
        return None

  
    # _find_pokemon_by_name
  
    # [역할]
    # - LLM이 선택한 pokemon_name을 실제 Pokemon 객체(교체 후보)로 매핑한다.
    #
    # [주의]
    # - 교체 가능한 후보(battle.available_switches) 안에서만 탐색한다.
    # - 이미 기절한 포켓몬, 필드에 없는 포켓몬을 LLM이 지목할 수 있으므로 검증이 필수.
  
    def _find_pokemon_by_name(self, battle: Battle, pokemon_name: str) -> Optional[Pokemon]:
        normalized_name = normalize_name(pokemon_name)
        for pkmn in battle.available_switches:
            if normalize_name(pkmn.species) == normalized_name:
                return pkmn
        return None

  
    # choose_move (핵심 턴 루프)
  
    # [역할]
    # - poke-env가 매 턴 호출하는 진입점.
    # - 여기서:
    #   1) battle_state_str 생성
    #   2) LLM에게 결정 요청(_get_llm_decision)
    #   3) function call 결과를 해석하여 실행(create_order)
    #   4) 실패 시 안전 폴백(랜덤 선택 또는 디폴트)
    #
    # [왜 폴백이 이토록 중요한가]
    # - LLM은 확률적이며, 가끔 잘못된 기술/교체 대상을 반환한다.
    # - 폴백이 없으면:
    #   - invalid command로 턴이 꼬이거나
    #   - 예외 발생으로 에이전트가 중단되고
    #   - 즉시 패배/탈락으로 이어질 수 있다.
  
    async def choose_move(self, battle: Battle) -> str:
        battle_state_str = self._format_battle_state(battle)

        # 하위 클래스(TemplateAgent 등)가 LLM 호출을 구현한다.
        decision_result = await self._get_llm_decision(battle_state_str)

        print(decision_result)

        decision = decision_result.get("decision")
        error_message = decision_result.get("error")

        action_taken = False
        fallback_reason = ""

        # 1) LLM이 function call을 제대로 줬을 때
        if decision:
            function_name = decision.get("name")
            args = decision.get("arguments", {})

            # (A) 기술 선택
            if function_name == "choose_move":
                move_name = args.get("move_name")
                if move_name:
                    chosen_move = self._find_move_by_name(battle, move_name)
                    if chosen_move and chosen_move in battle.available_moves:
                        action_taken = True
                        chat_msg = f"AI Decision: Using move '{chosen_move.id}'."
                        print(chat_msg)
                        return self.create_order(chosen_move)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid move '{move_name}'."
                else:
                    fallback_reason = "LLM 'choose_move' called without 'move_name'."

            # (B) 교체 선택
            elif function_name == "choose_switch":
                pokemon_name = args.get("pokemon_name")
                if pokemon_name:
                    chosen_switch = self._find_pokemon_by_name(battle, pokemon_name)
                    if chosen_switch and chosen_switch in battle.available_switches:
                        action_taken = True
                        chat_msg = f"AI Decision: Switching to '{chosen_switch.species}'."
                        print(chat_msg)
                        return self.create_order(chosen_switch)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid switch '{pokemon_name}'."
                else:
                    fallback_reason = "LLM 'choose_switch' called without 'pokemon_name'."

            # (C) 알 수 없는 함수 호출
            else:
                fallback_reason = f"LLM called unknown function '{function_name}'."

        # 2) 실패/오류/무응답일 때 폴백
        if not action_taken:
            if not fallback_reason:
                if error_message:
                    fallback_reason = f"API Error: {error_message}"
                elif decision is None:
                    fallback_reason = "LLM did not provide a valid function call."
                else:
                    fallback_reason = "Unknown error processing LLM decision."

            print(f"Warning: {fallback_reason} Choosing random action.")

            # 가능한 행동이 있으면 랜덤 선택(게임 지속 보장)
            if battle.available_moves or battle.available_switches:
                return self.choose_random_move(battle)
            else:
                # 정말 아무 것도 못하면 디폴트(보통 struggle)
                print("AI Fallback: No moves or switches available. Using Struggle/Default.")
                return self.choose_default_move(battle)

  
    # _get_llm_decision (추상 메서드)
  
    # [역할]
    # - 실제 LLM API 호출부를 하위 클래스에서 구현하도록 강제한다.
    # - 이렇게 분리하면:
    #   - OpenAI / Mistral / Gemini / 로컬 LLM 등 백엔드가 달라도
    #     나머지 게임 로직은 동일하게 재사용 가능하다.
  
    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _get_llm_decision")
```

## TemplateAgent(블루프린트)

여기서 제시된 템플릿은 “실행 불가한 설계 예시”다.
하지만 구조적으로는 핵심을 정확히 보여준다.

```python
class TemplateAgent(LLMAgentBase):
    """Uses Template AI API for decisions."""
    def __init__(self, api_key: str = None, model: str = "model-name", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 어떤 모델을 쓸지 지정(예: Qwen, Mistral, OpenAI 등)
        self.model = model

        # 실제 LLM 클라이언트(여기서는 가짜 Provider)
        self.template_client = TemplateModelProvider(api_key=...)

        # 표준 도구 스키마를 LLM 호출에 넘길 수 있게 리스트화
        self.template_tools = list(self.standard_tools.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        """Sends state to the LLM and gets back the function call decision."""
        system_prompt = (
            "You are a ..."
        )
        user_prompt = f"..."

        try:
            # 여기서 LLM을 호출하고,
            response = await self.template_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            # LLM이 function-call 형태로 무엇을 선택했는지 추출해서
            message = response.choices[0].message

            # LLMAgentBase가 이해하는 표준 형태로 반환한다.
            return {"decision": {"name": function_name, "arguments": arguments}}

        except Exception as e:
            print(f"Unexpected error during call: {e}")
            return {"error": f"Unexpected error: {e}"}
```

## 코드 밖 “추가 설명”

### 매 턴 실행 파이프라인

- poke-env가 choose_move(battle) 호출
- _format_battle_state(battle)로 LLM 입력 문자열 생성
- _get_llm_decision(battle_state_str)로 LLM에게 결정 요청
- LLM이 {name: "choose_move", arguments:{move_name:"..."}} 반환
- _find_move_by_name()로 실제 Move 객체 찾아 검증
- create_order(move)로 Showdown 서버에 명령 전송
- 실패 시 choose_random_move()로 폴백

### 왜 “Function Calling(툴 호출)”이 필수인가

- 게임은 “실행 가능한 명령”이 필요하다
- LLM 자연어는 실행 불가한 문자열(설명/잡담/두 문장 이상)을 섞기 쉽다
- 따라서:
  - 가능한 행동을 STANDARD_TOOL_SCHEMA로 제한
  - 결과를 엄격히 파싱/검증
  - 실패하면 폴백으로 게임을 지속

### 점수 올리기 위해 수정 필요한 구간

- _format_battle_state에 “최근 1~2턴 요약” 추가
  - 예: “Last turn: I used X, opponent used Y, damage was Z”

- available_moves를 단순 나열하지 말고 “추천 순위 힌트”를 넣기
  - 예: “Heuristic: prefer STAB, higher base power, super-effective”

- _get_llm_decision에서:
  - 시스템 프롬프트에 “절대 설명하지 말고 function call만” 강제
  - 온도 0으로 낮추고
  - 잘못된 호출이면 재시도(1회 정도) 후 폴백

## 결론

이 코드는 포켓몬용으로 보이지만, 실제로는 범용 Agent 패턴이다.

- 상태를 텍스트로 요약(Observation)
- 도구 호출로 행동(Action)
- 실패는 폴백으로 복원(Resilience)

즉, 게임이든 웹이든 RPA든, “Agent”의 골격은 동일하다.

참고자료
Huggingface, agents course, https://huggingface.co/learn