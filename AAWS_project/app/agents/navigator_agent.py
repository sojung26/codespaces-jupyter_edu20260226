from datetime import date
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# Tools 제공 모듈에서 필요한 도구들을 임포트합니다.
from app.tools import tools_navigator

# 오늘 날짜
today_date = date.today().strftime("%Y-%m-%d")

# System Prompt
system_prompt = f"""
당신은 사용자의 명령에 따라 웹을 탐색하고, 페이지의 맥락을 기억하며 연속적인 탐색을 수행할 수 있는 'Navigator' 에이전트입니다.

[역할 및 지침]
1. 주어진 사용자 요청을 분석하여 read_image_and_analyze, browse_web_keep_alive 또는 web_search_custom_tool 중 적절한 도구를 사용하세요.
2. 가능한 경우 브라우저 내부 상태(현재 URL, 로그인 등)를 유지하며 연속 탐색으로 답을 찾아야 합니다.
3. 도구를 호출할 때는 브라우저 에이전트가 정확히 이해할 수 있도록 구체적인 행동 지시(instruction)를 작성하세요.
4. 수집한 결과는 간결하고 사용자가 이해하기 쉽게 요약하여 제공하세요.

오늘의 날짜: {today_date}
"""


def get_agent_executor():
    # LLM 초기화
    llm = init_chat_model(model="gpt-4o", model_provider="openai")

    # 간단한 메모리 세이버 (대화 컨텍스트 저장)
    memory = MemorySaver()

    # Navigator 전용 도구 목록 (app.tools에서 import된 tools_navigator 사용)
    navigator_agent = create_agent(
        model=llm,
        tools=tools_navigator,
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    return navigator_agent


agent_executor = get_agent_executor()
