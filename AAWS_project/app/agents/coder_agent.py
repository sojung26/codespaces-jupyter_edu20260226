import os
from datetime import date
from dataclasses import dataclass
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware

from app.tools.coder_tool import execute_python_code

# 오늘 날짜
today_date = date.today().strftime("%Y-%m-%d")

# Coder Context
@dataclass
class CoderContext:
    pass

# System Prompt
CODER_SYSTEM_PROMPT = f"""
당신은 파이썬 코드를 작성, 분석, 실행할 수 있는 '시니어 파이썬 개발자'입니다.

[역할 및 지침]
1. 목적에 맞게 도구를 명확히 구분해서 사용하세요:
   - 파일 내용을 읽거나 검색할 때: 파이썬 코드를 작성하지 말고, 반드시 내장된 파일 검색 도구를 우선적으로 사용하세요.
   - 새로운 로직이나 파이썬 스크립트를 작성하여 테스트할 때: 코드를 작성한 후 `execute_python_code` 도구를 사용하세요.
2. 코드를 작성했다면 반드시 `execute_python_code`를 실행하여 결과를 검증하세요.
3. 실행 로그에 에러(Error)가 발생하면, 즉시 에러 사유를 파악하고 코드를 수정한 뒤 다시 실행(디버깅)하세요. 에러 없이 성공할 때까지 스스로 반복해야 합니다.
4. 모든 작업이 완료되면, 최종적으로 해결된 방법과 결과를 사용자에게 짧고 명확하게 요약해 주세요.

오늘의 날짜: {today_date}
"""

def get_agent_executor():
    # LLM 초기화
    coder_model = init_chat_model("google_genai:gemini-flash-latest", temperature=0.2)
    
    # 메모리 저장소
    memory = MemorySaver()
    
    # Artifact 디렉토리 경로
    artifact_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "code_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Coder 에이전트 생성
    coder_agent = create_agent(
        model=coder_model,
        system_prompt=CODER_SYSTEM_PROMPT,
        context_schema=CoderContext,
        tools=[execute_python_code],
        middleware=[
            FilesystemFileSearchMiddleware(
                root_path=artifact_dir,  # ✅ 검색 범위를 code_artifacts로 강제
                use_ripgrep=True,
                max_file_size_mb=10,
            )
        ],
        checkpointer=memory
    )
    
    return coder_agent

agent_executor = get_agent_executor()
