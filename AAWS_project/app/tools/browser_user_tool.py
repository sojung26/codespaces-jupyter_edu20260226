from browser_use import Browser, Agent, ChatGoogle
from langchain_core.tools import tool
import os

# DISPLAY 환경변수 확인
print(f"✅ DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")

try:
    # 💡 핵심: 세션을 유지하는 공유 브라우저 인스턴스를 하나만 생성합니다.
    print("🚀 브라우저 시작 중...")
    shared_browser = Browser(
        headless=False,
        disable_security=True,
        window_size={'width': 1280, 'height': 720},
        keep_alive=True  # 도구 호출이 끝나도 브라우저가 종료되지 않습니다.
    )
    print("✅ 브라우저 생성 성공!")
except Exception as e:
    print(f"❌ 브라우저 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    raise

@tool
async def browse_web_keep_alive(instruction: str) -> str:
    """
    공유된 브라우저 세션을 사용하여 웹 탐색을 수행하고 결과를 반환합니다.
    여러 턴의 도구 호출에서 브라우저 상태(현재 페이지, 로그인 상태 등)를 유지합니다.
    
    Args:
        instruction: 브라우저가 수행해야 할 구체적인 행동 지시문 (예: '현재 페이지에서 두 번째 링크 클릭해')
    """
    print(f"\n🌐 [Browser Tool - Keep Alive] 행동 개시: {instruction}")
    
    bu_llm = ChatGoogle(model="gemini-flash-latest")
    #bu_llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    
    # 공유 브라우저를 전달해서 세션을 유지합니다.
    agent = Agent(task=instruction, llm=bu_llm, browser=shared_browser)
    history = await agent.run(max_steps=10)
    
    result_text = history.final_result()
    if not result_text:
        return "브라우저 조작을 시도했으나 명확한 결과를 얻지 못했습니다. 다른 명령으로 재시도해보세요."
    
    # 현재 페이지 URL 정보 추가 (맥락 유지용)
    last_url = None
    try:
        if hasattr(history, "urls"):
            urls_list = history.urls() or []
            if urls_list:
                last_url = urls_list[-1]
    except Exception:
        last_url = None
    
    if not last_url:
        try:
            all_results = getattr(history, "all_results", None) or getattr(history, "results", None) or []
            for item in reversed(all_results):
                text_candidates = []
                if hasattr(item, "long_term_memory") and item.long_term_memory:
                    text_candidates.append(item.long_term_memory)
                if hasattr(item, "extracted_content") and item.extracted_content:
                    text_candidates.append(item.extracted_content)
                if hasattr(item, "extracted_content") and isinstance(item.extracted_content, list):
                    text_candidates.extend(item.extracted_content)
                for t in text_candidates:
                    if isinstance(t, str) and ("http://" in t or "https://" in t):
                        import re
                        m = re.search(r"https?://[^\s,'\)\]]+", t)
                        if m:
                            last_url = m.group(0)
                            break
                if last_url:
                    break
        except Exception:
            last_url = None

    url_info = f"현재 위치: {last_url}" if last_url else "현재 위치: (URL 확인 불가)"
    
    return f"{url_info}\n결과: {result_text}"
