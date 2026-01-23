from __future__ import annotations

from pathlib import Path


def load_prompt(prompt_filename: str, encoding: str = "utf-8") -> str:
    """
    agents/prompts/ 아래의 프롬프트 파일(.md)을 읽어 문자열로 반환합니다.

    사용 예:
        prompt = load_prompt("safety_notice.prompt.md")
        prompt = load_prompt("explain_topk.prompt.md")

    - prompt_filename: 파일명만 전달(권장). 예: "safety_notice.prompt.md"
    - encoding: 기본 utf-8
    """
    # 이 loader.py 파일이 있는 폴더 = agents/prompts
    prompts_dir = Path(__file__).resolve().parent
    path = prompts_dir / prompt_filename

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding=encoding)
