PROMPT_DIR = Path(__file__).resolve().parent

with open(PROMPT_DIR / "custom_system_prompt.md.j2", "r") as file:
    CUSTOM_SYSTEM_PROMPT = file.read()

 with open(PROMPT_DIR / "owm_tool_system_prompt.md.j2", "r") as file:
    OWM_TOOL_SYSTEM_PROMPT = file.read()

__all__ = ["CUSTOM_SYSTEM_PROMPT", "OWM_TOOL_SYSTEM_PROMPT"]

