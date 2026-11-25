import os
from pathlib import Path
from pydantic import BaseModel


class AppConfig(BaseModel):
    app_name: str = "Intelligent Personal Finance Advisor Agent"
    app_version: str = "1.0.0"
    storage_dir: Path = Path("app/storage")
    reports_dir: Path = Path("app/storage/reports")
    logs_dir: Path = Path("app/logs")
    history_file: Path = Path("app/storage/conversation_history.json")
    # Snowflake Cortex OpenAI-compatible endpoint.
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://ww83019.ap-northeast-1.aws.snowflakecomputing.com/api/v2/cortex/v1",
    )
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    # Snowflake OAuth/PAT token; read from SNOWFLAKE_TOKEN or the token file on each LLM call.
    llm_token_file: Path = Path(os.getenv("SNOWFLAKE_TOKEN_FILE_PATH", "/snowflake/session/token"))
    llm_token_type: str = os.getenv("SNOWFLAKE_TOKEN_TYPE", "OAUTH")
    llm_timeout_seconds: int = 45
    llm_max_retries: int = 2
    llm_enabled: bool = True
    llm_strict_mode: bool = False


config = AppConfig()
