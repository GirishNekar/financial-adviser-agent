Set-Location $PSScriptRoot
$env:Path = "$env:ProgramFiles\Git\cmd;$env:Path"
$env:SNOWFLAKE_TOKEN_FILE_PATH = "$env:USERPROFILE\.snowflake\fa_agent_token"
$env:SNOWFLAKE_TOKEN_TYPE = 'PROGRAMMATIC_ACCESS_TOKEN'
& .\.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py
