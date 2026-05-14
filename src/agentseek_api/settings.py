from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    APP_NAME: str = "AgentSeek API"
    PORT: int = 2026
    METADATA_DB_URL: str | None = None
    METADATA_DB_BACKEND: str = "auto"
    SEEKDB_URL: str = "mysql+aiomysql://root%40test:@localhost:2881/seekdb"
    AGENTSEEK_GRAPHS: str | None = None
    AUTH_TYPE: str = "noop"
    AUTH_MODULE_PATH: str | None = None

    OCEANBASE_HOST: str = "localhost"
    OCEANBASE_PORT: str = "2881"
    OCEANBASE_USER: str = "root@test"
    OCEANBASE_PASSWORD: str = ""
    OCEANBASE_DB_NAME: str = "seekdb"


settings = Settings()
