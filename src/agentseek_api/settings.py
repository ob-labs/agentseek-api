from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_API_PORT = 2024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    APP_NAME: str = "AgentSeek API"
    PORT: int = DEFAULT_API_PORT
    METADATA_DB_URL: str | None = None
    METADATA_DB_BACKEND: str = "auto"
    SEEKDB_URL: str = "mysql+aiomysql://root%40test:@localhost:2881/seekdb"
    EXECUTOR_BACKEND: str = "inline"
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    REDIS_RUN_QUEUE_KEY: str = "agentseek:runs:pending"
    REDIS_RUN_PROCESSING_KEY: str = "agentseek:runs:processing"
    REDIS_WORKER_POLL_TIMEOUT_SECONDS: int = 1
    REDIS_WORKER_LOCK_KEY: str = "agentseek:worker:active"
    REDIS_WORKER_LOCK_TTL_SECONDS: int = 30
    AGENTSEEK_GRAPHS: str | None = None
    AUTH_TYPE: str = "noop"
    AUTH_MODULE_PATH: str | None = None
    AUTH_API_KEYS: str | None = None
    AUTH_JWT_SECRET: str | None = None
    AUTH_JWT_ALGORITHM: str = "HS256"

    OCEANBASE_HOST: str = "localhost"
    OCEANBASE_PORT: str = "2881"
    OCEANBASE_USER: str = "root@test"
    OCEANBASE_PASSWORD: str = ""
    OCEANBASE_DB_NAME: str = "seekdb"


settings = Settings()
