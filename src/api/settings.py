import os
from os.path import join
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from functools import lru_cache
from api.config import UPLOAD_FOLDER_NAME

root_dir = os.path.dirname(os.path.abspath(__file__))
env_path = join(root_dir, ".env.aws")

env_file_path = join(root_dir, ".env")
if os.path.exists(env_file_path):
    load_dotenv(env_file_path, override=False)

if os.path.exists(env_path):
    print(f"Loading environment variables from {env_path}")
    # Ensure deployment-specific values override placeholder local defaults.
    load_dotenv(env_path, override=True)


class Settings(BaseSettings):
    google_client_id: str | None = None

    google_application_credentials: str | None = None
    bq_project_name: str | None = None
    bq_dataset_name: str | None = None

    openai_api_key: str | None = None
    s3_bucket_name: str | None = None  # only relevant when running the code remotely
    s3_folder_name: str | None = None  # only relevant when running the code remotely
    local_upload_folder: str = (
        UPLOAD_FOLDER_NAME  # hardcoded variable for local file storage
    )
    sentry_dsn: str | None = None
    sentry_environment: str | None = "development"
    env: str | None = None
    slack_user_signup_webhook_url: str | None = None
    slack_course_created_webhook_url: str | None = None
    slack_usage_stats_webhook_url: str | None = None
    slack_alert_webhook_url: str | None = None
    model_config = SettingsConfigDict(env_file=join(root_dir, ".env"), extra="ignore")


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()

if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
