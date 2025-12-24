import secrets
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.types import SecretStr

class AWSSettings(BaseSettings):
    """API key settings."""

    s3_bucket_name: str = Field(
        default="my-s3-bucket",
        description="S3 bucket name for file storage"
    )
    access_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="AWS access key"
    )
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="AWS secret key"
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region"
    )
    voice_call_folder: str = Field(
        default="documents",
        description="Folder for document uploads in S3"
    )

class NodoHistoryDbSettings(BaseSettings):
    """Nodo history db"""

    db_host: str = Field(
        default="my-db-host",
        description="DB host name"
    )
    db_port: str= Field(
        default="my-db-port",
        description="DB port"
    )
    db_name: str = Field(
        default="my-name-db",
        description="DB NAME"
    )
    db_user: str = Field(
        default="my-db-name",
        description="DB USERNAME"
    )
    db_password: str = Field(
        default="my-db-password",
        description="DB password"
    )
    vectordb: str = Field(
        default="my-vectordb",
        description="vectordb"
    )
    apartmentdb: str = Field(
        default="my-apaertmentdb",
        description="apartmentdb"
    )
    history: str = Field(
        default="conversation-history",
        description="nodo history api"
    )
    apiloginurl: str = Field(
        default="",
        description=""
    )
    apiloginpw: str = Field(
        default="",
        description=""
    )
    apiloginuser: str = Field(
        default="",
        description=""
    )

class CrmDbSettings(BaseSettings):
    """Nodo history db"""
    token: str = Field(
        default="ey12344567888819929192",
        description="API key of CRM"
    )
    url: str = Field(
        default="https://localhost/rest/",
        description="my crm url"
    )

    
class LLMSettings(BaseSettings):
    """API key settings."""

    base_url: str = Field(
        default="my-s3-bucket",
        description="S3 bucket name for file storage"
    )
    model: str = Field(
        default="my-llm-model",
        description="AWS access key"
    )


class Settings(BaseSettings):

    aws: AWSSettings = AWSSettings()
    dbnodo: NodoHistoryDbSettings = NodoHistoryDbSettings()
    crm: CrmDbSettings = CrmDbSettings()
    llm: LLMSettings = LLMSettings()
    # Database URL
    # @property
    # def ASYNC_DATABASE_URL(self) -> str:
    #     # URL encode the password to handle special characters
    #     encoded_password = urllib.parse.quote_plus(self.DB_PASSWORD)
    #     return f"postgresql+asyncpg://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # @property
    # def SYNC_DATABASE_URL(self) -> str:
    #     # URL encode the password to handle special characters
    #     encoded_password = urllib.parse.quote_plus(self.DB_PASSWORD)
    #     return f"postgresql://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", 
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings()