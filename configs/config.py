from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent

@dataclass
class DataConfig:
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"

@dataclass
class DatabaseConfig:
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "fraudshield"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "mac"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    model_name: str = "fraudshield_v1"

@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))

config = AppConfig()
