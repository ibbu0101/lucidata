from dataclasses import dataclass


@dataclass
class LLMSettings:
    default_provider: str = "ollama"
    default_model: str = "llama3"
    default_timeout_seconds: float = 10.0
    max_validation_retries: int = 2
    temperature: float = 0.2


settings = LLMSettings()