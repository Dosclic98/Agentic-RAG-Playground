"""Default local model settings; edit these values to configure the chat."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """Settings shared by the model client, conversation, and PDF tools."""

    model: str = "qwen3.8:27b-q8_0"
    host: str = "http://127.0.0.1:11434"
    context_length: int = 65536
    max_output_tokens: int = 8192
    project_root: Path = Path(__file__).resolve().parents[1]
    ocr_data_path: str = "data/tessdata"
    output_width: int = 100
    verbose: bool = False
