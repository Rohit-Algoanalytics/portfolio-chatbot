from abc import ABC, abstractmethod
from typing import Any

class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def get_chat_model(self) -> Any:
        """
        Return a LangChain-compatible chat model.
        """
        pass
