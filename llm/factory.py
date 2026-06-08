from llm.config import LLM_CONFIG
from llm.providers import OpenAILLM , GroqLLM


PROVIDER_REGISTRY = {
    "openai": OpenAILLM(),
    "groq": GroqLLM()
}

class LLMFactory:
    def __init__(self):
        self.config = LLM_CONFIG

    def get_llm(self, agent_name: str):
        agent_cfg = self.config["agents"].get(
            agent_name,
            self.config["default"]
        )

        provider_name = agent_cfg["provider"]
        model = agent_cfg["model"]
        temperature = agent_cfg.get("temperature", 1)

        provider = PROVIDER_REGISTRY.get(provider_name)
        if not provider:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        return provider.get_chat_model(
            model=model,
            temperature=temperature
        )
