from langchain_core.callbacks import BaseCallbackHandler

class TokenCounterCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage", {})

        self.total_tokens += usage.get("total_tokens", 0)
