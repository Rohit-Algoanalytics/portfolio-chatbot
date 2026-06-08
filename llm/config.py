LLM_CONFIG = {
    "default": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "temperature": 1,
    },

    "agents": {
        "reasoning": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "temperature": 1,
        },

        "code_generation": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "temperature": 1,
        },

        "plot_recommender": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "temperature": 1,
        },

        "plot_code_generation": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "temperature": 1,
        },

        "generate_response": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "temperature": 1,
        },
        "refine_query": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "temperature": 1,
        }
    }
}
