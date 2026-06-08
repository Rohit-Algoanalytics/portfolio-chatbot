
from utils.agentstate import AgentState
from langchain_core.messages import  AIMessage
from llm.factory import LLMFactory
from utils.token_counter import TokenCounterCallback


token_cb = TokenCounterCallback()
llm = LLMFactory().get_llm("generate_response")

# Response Generator Agent Node
def response_generator_agent(state: AgentState) -> AgentState:
    """Generates a natural language response"""
    # serch_tool = DuckDuckGoSearchRun(region = 'us_en')
    
    # llm2 = get_llm4()
    # llm1 = llm1.bind_tools(serch_tool)
    query = state["query"]
    result = state["result"]
    
    response_prompt = f"""
    You are a financial analysis assistant.

    Your task is to create a **professional, data-driven summary** based on the provided query and the final filtered dataset.

    ### Important Instructions:
    1. **Do NOT apply or assume any new filters or transformations** — the dataset provided already reflects all filters based on the user's query.
    2. Focus only on **interpreting and summarizing** the dataset.
    3. If the dataset is small (less than ~50 rows), analyze it in detail:
    - Highlight key patterns, outliers, top or bottom performers, and overall trends.
    - Mention notable statistics such as average, median, or range of key financial ratios.
    4. If the dataset is large (more than ~50 rows):
    - Avoid listing individual values.
    - Summarize general patterns and trends instead.
    - Briefly restate what the query was and what kind of companies or conditions it represents.
    - Mention the dominant or interesting observations at an aggregate level.
    5. Keep the tone professional and data-focused, suitable for a financial report.
    6. Add some emojis to make the response catchy . 

    ### Context:
    Query: {query}

    ### Final Dataset (Filtered):
    {str(result)}

    Return your answer as a **short, structured financial summary** — no code, no markdown formatting, and no restatement of instructions.
    """

    # state["thinking_steps"].append("📝 Generating natural language response...")
    
    response = llm.invoke(response_prompt)
    final_response = response.content
    
    # state["thinking_steps"].append("✅ Response generated")
    state["messages"] = state["messages"] + [AIMessage(content=final_response)] #type:ignore
    state["Thinking"] = {"reasoning":"✅ Response generated"}
    state.setdefault("agents_used", []).append("generate_response")
    print(state["tokens_consumed"])
    return state