# Build the LangGraph workflow
from utils.agentstate import AgentState
from langgraph.graph import StateGraph, END 
from agents import code_generation_agent,executor_agent,response_generator_agent,metric_extractor,plot_recommender,plot_code_generation_agent,plot_executor_agent
# from agents import analyzer_agent,chain_of_thought,executor_agent,response_generator_agent,code_generator_agent

def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyzer", metric_extractor)
    workflow.add_node("plot_recommender", plot_recommender)
    workflow.add_node("plot_code_generation",plot_code_generation_agent)
    # workflow.add_node("chain_of_thought",chain_of_thought)
    # workflow.add_node("web_search",web_search_agent)
    workflow.add_node("code_generator", code_generation_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("plot_executor_agent", plot_executor_agent)
    workflow.add_node("response_generator", response_generator_agent)

    workflow.add_edge("analyzer", "code_generator")
    workflow.add_edge("code_generator", "executor")
    # workflow.add_edge("chain_of_thought","code_generator")
    workflow.add_edge("executor", "plot_recommender")
    workflow.add_edge("plot_recommender", "plot_code_generation")
    workflow.add_edge("plot_code_generation", "plot_executor_agent")
    workflow.add_edge("plot_executor_agent", "response_generator")
    workflow.add_edge("response_generator", END)
    
    # Set entry point
    workflow.set_entry_point("analyzer")
    

    return workflow.compile()