
from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
import operator
import pandas as pd 

# Agent State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    dataset: pd.DataFrame
    relevant_columns : list
    conditions : list
    query: str
    code: str
    plot_code : str
    result: str
    visualization: dict
    Thinking: Any
    agents_used : list
    tokens_consumed : int
    variables : list
