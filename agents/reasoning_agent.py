import json
import logging
from dotenv import load_dotenv
from utils.agentstate import AgentState
# from utils.llm_client import get_llm4,get_llm5
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from utils.sanitize_json import clean_json_response
from llm.factory import LLMFactory
from utils.token_counter import TokenCounterCallback

token_cb = TokenCounterCallback()
llm = LLMFactory().get_llm("reasoning")

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("reasoning_agent")

metric_extraction_prompt = ChatPromptTemplate.from_template("""
    You are a financial data analysis assistant that extracts structured information from natural language queries.

    Your task is to interpret a financial query and generate **clear, executable analysis steps** that will guide a code generation agent to write accurate Python/pandas code.
    Note - If the query mentions about plots , do no include it in the steps . 
    ### Dataset Context:
    The dataset provided represents the **complete source of truth** for all available financial ratios and metrics.
    - You must **only select metrics/columns that exist within the provided dataset**.
    - Do **not** invent or assume metrics that are not listed.
    - Always TRY to include 'tradingsymbol', 'sector', and 'profit_loss' columns as they are essential for analysis and visualization.
    - If a user mentions any figures in percentages(%) , keep the values as it is eg 10% -> 10% and not 0.10 .
    - If the user mentions about about Momentum filters and value filters in the query below is some context about the filters . All the values in the fields are boolean values "TRUE"/"FALSE" -
      MOMENTUM FILTERS : 
            momentum_filter_Final_Selected - FILTER WHERE MARKOWITZ OPTIMIZATION IS APPLIED AND WE GET SET OF 6 STOCKS .
            momentum_filter_Meets_Price_Ranking_Condition - FILTER WHERE THE STOCKS COMPOSITE AVERAGE OF P/S , P/B , P/E is computed and sorted in ascending order and top 50 % stocks are selected .
            momentum_filter_Meets_ROCE_ROE_Condition - FILTER WHERE THE STOCKS MEET THE CRITERIA FOR ROCE AND ROE .
            momentum_filter_Momentum_F1 - THIS FILTERS IS APPLIED IS ABOUT COMPUTING ALPHA AND BETA AGAINST THE STOCK AND INDEX . 
      VALUE FILTERS : 		
            value_filter_F3_Max_Sharpe - Final filter  WHERE THE STOCKS MEET THE CRITERIA FOR VALUE INVESTING .
            value_filter_F2_Price_Ranking - FILTER WHERE THE STOCKS COMPOSITE AVERAGE OF P/S , P/B , P/E is computed and sorted in ascending order and top 50 % stocks are selected .
            value_filter_ROE_ROCE_greater_than_15 - FILTER WHERE THE STOCKS MEET THE CRITERIA FOR ROCE AND ROE .
            value_filter_F1_Momentum - THIS FILTERS IS APPLIED IS ABOUT COMPUTING ALPHA AND BETA AGAINST THE STOCK AND INDEX . 
    Available dataset schema:
    {dataset_context}

    ---

    ### Instructions:

    #### 1. **Relevant Columns**

    Identify which columns in the query **and map them directly to the corresponding column names that exist in the provided dataset**.  
    - The dataset context represents the **complete list of available columns**, and you must only use column names or metric labels that exist in this dataset.  
    - If the user mentions general concepts (e.g., “profitability”, “growth”, “valuation”), correlate them with all relevant metrics **actually present in the dataset** that belong to those categories.  
    - If the user mentions abbreviations (e.g., “investment value”, “profit and loss”), match them to their **full dataset column names** .  
    - Do **not** create or infer new metrics or column names that are not explicitly part of the dataset.  
    - Return only the valid column names from the dataset that match or are contextually related to the query.


    #### 2. **Analysis Steps** (formerly "Conditions")

    Extract **sequential, executable analysis steps** that describe:
    - Data filtering operations
    - Time-based constraints
    - Statistical calculations (mean, variance, CV, percentiles etc.)
    - Logical operators ("and" and "or")
    - Ranking/sorting operations
    - Comparative conditions

    **Critical Rules:**

    **A. Statistical Operations**
    When the query mentions aggregations or statistical methods, explicitly state:
    - The metric to aggregate (e.g., "<metric_name>")
    - The time period for aggregation (e.g., "<metric_name> 2021-2024")
    - The statistical operation (e.g., "mean", "standard deviation", "coefficient of variation")
    - "And" or "OR" operators - Identify if there are "and" and "or" conditions . In such cases the both the conditions have to be part of the single step . 
    - The condition to apply (e.g., "> 0", "< 10%")

    **C. Ranking Operations**
    For "top N" or "bottom N" queries:
    - If a metric is specified for ranking, use that metric
    - If no ranking metric is specified, default to Market Cap
    - Format: "Rank companies by [metric] and select top/bottom N"

    **D. Sequential Logic**
    List steps in the **exact order** they should be executed:
    1. First, filter by time period
    2. Then, calculate any required statistics (if needed) 
    3. Then, apply threshold conditions . 
    4. Apply logical operators (and , or ,etc) (if needed) 
    4. Finally, rank or sort if needed .

    **E. Step Formatting**
    Each step should be:
    - **Actionable**: Clear enough for a code generator to implement
    - **Specific**: Include exact column names, operators, and values
    - **Complete**: Mention the metric, operation, time period, and condition
    - **Numbered**: Use sequential numbering for execution order

    ---

    ### Output Format (strict JSON only, no markdown, no explanations):

    {{
    "relevant_columns": ["<metric_1>", "<metric_2>", ...],
    "conditions": [
        "Step 1: <First operation with exact details>",
        "Step 2: <Second operation with exact details>",
        ...
    ],
    "reasoning": "<Brief summary of what the query is asking for>"
    }}

    ---

    ### Examples:

    **Example 1:**
    User Query: "Get companies where the mean Free Cash Flow (FCF) for the period 2021–2024 is positive, and the coefficient of variation (CV) in their Revenue Growth is below 10%, showing steady growth."

    Expected Output:
    {{
    "relevant_columns": ["symbol", "fiscalYear", "period", "Free Cash Flow (FCF)", "Revenue Growth"],
    "conditions": [
        "Step 1: Filter data for fiscal years 2021-2024 where period == 'FY'.",
        "Step 2: Group data by Symbol and calculate the mean of Free Cash Flow (FCF) across years 2021-2024 for each company.",
        "Step 3: Filter companies where the calculated mean FCF is greater than 0.",
        "Step 4: Group data by Symbol and calculate the coefficient of variation (standard deviation / mean) of Revenue Growth across years 2021-2024 for each company.",
        "Step 5: Filter companies where the calculated CV of Revenue Growth is less than 0.10."
    ],
    "reasoning": "Find companies with consistent positive cash flow (mean FCF > 0) and stable revenue growth (low variance indicated by CV < 10%) over the 2021-2024 period."
    }}

---

User Query: {user_query}
""")

def metric_extractor(state: AgentState) -> AgentState:
    """
    Extracts relevant metrics, time ranges, conditions, and visualization info 
    from a user's natural language financial query based on dataset context.
    """

    user_query = state.get("query", "")
    dataset = state.get("dataset")

    # Defensive check
    if dataset is None or dataset.empty:
        logger.error("No dataset found in state — cannot proceed with metric extraction.")
        state["result"] = "❌ No dataset provided. Please upload a valid dataset."
        return state

    # Prepare dataset context
    column_list = dataset.columns.tolist()
    dataset_context = (
        f"The uploaded dataset has {len(dataset)} rows and {len(column_list)} columns.\n"
        f"Columns:\n{json.dumps(column_list, indent=2)}"
    )

    # Initialize LLM
    try:
        logger.info("LLM initialized successfully for metric extraction.")
    except Exception as e:
        logger.exception("Failed to initialize LLM client.")
        state["result"] = f"LLM initialization failed: {e}"
        return state


    # Run the chain
    try:
        chain = metric_extraction_prompt | llm | StrOutputParser()
        response_text = chain.invoke({
            "user_query": user_query,
            "dataset_context": dataset_context
        },
        config = {"callbacks":[token_cb]})
        logger.info("LLM response received successfully.")
    except Exception as e:
        logger.exception("Error during LLM invocation or prompt execution.")
        state["result"] = f"LLM call failed: {e}"
        return state

    # Parse JSON output safely
    try:
        response_json = json.loads(response_text)
        logger.info("Successfully parsed JSON response from LLM.")
    except json.JSONDecodeError:
        logger.warning("Raw LLM response was not valid JSON. Attempting to clean and reparse.")
        try:
            cleaned = clean_json_response(response_text)
            response_json = json.loads(cleaned)
            logger.info("Successfully parsed cleaned JSON response.")
        except Exception as e:
            logger.exception("Failed to parse LLM output even after cleaning.")
            state["result"] = f"❌ Failed to parse LLM response: {e}\nRaw output:\n{response_text}"
            return state

    # Final validation
    if not isinstance(response_json, dict) or "relevant_columns" not in response_json:
        logger.error(f"Unexpected LLM JSON structure: {response_json}")
        state["result"] = f"❌ Invalid response structure: {response_json}"
        return state
    logger.info(response_json)
    # Update the state safely
    try:
        ai_message = AIMessage(content=json.dumps(response_json, indent=2), name="reasoning_agent")

        updated_messages = list(state.get("messages", [])) + [ai_message]
        state["tokens_consumed"] += token_cb.total_tokens
        updated_state = state.copy()

        reasoning = response_json.get("reasoning", "")
        conditions = response_json.get("conditions", [])

        if conditions:
            reasoning = (
                reasoning
                + "\n\nConditions:\n"
                + "\n".join(f"{i+1}. {c}" for i, c in enumerate(conditions))
            )

        response_json["reasoning"] = reasoning

        updated_state.update({
            "messages": updated_messages,
            "Thinking": reasoning ,
            "relevant_columns": response_json.get("relevant_columns", []),
            "conditions": response_json.get("conditions", []),
        })

        # Track agent usage
        updated_state.setdefault("agents_used", []).append("reasoning_agent")

        logger.info("Metric extraction completed successfully.")
        return updated_state

    except Exception as e:
        logger.exception("Error while updating the agent state.")
        state["result"] = f"❌ State update failed: {e}"
        return state
