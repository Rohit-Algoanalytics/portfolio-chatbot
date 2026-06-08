import json
import logging
from dotenv import load_dotenv
from utils.agentstate import AgentState
from utils.sanitize_json import clean_json_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from llm.factory import LLMFactory
from utils.token_counter import TokenCounterCallback


token_cb = TokenCounterCallback()
llm = LLMFactory().get_llm("plot_code_generation")
load_dotenv()

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] plot_code_generation_agent: %(message)s"
)
logger = logging.getLogger("plot_code_generation_agent")
load_dotenv()

plot_code_generation_prompt = ChatPromptTemplate.from_template("""
    You are a Python data analysis assistant.

    You are given:
    1. A list of recommended plots to visually explain the analysis.
    2. The final filtered dataframe info. 

    Your task is to generate clean, executable Python code that:
    - Generates **interactive Plotly visualizations** based on `recommended_plots`.
    - You may generate multiple charts (up to 4) based on the recommendations.
    - However, all charts must be stored together under **a single variable named `fig`**.
    - Do NOT create separate variables like fig1, fig2, etc.
    - You can store them as:
        - a combined multi-subplot figure using `from plotly.subplots import make_subplots` if charts can fit together.
    - The output must include this single variable `fig` that contains all visualizations.
    - Do not store plots separately.
    - Return the plot strictly under the variable name called as (`fig`). 

    ---

    ### Important Rules:
    1. Assume 'df' is the filtered dataframe . 
    2. Do **not** include any comments, markdown fences, or explanations in the code.
    3. You are already provided with the filtered dataframe , do not apply any filtering logic based on the recommendation given to you . 
    4. Add Try except blocks for each figure even if one figure fails , other figures should be generated . 
    3. Return your output strictly in this JSON format (no markdown, no triple backticks, no explanations)
    6. Do not use statistical (mean , etc), logical ("and" , "or") , conditional operators (">","<") on the dataframe . Since the dataframe is already in the desired format.
    7. When generating Plotly subplots, always place pie charts inside a subplot with type='domain'. Use specs in make_subplots to define subplot types.  
    8. All the values mentioned in the dataframe are in fractions and not in percentages.


    dataset context : {dataset_context}
    Recommended plots :{recommended_plots}
    
                                                          
    {{
    "code": "<valid python code that can run directly>",
    "reasoning": "<short reasoning about how the code applies filters and generates plots>"
    }}
                                                          
                                    
""")


def plot_code_generation_agent(state: AgentState) -> AgentState:
    """
    Generates Python code for trader-focused visualizations (up to 4 Plotly plots)
    based on relevant columns, conditions, and plot recommendations.
    """

    try:
        logger.info("🎨 Starting plot_code_generation_agent...")

        # --- Extract inputs safely ---
        filtered_df = state.get("result", None)
        if filtered_df is None or not hasattr(filtered_df, "columns"):
            logger.error("No valid filtered dataframe found in state.")
            state["result"] = "❌ Missing or invalid filtered DataFrame for plot generation."
            return state

        column_list = filtered_df.columns.tolist()
        dataset_context = f"""
        The uploaded dataset has {len(filtered_df)} rows and {len(column_list)} columns.
        Columns:
        {json.dumps(column_list, indent=2)}
        """

        relevant_columns = state.get("relevant_columns", [])
        thinking = state.get("Thinking", {})
        recommended_plots = thinking.get("plot_recommendation", [])

        if not relevant_columns:
            logger.warning("⚠️ relevant_columns list is empty.")
        if not recommended_plots:
            logger.warning("⚠️ No recommended plots found — generating fallback visualization code.")

        logger.info(recommended_plots)
        logger.info(dataset_context)

        # --- Initialize LLM ---
        try:
            # llm = get_llm4()
            logger.info("✅ LLM initialized successfully.")
        except Exception as e:
            logger.exception("❌ Failed to initialize LLM.")
            state["result"] = f"Error initializing LLM: {str(e)}"
            return state

        # --- Create and run the chain ---
        chain = plot_code_generation_prompt | llm | StrOutputParser()

        try:
            logger.info("🚀 Invoking LLM for plot code generation...")
            response_text = chain.invoke({  
                "recommended_plots": recommended_plots,
                "dataset_context": dataset_context
            },
            config = {"callbacks": [token_cb]})
        except Exception as e:
            logger.exception("❌ LLM invocation failed during plot code generation.")
            state["result"] = f"Error invoking LLM: {str(e)}"
            return state

        # --- Clean and parse JSON ---
        response_text = clean_json_response(response_text)
        try:
            response_json = json.loads(response_text)
            logger.info("✅ Successfully parsed LLM JSON output.")
        except json.JSONDecodeError:
            logger.warning("⚠️ JSON parsing failed. Retrying after cleaning.")
            try:
                cleaned = clean_json_response(response_text)
                response_json = json.loads(cleaned)
                logger.info("✅ Parsed JSON successfully after cleanup.")
            except Exception as e:
                logger.exception("❌ Failed to parse JSON after cleaning.")
                state["result"] = f"Invalid JSON from model: {str(e)}"
                return state

        # --- Validate JSON Structure ---
        if "code" not in response_json:
            logger.error("❌ LLM response missing 'code' key.")
            state["result"] = f"Unexpected LLM output: {response_json}"
            return state

        # --- Update Agent State ---
        ai_message = AIMessage(content=json.dumps(response_json, indent=2), name="plot_code_generation")
        updated_messages = list(state.get("messages", []))
        updated_messages.append(ai_message)
        state["tokens_consumed"] += token_cb.total_tokens
        updated_state = state.copy()
        updated_state.update({
            "messages": updated_messages,
            "plot_code": response_json.get("code", ""),
            "Thinking": response_json,
        })
        updated_state.setdefault("agents_used", []).append("plot_code_generation")

        logger.info("✅ Plot code generation completed successfully.")
        return updated_state

    except Exception as e:
        logger.exception("Unhandled exception in plot_code_generation_agent.")
        state["result"] = f"❌ Unexpected error in plot_code_generation_agent: {str(e)}"
        return state