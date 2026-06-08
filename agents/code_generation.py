import json
import logging
from dotenv import load_dotenv
from utils.agentstate import AgentState
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from utils.sanitize_json import clean_json_response
from llm.factory import LLMFactory
from utils.token_counter import TokenCounterCallback

load_dotenv()

token_cb = TokenCounterCallback()
llm = LLMFactory().get_llm("code_generation")

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("code_generation_agent")

code_generation_prompt = ChatPromptTemplate.from_template("""
    You are a Python data analysis assistant.

    You are given:
    1. The user query . 
    1. A pandas DataFrame named `df` containing stock fundamentals data.
    2. A list of relevant columns from the dataset.
    3. A list of filtering conditions mentioned in form of steps , derived from a natural language query containing which years,quarters,metrics are to be selected. 

    Your task is to generate clean, executable Python code that:
    - Filters the DataFrame based on the given conditions, years, and quarters. 
    - If a certain range of years or quarters is given , ensure that only records with complete data across all required time periods or categories are included . 
    - Uses **pandas** filtering (`loc[]`, or boolean masking`).
    - Handles both **numeric** and **categorical** comparisons.
    - Returns the filtered dataframe (`result`) along with the relevant columns . 

    ---

    ### Important Rules:
    1. Define the variables explicitly at the start of your code:
    - `relevant_columns` = list of columns used in filtering.
    2. Apply all **filtering conditions** .
        - There might be multiple Conditions , generate filter conditions in chronological order . 
    3. Ensure string comparisons use quotes (e.g., `Company == "TCS"`).
    4. Based on the user query provided , decide what variable name you are going to assign for the questions asked . Append it to the variable mapping object . 
       If the user is asking for two things give me the variable names for which you are going to assign it for . 
    4. Store the final filtered dataframe strictly under the variable name `result`.
    5. Drop duplicates from the result dataframe using 'drop_duplicates' method at the last stage 
    6. Do **not** include any comments, markdown fences, or explanations in the code.
    7. Return your output strictly in this JSON format (no markdown, no triple backticks, no explanations):

    Query : {query}
    Relevant Columns: {relevant_columns}
    Conditions: {conditions}


                                                          
    {{
    "code": "<valid python code that can run directly>",
    "reasoning": "<short reasoning about how the code applies filters and generates plots>",
    "variable_mappings" : [<variable_1>,<variable_2>,...] 
    }}
                                                          
                                    
""")


def code_generation_agent(state: AgentState) -> AgentState:
    """
    Generates executable Python code based on extracted columns, conditions, and visualization hints.
    Handles both filtering logic and visualization generation.
    """
    try:
        logger.info("Starting code_generation_agent...")

        # Extract from state safely
        relevant_columns = state.get("relevant_columns", [])
        conditions = state.get("conditions", [])
        user_query = state.get("query", "N/A")

        logger.info(f"User Query: {user_query}")
        logger.info(f"Relevant Columns: {relevant_columns}")
        logger.info(f"Conditions: {conditions}")

        # Initialize LLM
        try:
            logger.info("LLM initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize LLM client.")
            state["result"] = f"❌ LLM initialization failed: {str(e)}"
            return state

        # Create LLM chain
        chain = code_generation_prompt | llm | StrOutputParser()

        # Call the LLM
        try:
            response_text = chain.invoke({
                "query" : user_query ,
                "relevant_columns": relevant_columns,
                "conditions": conditions
            },
            config = {"callbacks": [token_cb]})
            logger.info("LLM response received successfully.")
        except Exception as e:
            logger.exception("Error invoking LLM during code generation.")
            state["result"] = f"❌ LLM invocation failed: {str(e)}"
            return state

        # Clean JSON
        response_text = clean_json_response(response_text)

        # Parse JSON safely
        try:
            response_json = json.loads(response_text)
            logger.info("Successfully parsed LLM JSON output.")
        except json.JSONDecodeError:
            logger.warning("Invalid JSON detected. Attempting to clean and reparse.")
            try:
                response_json = json.loads(clean_json_response(response_text))
                logger.info("Successfully parsed after cleaning.")
            except Exception as e:
                logger.exception("Failed to parse LLM response even after cleaning.")
                state["result"] = f"❌ JSON parsing failed: {str(e)}\nRaw Output:\n{response_text}"
                return state

        # Final validation
        if not isinstance(response_json, dict) or "code" not in response_json:
            logger.error("Invalid LLM response structure.")
            state["result"] = f"❌ Unexpected LLM output format:\n{response_text}"
            return state

        # Append to messages
        ai_message = AIMessage(content=json.dumps(response_json, indent=2), name="code_generation")
        updated_messages = list(state.get("messages", [])) + [ai_message]
        state["tokens_consumed"] += token_cb.total_tokens
        # Update agent state
        updated_state = state.copy()
        updated_state.update({
            "messages": updated_messages,
            "Thinking": response_json,
            "code": response_json.get("code", ""),
        })

        updated_state.setdefault("agents_used", []).append("code_generation")

        logger.info("✅ Code generation completed successfully.")
        return updated_state

    except Exception as e:
        logger.exception("Unhandled error in code_generation_agent.")
        state["result"] = f"❌ Unexpected error in code_generation_agent: {str(e)}"
        return state