from utils.agentstate import AgentState
from dotenv import load_dotenv
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from utils.sanitize_json import clean_json_response
from llm.factory import LLMFactory
from utils.token_counter import TokenCounterCallback

token_cb = TokenCounterCallback()
llm = LLMFactory().get_llm("plot_recommender")

load_dotenv()
# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] plot_recommender: %(message)s"
)
logger = logging.getLogger("plot_recommender")


plot_recommendation_prompt = ChatPromptTemplate.from_template("""
You are a **financial data visualization assistant** specialized in equity and stock analysis.

You are given:
- User query , analyse if the user is asking for a desired plot . If yes then user's desired plot should be one of the generated plots . Do not add any data filtering logic in the code , the data provided is filtered based on users conditions . 
- Dataset context on which the plots will be generated on . Generated plots based on the provided dataset context only .

Your task:
Generate **up to 4 intelligent visualization recommendations** that would help a **trader or financial analyst** gain deeper insights from the filtered dataset.

---

### 🧠 Think Like a Trader:
When making recommendations, consider how professional investors visualize:
- **Profitability vs Valuation** (e.g., ROE vs P/E)
- **Growth over Time** (e.g., Revenue or EPS growth trends)
- **Risk vs Return** (e.g., Volatility vs Sharpe-like metrics)
- **Sector or Company Comparison** (e.g., bar charts by sector)
- **Momentum and Seasonality** (e.g., line plots by quarter)
- **Correlation & Outlier Detection** (e.g., scatter or heatmaps)

If the query or metrics imply multi-year or quarterly data, prefer **trend-based visualizations**.
If they imply relative performance or ranking, prefer **bar or scatter-based charts**.

---

### 📊 Examples of Possible Chart Types
- `"line chart"` – Trend of a metric over time or across quarters.
- `"bar chart"` – Comparison of a metric across companies or sectors.
- `"pie chart"` – Distribution of industry wise stocks.
- `"scatter plot"` – Relationship between two metrics (e.g., ROE vs P/E).
- `"heatmap"` – Multi-metric correlation matrix or sectoral comparison.
- `"box plot"` – Distribution of a financial metric across companies.
- `"bubble chart"` – Three-dimensional comparison (e.g., ROE vs P/E sized by Market Cap).

---

### 🔍 Input Context

Query : {user_query}
dataset_context : {dataset_context}

---

### 🧩 Output Format (strict JSON only — no markdown, no text, no code fences)

{{
"plot_recommendation": [
    {{
    "description": "<short natural language description of what this chart shows>",
    "insight": "<what insight a trader or analyst could gain from this visualization>"
    }},
    ...
]
}}

---

### 🧠 Guidelines for Recommendations
1. Suggest **up to 4** diverse and insightful visualizations (line, bar, scatter, etc.).
2. Avoid redundant plots — each should reveal a **unique angle** on the data.
3. Base chart type and description **only** on `relevant_columns` and `conditions`.
4. Ensure each chart recommendation is **actionable for a trader**, not generic.
5. If the context is unclear, infer the most meaningful charts based on the metrics provided.
6. Do not add any data filtering logic into the recommendation , infer user's query only for plot generation . 

Now, based on the above context, generate your visualization recommendations.
""")


def plot_recommender(state: AgentState) -> AgentState:
    """
    LLM-based plot recommendation agent.
    Takes relevant columns and conditions from the previous step,
    and suggests visualizations (up to 4) that are most useful for analysis.
    """

    try:
        logger.info("🎯 Starting Plot Recommender Agent...")

        # Extract query and context safely
        user_query = state.get("query", "")
        relevant_columns = state.get("relevant_columns", [])
        conditions = state.get("conditions", [])
        thinking = state.get("Thinking", {})
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

        # Sanity check
        if not user_query.strip():
            logger.error("No user query provided.")
            state["result"] = "❌ Missing user query for plot recommendation."
            return state

        if not relevant_columns:
            logger.warning("No relevant columns found — fallback to empty list.")
        if not conditions:
            logger.warning("No conditions found — fallback to empty list.")

        # Initialize LLM
        try:
            # llm = get_llm4()
            logger.info("✅ LLM initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize LLM client.")
            state["result"] = f"❌ LLM initialization failed: {e}"
            return state


        # Create LangChain chain
        chain = plot_recommendation_prompt | llm | StrOutputParser()

        # Invoke the LLM
        try:
            logger.info("🧠 Querying LLM for plot recommendations...")
            response_text = chain.invoke({
                "user_query": user_query,
                "relevant_columns": relevant_columns,
                "conditions": conditions,
                "dataset_context":dataset_context
            },
            config =  {"callbacks":[token_cb]})
            logger.info("✅ LLM responded with plot recommendations.")

        except Exception as e:
            logger.exception("Error during LLM invocation.")
            state["result"] = f"❌ Plot recommender LLM call failed: {e}"
            return state

        # Parse JSON safely
        try:
            response_json = json.loads(response_text)
            logger.info("✅ Successfully parsed LLM JSON output.")
        except json.JSONDecodeError:
            logger.warning("⚠️ Invalid JSON detected — attempting cleanup.")
            try:
                cleaned = clean_json_response(response_text)
                response_json = json.loads(cleaned)
                logger.info("✅ Successfully parsed cleaned JSON response.")
                logger.info(response_json)
            except Exception as e:
                logger.exception("❌ Failed to parse even after cleaning.")
                state["result"] = f"Failed to parse LLM JSON response: {e}\nRaw Output:\n{response_text}"
                return state

        # Validate structure
        if not isinstance(response_json, dict) or "plot_recommendation" not in response_json:
            logger.error(f"Unexpected LLM response format: {response_json}")
            state["result"] = f"❌ Invalid plot recommendation structure: {response_json}"
            return state

        # Update thinking state with recommendations
        plot_recs = response_json.get("plot_recommendation", [])
        thinking["plot_recommendation"] = plot_recs
        thinking["reasoning"] = "\n".join(
            f"{i+1}. {rec['insight']}"
            for i, rec in enumerate(plot_recs)
        )

        # Append message
        ai_message = AIMessage(content=json.dumps(response_json, indent=2), name="plot_recommender")
        updated_messages = list(state.get("messages", [])) + [ai_message]
        state["tokens_consumed"] += token_cb.total_tokens
        # Update overall agent state
        updated_state = state.copy()
        updated_state.update({
            "messages": updated_messages,
            "Thinking": thinking,
        })

        updated_state.setdefault("agents_used", []).append("plot_recommender")
        
        logger.info(f"🎨 Plot recommendation completed. Total suggestions: {len(plot_recs)}")
        return updated_state
        
    except Exception as e:
        logger.exception("Unhandled exception in plot_recommender.")
        state["result"] = f"❌ Unexpected error in plot_recommender: {e}"
        return state