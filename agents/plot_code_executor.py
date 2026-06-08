import io
import logging
import traceback
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from utils.agentstate import AgentState

load_dotenv()

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] plot_executor_agent: %(message)s"
)
logger = logging.getLogger("plot_executor_agent")


def plot_executor_agent(state: AgentState) -> AgentState:
    """
    Executes generated code or plot code safely with logging, output capture, and error handling.
    Handles both Matplotlib and Plotly figures and supports multiple plots in a list.
    """

    output_buffer = io.StringIO()
    exec_globals = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'result': None,
        'fig': None
    }

    try:
        agents_used = state.get("agents_used", [])
        if not agents_used:
            state["result"] = "❌ No agents recorded in the pipeline."
            logger.error("No agent history found in state.")
            return state

        last_agent = agents_used[-1]
        logger.info(f"🔍 Last agent: {last_agent}")

        with redirect_stdout(output_buffer):
            # --- Plot Code Execution ---
            df = state.get("result")
            if df is None or not hasattr(df, "columns"):
                logger.error("Missing or invalid DataFrame in 'state[result]'.")
                state["result"] = "❌ Missing or invalid DataFrame for plotting."
                return state

            plot_code = state.get("plot_code", "")
            if not plot_code.strip():
                logger.error("Plot code missing from state.")
                state["result"] = "❌ No plot code available to execute."
                return state

            exec_globals["df"] = df
            logger.info("🎨 Executing plot code...")
            try:
                exec(plot_code, exec_globals)
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"❌ Error executing plot code: {e}")
                state["visualization"] = {'figure': "" , 'type': 'plotly'}
                state["error_traceback"] = error_trace
                state["Thinking"]={"reasoning" : f"❌ Error executing plot code: {e}" }
                return state

            fig = exec_globals.get("fig", None)

            # Handle multiple plots (e.g., fig = [fig1, fig2, fig3])
            if isinstance(fig, list):
                logger.info(f"📊 Multiple figures detected: {len(fig)} plots.")
                state["visualization"] = {'figure': fig, 'type': 'plotly-multi'}
            elif fig is not None:
                logger.info("📈 Single figure detected.")
                state["visualization"] = {'figure': fig, 'type': 'plotly'}
            else:
                logger.warning("⚠️ Plot executed but no figure variable found.")

        print(type(fig))
        # --- Capture stdout (any print statements from exec) ---
        stdout_output = output_buffer.getvalue().strip()
        if stdout_output:
            logger.info("🧾 Captured stdout from execution.")
            state["execution_log"] = stdout_output

        # Append execution summary to message history
        ai_message = AIMessage(content=f"✅ Execution completed for agent: {last_agent}")
        state.setdefault("messages", []).append(ai_message)
        state["Thinking"]={"reasoning" : f"✅ Execution completed for agent: {last_agent}" }
        state.setdefault("agents_used", []).append("plot_code_executor")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"❌ Unhandled Exception in Executor: {e}")
        state["result"] = f"❌ Unexpected Error in Executor: {str(e)}"
        state["error_traceback"] = error_trace
        state["Thinking"]={"reasoning" : f"❌ Execution failed for {last_agent}:\n```\n{error_trace}\n```" }
        ai_message = AIMessage(content=f"❌ Execution failed for {last_agent}:\n```\n{error_trace}\n```")
        state.setdefault("messages", []).append(ai_message)

    return state
