import io
import traceback
import logging
from contextlib import redirect_stdout
from utils.agentstate import AgentState
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.messages import AIMessage

load_dotenv()

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] executor_agent: %(message)s"
)
logger = logging.getLogger("executor_agent")


def executor_agent(state: AgentState) -> AgentState:
    """Executes either the main generated code or the plot code based on the last agent used."""

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
            logger.error("No agent history found in state.")
            state["result"] = "❌ No agents recorded in the pipeline. Cannot determine execution context."
            return state

        last_agent = agents_used[-1]
        logger.info(f"🔍 Last agent: {last_agent}")

        with redirect_stdout(output_buffer):
            code = state.get("code", "")
            df= state.get("dataset")
            # df = state.get("dataset", None)

            if df is None or df.empty:
                logger.error("Dataset missing or empty in state.")
                state["result"] = "❌ No valid dataset available for execution."
                return state

            if not code.strip():
                logger.error("No code found to execute.")
                state["result"] = "❌ No code found for execution."
                return state

            logger.info("⚙️ Executing main analysis code...")
            exec_globals["df"] = df
            exec(code, exec_globals)

            result = exec_globals.get("result", None)
            fig = exec_globals.get("fig", None)

            if result is not None:
                state["result"] = result
                logger.info("✅ Code executed and DataFrame result generated.")
            else:
                logger.warning("Code executed but no result DataFrame found.")

            if fig is not None:
                state["visualization"] = {"figure": fig, "type": "matplotlib"}
                logger.info("📊 Figure object captured successfully.")

        # Capture standard output (print logs inside exec)
        stdout_output = output_buffer.getvalue().strip()
        if stdout_output:
            logger.info(f"🧾 Captured stdout during execution:\n{stdout_output}")
            state["execution_log"] = stdout_output

        # Append message to history
        ai_message = AIMessage(content=f"✅ Execution complete for agent: {last_agent}" , name = "code_executor")
        state["messages"].append(ai_message)
        state["Thinking"] = {"reasoning": f"✅ Execution complete for agent: {last_agent}"}
        state.setdefault("agents_used", []).append("code_executor")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"❌ Execution failed: {e}\n{error_trace}")
        state["result"] = f"❌ Error during execution: {str(e)}"
        state["error_traceback"] = error_trace
        state["Thinking"] = {"reasoning": f"❌ Execution failed for agent `{last_agent}`:\n```\n{error_trace}\n```"}
        ai_message = AIMessage(content=f"❌ Execution failed for agent `{last_agent}`:\n```\n{error_trace}\n```")
        state["messages"].append(ai_message)
    return state
