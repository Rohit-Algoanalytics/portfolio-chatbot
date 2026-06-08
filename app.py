import streamlit as st
from generate_workflow import create_workflow
import pandas as pd
from langchain_core.messages import HumanMessage
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.s3_file_loader import load_portfolio_data, get_user_list
from dotenv import load_dotenv

load_dotenv(override=True)
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Portfolio agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top, #ffffff 0%, #f2f2f2 70%) !important;
    }
    .main { background: transparent !important; }
    .thinking-box { box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05); }
    .code-box {
        background-color: #0d1117;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background: rgba(245, 245, 245, 0.9);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "thinking_history" not in st.session_state:
    st.session_state.thinking_history = []
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "selected_user" not in st.session_state:
    st.session_state.selected_user = None
if "user_list" not in st.session_state:
    st.session_state.user_list = []


@st.cache_data(ttl=300)
def fetch_user_list():
    return get_user_list()


def main():
    # --- Header ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📈 Portfolio Chatbot")
    with col2:
        st.session_state.show_thinking = st.checkbox(
            "Show Agent Thinking",
            value=st.session_state.show_thinking
        )

    # --- Sidebar ---
    with st.sidebar:
        st.header("👤 Select User")

        if st.button("🔄 Refresh User List", use_container_width=True):
            st.cache_data.clear()

        try:
            user_list = fetch_user_list()
            print(user_list)
        except Exception as e:
            st.error(f"Failed to fetch users from S3: {e}")
            user_list = []

        if user_list:
            selected_user = st.selectbox(
                "Choose a user",
                options=["-- Select a user --"] + user_list,
                index=0,
                key="user_selectbox"
            )

            if selected_user != "-- Select a user --":
                if st.session_state.selected_user != selected_user:
                    with st.spinner(f"Loading portfolio for {selected_user}..."):
                        try:
                            st.session_state.dataset = load_portfolio_data(selected_user)
                            st.session_state.selected_user = selected_user
                            st.session_state.messages = []
                            st.success(f"✅ Loaded: {selected_user}")
                        except Exception as e:
                            st.error(f"Failed to load portfolio data: {e}")
                            st.session_state.dataset = None
        else:
            st.warning("No users found in S3 bucket.")

        # Show dataset info if loaded
        if st.session_state.dataset is not None:
            st.divider()
            st.success(
                f"📊 Active user: **{st.session_state.selected_user}**\n\n"
                f"{st.session_state.dataset.shape[0]} rows × {st.session_state.dataset.shape[1]} cols"
            )
            with st.expander("📊 Dataset Preview"):
                st.dataframe(st.session_state.dataset)
            with st.expander("📋 Dataset Info"):
                st.write("**Columns:**", list(st.session_state.dataset.columns))
                st.write("**Data Types:**")
                st.write(st.session_state.dataset.dtypes)

        st.divider()
        st.header("💡 Sample Queries")
        sample_queries = [
            "Show top 10 companies by Market Cap",
            "Calculate average ROE for last 5 years",
            "Find companies with ROE > 0.15 and ROCE > 0.20",
            "Analyze financial performance trends",
            "Show top 5 companies by ROCE with visualization",
            "Compare Net Profit between 2023 and 2024",
            "Rank companies by P/E ratio",
            "Find undervalued companies based on P/E",
            "Get the latest financial news about Reliance Industries"
        ]
        for query in sample_queries:
            if st.button(query, key=query, use_container_width=True):
                if st.session_state.dataset is not None:
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.rerun()

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- If no dataset loaded ---
    if st.session_state.dataset is None:
        st.info("👆 Please select a user from the sidebar to start analyzing their portfolio")
        return

    # --- Display chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "Thinking" in msg and st.session_state.show_thinking:
                with st.expander("🤔 Agent Thinking Process"):
                    st.markdown(msg["Thinking"])
            if "code" in msg:
                with st.expander("💻 Generated Code"):
                    st.code(msg["code"], language="python")
            if "dataframe" in msg:
                with st.expander("📊 Filtered DataFrame"):
                    st.dataframe(msg["dataframe"], use_container_width=True)

            st.write(msg["content"])

            if "visualization" in msg:
                fig_obj = msg["visualization"].get("figure")
                if fig_obj is not None:
                    if isinstance(fig_obj, go.Figure):
                        st.plotly_chart(fig_obj, use_container_width=True)
                    elif hasattr(fig_obj, "savefig"):
                        st.pyplot(fig_obj)
                    elif hasattr(fig_obj, "get_figure"):
                        st.pyplot(fig_obj.get_figure())
                    else:
                        st.pyplot(plt.gcf())

    # --- Chat input ---
    if prompt := st.chat_input("Ask anything about your financial data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                app = create_workflow()

                initial_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "dataset": st.session_state.dataset,
                    "query": prompt,
                    "selected_user": st.session_state.selected_user,
                    "code": "",
                    "result": "",
                    "visualization": {},
                    "Thinking": "",
                    "agents_used": [],
                    "tokens_consumed": 0,
                    "variables": []
                }

                response_placeholder = st.empty()
                thinking_placeholder = st.empty()
                code_placeholder = st.empty()
                viz_placeholder = st.empty()

                streamed_response = ""
                streamed_code = ""
                streamed_reasoning = []
                final_state = None

                for event in app.stream(initial_state):
                    for node_name, current_state in event.items():

                        if "Thinking" in current_state:
                            thinking_data = current_state.get("Thinking")
                            reasoning_text = ""
                            if isinstance(thinking_data, dict):
                                reasoning_text = thinking_data.get("reasoning", "")
                            elif isinstance(thinking_data, str):
                                reasoning_text = thinking_data
                            if reasoning_text:
                                streamed_reasoning.append(f"**{node_name.capitalize()} Reasoning:**\n{reasoning_text}")
                                combined_reasoning = "\n\n".join(streamed_reasoning)
                                thinking_placeholder.markdown(f"### 🤔 Agent Thinking Process\n\n{combined_reasoning}▌")

                        if node_name == "code_generator":
                            code_text = str(current_state.get("code", ""))
                            if code_text:
                                streamed_code = code_text
                                code_placeholder.markdown("### 💻 Generated Code")
                                code_placeholder.code(streamed_code, language="python")

                        if node_name == "response_generator":
                            msg_list = current_state.get("messages", [])
                            if msg_list:
                                latest_msg = msg_list[-1]
                                if hasattr(latest_msg, "content"):
                                    streamed_response = latest_msg.content
                                elif isinstance(latest_msg, dict):
                                    streamed_response = latest_msg.get("content", "")
                                response_placeholder.markdown(streamed_response + "▌")

                        if node_name == "executor":
                            vis = current_state.get("visualization", {})
                            if vis and "figure" in vis:
                                fig_obj = vis["figure"]
                                if isinstance(fig_obj, go.Figure):
                                    st.plotly_chart(fig_obj, use_container_width=True)
                                elif hasattr(fig_obj, "savefig"):
                                    viz_placeholder.pyplot(fig_obj)
                                elif hasattr(fig_obj, "get_figure"):
                                    viz_placeholder.pyplot(fig_obj.get_figure())
                                else:
                                    viz_placeholder.pyplot(plt.gcf())

                        if node_name == "generate_reponse":
                            dataframe = current_state.get("result")
                            if dataframe is not None:
                                st.dataframe(dataframe)

                        final_state = current_state

                if final_state and isinstance(final_state.get("result"), pd.DataFrame):
                    result_df = final_state["result"]
                    st.markdown("### 📊 Resulting DataFrame")
                    st.dataframe(result_df, use_container_width=True)
                    st.caption(f"Showing {len(result_df)} rows × {len(result_df.columns)} columns")

                if streamed_reasoning:
                    combined_reasoning = "\n\n".join(streamed_reasoning)
                    thinking_placeholder.markdown(f"### 🤔 Agent Thinking Process\n\n{combined_reasoning}")
                if streamed_code:
                    code_placeholder.markdown("### 💻 Generated Code")
                    code_placeholder.code(streamed_code, language="python")

                response_placeholder.markdown(streamed_response)

                msg_data = {
                    "role": "assistant",
                    "content": streamed_response,
                    "Thinking": "\n\n".join(streamed_reasoning),
                    "code": streamed_code,
                    "dataframe": final_state["result"]
                }
                if final_state and final_state.get("visualization"):
                    msg_data["visualization"] = final_state["visualization"]

                st.session_state.messages.append(msg_data)
                st.rerun()


if __name__ == "__main__":
    main()
