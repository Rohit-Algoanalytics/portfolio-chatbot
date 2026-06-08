from fastapi import FastAPI, HTTPException , UploadFile , File
from pydantic import BaseModel
from typing import List , Optional 
import pandas as pd 
from agents import refine_financial_query,metric_extractor
from utils.agentstate import AgentState
from generate_workflow_fast_api import create_workflow
from fastapi.middleware.cors import CORSMiddleware
from utils.stream_utils import stream_agent_response
import json
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
import numpy as np 

app = FastAPI(
    title="NL Query Agent API",
    description="API for refining and enhancing financial natural language queries",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("output_india_final.csv")
# ---------- Request & Response Schemas ----------

class QueryRefineRequest(BaseModel):
    query: str


class QueryRefineResponse(BaseModel):
    original_query: str
    refined_query: str
    missing_clarifications: List[str]
    suggested_queries: List[str]


class ReasoningRequest(BaseModel):
    query: str


class ReasoningResponse(BaseModel):
    relevant_columns: List[str]
    conditions: List[str]

class ReasoningStatePayload(BaseModel):
    query: str
    country : str
    relevant_columns: List[str]
    conditions: List[str]
    reasoning: Optional[str] = ""

class QueryRequest(BaseModel):
    query: str
    country: str  # "USA" | "India"
    relevant_columns : List[str]
    conditions : List[str]

# ---------- Endpoint ----------

@app.post("/nl-query-agent/refine-query",response_model=QueryRefineResponse,summary="Refine a financial query",description="Refines a raw financial query and suggests improved finance-oriented alternatives")
def refine_query(request: QueryRefineRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = refine_financial_query(request.query)

    if "error" in result:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to refine query",
                "raw_output": result.get("raw_output")
            }
        )

    return result


@app.post(
    "/nl-query-agent/reasoning-extract",
    response_model=ReasoningResponse,
    summary="Extract reasoning steps from financial query"
)
def extract_reasoning(request: ReasoningRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Create dummy DataFrame ONLY for schema awareness
    # df = pd.DataFrame(columns=dataset_columns)

    initial_state: AgentState = {
        "query": request.query,
        "messages": [],
        "Thinking": "",
        "relevant_columns": [],
        "conditions": [],
        "agents_used": [],
        "tokens_consumed" : 0,
        'country' : "India"
    }

    try:
        updated_state = metric_extractor(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning agent execution failed: {str(e)}"
        )
    
    if "relevant_columns" not in updated_state or "conditions" not in updated_state:
        raise HTTPException(
            status_code=500,
            detail="Invalid response from reasoning agent"
        )

    return {
        "relevant_columns": updated_state["relevant_columns"],
        "conditions": updated_state["conditions"]
    }



@app.post("/nl-query-agent/run",summary="Continue NL Query Agent from reasoning state")
async def continue_from_reasoning( state_payload: ReasoningStatePayload):

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    # ----------------------------
    # 2. Rehydrate AgentState
    # ----------------------------
    state: AgentState = {
        "query": state_payload.query,
        "messages": [],
        "Thinking": state_payload.reasoning or "",
        "relevant_columns": state_payload.relevant_columns,
        "conditions": state_payload.conditions,
        "code": "",
        "plot_code": "",
        "result": None,
        "visualization": {},
        "agents_used": ["reasoning_agent"],  # critical,
        "tokens_consumed" : 0 , 
        "country" : state_payload.country
    }

    # ----------------------------
    # 3. Run workflow from NEXT node
    # ----------------------------
    try:
        
        workflow = create_workflow()
        final_state = workflow.invoke(state)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {e}"
        )

    # ----------------------------
    # 4. Prepare response
    # ----------------------------
    result_df = final_state.get("result")
    result_df = result_df.fillna("")

    response = {
        "agents_used": final_state.get("agents_used", []),
        "reasoning": final_state.get("Thinking", ""),
        "generated_code": final_state.get("code", ""),
        "generated_plot_code": final_state.get("plot_code", ""),
        "conditions": final_state.get("conditions", [])
    }

    if isinstance(result_df, pd.DataFrame):
        response["result_preview"] = {
            "rows": len(result_df),
            "columns": list(result_df.columns),
            "data": result_df.to_dict(orient="records")
        }

    if final_state.get("visualization"):
        fig = final_state.get("visualization")
        fig = fig.get("figure","")
        print(type(fig))
        print(fig)
        fig =  fig.to_json()
        response["visualization"] = fig
    else:
        response["visualization"] = {"available": False}
    return response


@app.post("/nl-query-agent/stream")
async def stream_nl_query(request: QueryRequest):
    try:
        workflow = create_workflow()
        state: AgentState = {
            "query": request.query,
            "messages": [],
            "Thinking": "",
            "relevant_columns": request.relevant_columns,
            "conditions": request.conditions,
            "code": "",
            "plot_code": "",
            "result": None,
            "visualization": {},
            "agents_used": ["reasoning_agent"],
            "tokens_consumed": 0,
            "country": request.country
        }

        state_container = {}

        async def event_generator():
            # Stream the response with proper SSE formatting
            async for chunk in stream_agent_response(state, workflow, state_container):
                yield f"{chunk}"
            
            # Get the final state from container
            final_state = state_container.get('final_state', state)
            
            print("=== FINAL STATE ===")
            result_df = final_state.get("result")
            
            response_payload = {
                "tokens_consumed": final_state.get("tokens_consumed", 0),
                "plot_code": final_state.get("plot_code", ""),
                "code": final_state.get("code", ""),
                "dataframe": None,
                "visualization": None
            }
            
            # DataFrame serialization
            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                df_cleaned = result_df.copy()
                df_cleaned = df_cleaned.replace([np.inf, -np.inf], None)
                df_cleaned = df_cleaned.where(pd.notnull(df_cleaned), None)
                response_payload["dataframe"] = df_cleaned.to_dict(orient="records")

            
            # Plot serialization
            vis = final_state.get("visualization")
            if vis and "figure" in vis:
                fig = vis["figure"]
                if hasattr(fig, "to_plotly_json"):
                    response_payload["visualization"] = fig.to_plotly_json()
                    print("Visualization serialized with to_plotly_json()")
                elif hasattr(fig, "to_json"):
                    response_payload["visualization"] = fig.to_json()
                    print("Visualization serialized with to_json()")
                elif hasattr(fig, "to_dict"):
                    response_payload["visualization"] = fig.to_dict()
                    print("Visualization serialized with to_dict()")
            
            # # Serialize and send final event
            # try:
            #     final_json = json.dumps(response_payload, allow_nan=False)
            # except ValueError:
            #     # Fallback if NaN values slip through
            #     final_json = json.dumps(response_payload).replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
            
            # print(f"Final JSON size: {len(final_json)} bytes")
            
            # Send as SSE event
            print(response_payload)
            yield f"event: final\ndata: {response_payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
