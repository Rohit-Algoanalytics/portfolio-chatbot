import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from llm.factory import LLMFactory
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("refine_query")

llm = LLMFactory().get_llm("refine_query")

QUERY_REFINER_PROMPT = ChatPromptTemplate.from_template("""
You are a **Financial Query Refinement Assistant**.

The user provides a raw natural-language query related to stock market or financial analysis.
Your role is to **improve the quality of the query**, not to answer it.

---

### 🎯 Objectives

1. **Rewrite the original query** into a:
   - Grammatically correct
   - Clear
   - Finance-aware
   - Analyst-grade version  
   This rewritten query should preserve the user’s original intent but remove ambiguity.

2. **Identify missing or unclear constraints**, such as:
   - Time period (years, quarters, ranges)
   - Financial metrics or ratios
   - Thresholds or comparison logic
   - Ranking criteria (top/bottom, sort order)
   - Scope (single stock vs universe of stocks)

3. **Suggest alternative, professional-grade financial queries** that:
   - Reflect how a trader, analyst, or portfolio manager would phrase the question
   - Are realistic and actionable
   - Explore the same intent from slightly different analytical angles

---

### ❗ Strict Rules

- ❌ Do NOT apply filters or calculations
- ❌ Do NOT generate code
- ❌ Do NOT assume or invent data
- ❌ Do NOT answer the query
- ✅ ONLY focus on **language refinement and analytical framing**
- ✅ Suggested queries should be **finance- and trading-oriented**, not generic

---

### 📤 Output Format  
Return **STRICT JSON only** (no markdown, no explanations, no extra text):

{{
  "original_query": "<exact user query>",
  "refined_query": "<grammatically correct, precise, finance-aware rewritten query>",
  "missing_clarifications": [
    "<missing or ambiguous aspect 1>",
    "<missing or ambiguous aspect 2>"
  ],
  "suggested_queries": [
    "<professional trading/analysis-oriented alternative query 1>",
    "<professional trading/analysis-oriented alternative query 2>",
    "<professional trading/analysis-oriented alternative query 3>"
  ]
}}

---

User Query:
{user_query}
""")



def refine_financial_query(user_query: str) -> dict:
    chain = QUERY_REFINER_PROMPT | llm | StrOutputParser()

    response = chain.invoke({"user_query": user_query})

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON from LLM",
            "raw_output": response
        }



