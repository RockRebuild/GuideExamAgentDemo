# server/state.py
# Per-request context variables (replaces st.session_state)
import contextvars
from typing import List


# Contexts collected during a single chat request (for RAGAS evaluation)
request_contexts: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "request_contexts", default=[]
)

# Tool call records for the current request
request_tool_records: contextvars.ContextVar[List[dict]] = contextvars.ContextVar(
    "request_tool_records", default=[]
)
