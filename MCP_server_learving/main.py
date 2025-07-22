from mcp.server.fastmcp import FastMCP
import uvicorn
import os
from typing import List

# In-memory mock database with 20 leave days to start
employee_leaves = {
    "E001": {"balance": 18, "history": ["2024-12-25", "2025-01-01"]},
    "E002": {"balance": 20, "history": []}
}

# Create MCP server
mcp = FastMCP("LeaveManager")


# Tool definitions (keep your existing tools)
@mcp.tool()
def get_leave_balance(employee_id: str) -> str:
    """Check how many leave days are left for the employee"""
    data = employee_leaves.get(employee_id)
    if data:
        return f"{employee_id} has {data['balance']} leave days remaining."
    return "Employee ID not found."


@mcp.tool()
def apply_leave(employee_id: str, leave_dates: List[str]) -> str:
    """Apply leave for specific dates"""
    if employee_id not in employee_leaves:
        return "Employee ID not found."

    requested_days = len(leave_dates)
    available_balance = employee_leaves[employee_id]["balance"]

    if available_balance < requested_days:
        return f"Insufficient leave balance. You requested {requested_days} day(s) but have only {available_balance}."

    employee_leaves[employee_id]["balance"] -= requested_days
    employee_leaves[employee_id]["history"].extend(leave_dates)
    return f"Leave applied for {requested_days} day(s). Remaining balance: {employee_leaves[employee_id]['balance']}."


@mcp.tool()
def get_leave_history(employee_id: str) -> str:
    """Get leave history for the employee"""
    data = employee_leaves.get(employee_id)
    if data:
        history = ', '.join(data['history']) if data['history'] else "No leaves taken."
        return f"Leave history for {employee_id}: {history}"
    return "Employee ID not found."


if __name__ == "__main__":
    # Use the MCP server's built-in SSE app
    app = mcp.sse_app()
    port = int(os.environ.get("PORT", 8001))

    # Bind to 0.0.0.0 to accept external traffic
    uvicorn.run(app, host="0.0.0.0", port=port)
