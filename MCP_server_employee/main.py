from mcp.server.fastmcp import FastMCP
import uvicorn

# Mock employee database
employees = {
    "E001": {
        "name": "John Doe",
        "department": "Engineering",
        "position": "Senior Developer",
        "email": "john.doe@company.com",
        "phone": "+1-555-0101",
        "manager": "M001"
    },
    "E002": {
        "name": "Jane Smith",
        "department": "Marketing",
        "position": "Marketing Manager",
        "email": "jane.smith@company.com",
        "phone": "+1-555-0102",
        "manager": "M002"
    }
}

mcp = FastMCP("EmployeeDirectory")


@mcp.tool()
def get_employee_info(employee_id: str) -> str:
    """Get detailed employee information"""
    emp = employees.get(employee_id)
    if emp:
        return f"Employee {employee_id}: {emp['name']}, {emp['position']} in {emp['department']}. Email: {emp['email']}, Phone: {emp['phone']}"
    return "Employee not found."


@mcp.tool()
def search_employees(department: str = None, position: str = None) -> str:
    """Search employees by department or position"""
    results = []
    for emp_id, emp_data in employees.items():
        if (department and emp_data['department'].lower() == department.lower()) or \
                (position and emp_data['position'].lower() == position.lower()):
            results.append(f"{emp_id}: {emp_data['name']} ({emp_data['position']})")

    return "Found employees:\n" + "\n".join(results) if results else "No employees found."


@mcp.tool()
def get_team_members(manager_id: str) -> str:
    """Get team members under a specific manager"""
    team = [f"{emp_id}: {emp_data['name']}" for emp_id, emp_data in employees.items()
            if emp_data['manager'] == manager_id]
    return "Team members:\n" + "\n".join(team) if team else "No team members found."

if __name__ == "__main__":
    # Use the MCP server's built-in SSE app
    app = mcp.sse_app()
    uvicorn.run(app, host="127.0.0.1", port=8002)
