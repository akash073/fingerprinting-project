from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-agent-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 25°C."

@mcp.tool()
def search(query: str) -> str:
    """Search for information on a topic."""
    return f"Results for '{query}': This is a placeholder search result."

if __name__ == "__main__":
    mcp.run()