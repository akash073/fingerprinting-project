from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import asyncio

# Load local model
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def run_model(prompt: str) -> str:
    """Run the local model on a prompt."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer(
        [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
        return_tensors="pt"
    ).to(device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.2, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

def pick_tool(user_input: str, tools: list) -> dict | None:
    """Ask the model to pick a tool and arguments."""
    tool_descriptions = "\n".join(
        [f"- {t.name}: {t.description} | args: {json.dumps({p: '...' for p in t.inputSchema.get('properties', {})})}"
         for t in tools]
    )

    prompt = f"""You are an agent with access to these tools:
{tool_descriptions}

User request: {user_input}

Reply with ONLY a JSON object like:
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}
If no tool is needed, reply with:
{{"tool": null, "args": {{}}}}"""

    response = run_model(prompt)

    # Extract JSON from response
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except Exception:
        return None

async def run_agent(user_input: str):
    """Connect to MCP server and run the agent loop."""
    server_params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch available tools
            tools = (await session.list_tools()).tools
            print(f"Available tools: {[t.name for t in tools]}\n")

            # Let model pick a tool
            decision = pick_tool(user_input, tools)
            print(f"Model decision: {decision}\n")

            if decision and decision.get("tool"):
                # Call the chosen tool
                result = await session.call_tool(decision["tool"], decision["args"])
                tool_output = result.content[0].text
                print(f"Tool result: {tool_output}\n")

                # Final response from model
                final_prompt = f"User asked: {user_input}\nTool '{decision['tool']}' returned: {tool_output}\nGive a helpful reply:"
                final_response = run_model(final_prompt)
                print(f"Agent: {final_response}")
            else:
                # No tool needed, answer directly
                print(f"Agent: {run_model(user_input)}")

if __name__ == "__main__":
    user_input = "What is the weather in Paris?"
    asyncio.run(run_agent(user_input))