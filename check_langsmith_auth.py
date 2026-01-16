from langsmith import Client

client = Client()
print("LangSmith Info:")
print(client.info)
print(f"\nWorkspace ID: {client.workspace_id}")
print(f"API URL: {client.api_url}")
