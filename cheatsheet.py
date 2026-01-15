import requests
import json
import sys

# Configuration
BASE_URL = "http://192.168.2.134:31180"
# If the user provided path was full, we might need to adjust, but usually /v1/models lives at the root or under /v1
# We will try a few variations.

def get_api_key():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input("Please enter your API Key: ").strip()

def check_models(base_url, api_key):
    paths = [
        "/v1/models",
        "/models",
        "/api/v1/models"
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"\n--- Checking for Models at {base_url} ---")

    found_models = []

    for path in paths:
        url = f"{base_url}{path}"
        try:
            print(f"Trying GET {url} ...")
            response = requests.get(url, headers=headers, timeout=5)

            # Print Headers for Provider Clues
            print("   Headers:", dict(response.headers))

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Found models endpoint at {path}")

                # Try to parse standard OpenAI format or direct list
                if 'data' in data:
                    for model in data['data']:
                        model_id = model.get('id', 'unknown')
                        print(f"   - Model found: {model_id}")
                        found_models.append(model_id)
                else:
                    print("   Response JSON (structure unknown):")
                    print(json.dumps(data, indent=2))

                return found_models
            elif response.status_code == 401:
                print("❌ 401 Unauthorized. Check your API Key.")
                return []
            else:
                print(f"❌ Failed with status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    return found_models

def test_chat(base_url, api_key, model_name):
    url = f"{base_url}/chat/completions"
    # Also try with /v1/ if the base didn't have it
    if "/v1" not in base_url and requests.post(url, headers={"Authorization": f"Bearer {api_key}"}).status_code == 404:
         url = f"{base_url}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, are you working?"}],
        "max_tokens": 50
    }

    print(f"\n--- Testing Chat Completion with model '{model_name}' ---")
    print(f"POST {url}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            print("✅ Success! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_embedding(base_url, api_key, model_name="text-embedding-ada-002"):
    url = f"{base_url}/embeddings"
    # Also try with /v1/ if the base didn't have it
    if "/v1" not in base_url and requests.post(url, headers={"Authorization": f"Bearer {api_key}"}).status_code == 404:
         url = f"{base_url}/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "input": "The food was delicious and the waiter..."
    }

    print(f"\n--- Testing Embeddings with model '{model_name}' ---")
    print(f"POST {url}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Success! Response:")
            # Use safe printing for potentially large embedding vectors
            data = response.json()
            if 'data' in data and len(data['data']) > 0 and 'embedding' in data['data'][0]:
                vec = data['data'][0]['embedding']
                print(f"   Embedding vector length: {len(vec)}")
                print(f"   First 5 values: {vec[:5]}")
            else:
                print(json.dumps(data, indent=2))
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def detect_provider(base_url):
    print(f"\n--- Detecting Provider at {base_url} ---")
    try:
        # Check Root
        resp = requests.get(base_url, timeout=5)
        print(f"GET / status: {resp.status_code}")
        print(f"GET / content: {resp.text[:100]}...")
        if "Ollama is running" in resp.text:
            print("🔍 Detected Provider: Ollama")
        elif "Swagger UI" in resp.text or "FastAPI" in resp.text:
             print("🔍 Detected Provider: FastAPI based (could be vLLM, LocalAI, etc)")

        # Check specific health endpoints
        health_resp = requests.get(f"{base_url}/health", timeout=5)
        if health_resp.status_code == 200:
             print(f"GET /health: {health_resp.json()}")

    except Exception as e:
        print(f"⚠️ Could not connect to root: {e}")

def main():
    print("LLM Endpoint Inspector")
    print("----------------------")

    detect_provider(BASE_URL)

    api_key = get_api_key()

    # 1. Try to find models
    models = check_models(BASE_URL, api_key)

    # # 2. Test EVERY model for embedding capability
    # print("\n--- 🕵️ Hunting for Embedding Models within available models ---")
    # if models:
    #     for m in models:
    #         test_embedding(BASE_URL, api_key, m)
    # else:
    #     test_embedding(BASE_URL, api_key, "text-embedding-ada-002")

    #3. If models found, test the first one for chat
    if models:
        for model in models:
            test_chat(BASE_URL, api_key, model)

    else:
        print("\nCould not list models automatically.")
        retry = input("Do you want to try a manual model name (e.g. 'gpt-3.5-turbo', 'llama3')? [y/N]: ")
        if retry.lower() == 'y':
            manual_model = input("Enter model name: ")
            test_chat(BASE_URL, api_key, manual_model)

if __name__ == "__main__":
    main()

#    - Model found: llama3.3:70b
#    - Model found: ibm/granite-docling:258m
#    - Model found: llama3:8b
#    - Model found: qwen3:8b
#    - Model found: deepseek-r1:8b
#    - Model found: gpt-oss:120b