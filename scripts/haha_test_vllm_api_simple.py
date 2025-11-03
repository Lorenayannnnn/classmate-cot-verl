#!/usr/bin/env python3
"""
Simple test script to verify vLLM API server is working.

This script tests the vLLM API endpoints without requiring the full ClassmateWorker setup.

Usage:
    python test_vllm_api_simple.py [host] [port] [model_name]

Example:
    python test_vllm_api_simple.py 127.0.0.1 8000 classmate-model-0
"""

import sys
import requests
import json


def test_vllm_api(host="127.0.0.1", port=8000, model_name="classmate-model-0"):
    """Test vLLM API server functionality."""
    
    base_url = f"http://{host}:{port}"
    print(f"Testing vLLM API at {base_url}")
    print(f"Model: {model_name}")
    print("-" * 50)
    
    try:
        # Test 1: Check server health
        print("1. Testing server health...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ Server is healthy")
        else:
            print(f"   ⚠ Health check returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    try:
        # Test 2: List available models
        print("2. Listing available models...")
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"   ✓ Found {len(models.get('data', []))} models:")
            for model in models.get('data', []):
                print(f"     - {model.get('id', 'unknown')}")
        else:
            print(f"   ❌ Model list failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Model list failed: {e}")
        return False
    
    try:
        # Test 3: Generate completion
        print("3. Testing text completion...")
        batch_prompts = [
            "Hello, how are you?",
            "Once upon a time in a land far away,",
            "The quick brown fox jumps over the lazy dog."
        ]
        data = {
            "model": model_name,
            "prompt": batch_prompts,
            "max_tokens": 20,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1
        }
        
        response = requests.post(
            f"{base_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                for choice in choices:
                    print(f"Input prompt: {batch_prompts[choice.get('index', 0)]}")
                    print(f"Output continuation: {choice.get('text', '')}")
            else:
                print(f"   ⚠ No choices in response: {result}")
        else:
            print(f"   ❌ Generation failed with status {response.status_code}")
            print(f"     Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False
    
    print("-" * 50)
    print("✓ All tests passed! vLLM API is working correctly.")
    return True

def test_openai_client():
    import time
    from openai import OpenAI

    openai_api_key = "EMPTY"  # vLLM's OpenAI-compatible server doesn't require a real API key
    openai_api_base = "http://127.0.0.1:8001/v1"  # Remove /v1 - OpenAI client adds it automatically

    client = OpenAI(
        base_url=openai_api_base,
        api_key=openai_api_key,
    )

    # Example of a batched prompt
    prompts = ["My name is"] * 16

    retry_delay = 5  # seconds
    attempt = 0

    completions = []
    while True:  # Keep trying until success
        attempt += 1
        try:
            print(f"Attempt {attempt}: Making API call...")
            start_time = time.time()
            completion = client.completions.create(model="meta-llama/Llama-3.2-1B-Instruct", prompt=prompts, temperature=0, max_tokens=256, n=1)
            time_needed = time.time() - start_time
            print(f"   <UNK> API call took {time_needed} seconds.")

            print("✅ Completion results:")
            for choice in completion.choices:
                print("--------Start--------")
                print(f"Input prompt: {prompts[choice.index]}")
                print(f"Output continuation: {choice.text}")
                print("--------End--------")
                completions.append(choice.text)
            return  # Success, exit the function

        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            print(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)




if __name__ == "__main__":
    # Parse command line arguments
    # host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    # port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    # model_name = sys.argv[3] if len(sys.argv) > 3 else "classmate-model-0"
    #
    # # Run tests
    # success = test_vllm_api(host, port, model_name)
    #
    # if not success:
    #     sys.exit(1)

    test_openai_client()



#python scripts/haha_test_vllm_api_simple.py 127.0.0.1 8000 meta-llama/Llama-3.2-1B-Instruct
#python scripts/haha_test_vllm_api_simple.py