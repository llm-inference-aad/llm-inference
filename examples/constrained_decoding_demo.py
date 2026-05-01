#!/usr/bin/env python3
"""
Examples of using constrained decoding with the LLM server.

This script demonstrates three types of constraints:
1. JSON Schema - Force JSON output with specific structure
2. Grammar (LBNF) - Enforce custom output format
3. Regex - Pattern matching constraints
"""

import requests
import json
import time
from typing import Optional

# Server configuration
SERVER_URL = "http://localhost:8001"


class ConstrainedLLMClient:
    """Client for using constrained decoding with the LLM server."""
    
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        self.generate_endpoint = f"{server_url}/generate"
        self.rag_endpoint = f"{server_url}/rag/context"
    
    def generate_with_json_schema(
        self,
        prompt: str,
        json_schema: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_rag: bool = False,
    ) -> dict:
        """
        Generate output constrained to match a JSON schema.
        
        Args:
            prompt: The input prompt
            json_schema: JSON schema dict defining the output structure
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with 'generated_text' and other metadata
        """
        prompt_to_send = prompt
        if use_rag:
            try:
                resp = requests.get(self.rag_endpoint, params={"query": prompt[:512]})
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("enabled") and data.get("context"):
                        prompt_to_send = f"Relevant context:\n{data['context']}\n\n{prompt}"
            except Exception:
                pass

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "constraint_type": "json",
            "json_schema": json_schema,
        }
        
        print(f"\n📋 JSON Schema Constraint Request")
        print(f"   Prompt: {prompt[:50]}...")
        print(f"   Schema keys: {list(json_schema.get('properties', {}).keys())}")
        
        start_time = time.time()
        response = requests.post(self.generate_endpoint, json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success ({elapsed:.2f}s)")
            return result
        else:
            print(f"   ❌ Error: {response.status_code}")
            return {"error": response.text}
    
    def generate_with_grammar(
        self,
        prompt: str,
        grammar: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_rag: bool = False,
    ) -> dict:
        """
        Generate output constrained to match an LBNF grammar.
        
        Args:
            prompt: The input prompt
            grammar: LBNF grammar string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with 'generated_text' and other metadata
        """
        prompt_to_send = prompt
        if use_rag:
            try:
                resp = requests.get(self.rag_endpoint, params={"query": prompt[:512]})
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("enabled") and data.get("context"):
                        prompt_to_send = f"Relevant context:\n{data['context']}\n\n{prompt}"
            except Exception:
                pass

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "constraint_type": "grammar",
            "constraint": grammar,
        }
        
        print(f"\n📜 Grammar Constraint Request")
        print(f"   Prompt: {prompt[:50]}...")
        
        start_time = time.time()
        response = requests.post(self.generate_endpoint, json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success ({elapsed:.2f}s)")
            return result
        else:
            print(f"   ❌ Error: {response.status_code}")
            return {"error": response.text}
    
    def generate_with_regex(
        self,
        prompt: str,
        pattern: str,
        max_new_tokens: int = 256,
        temperature: float = 0.5,
        use_rag: bool = False,
    ) -> dict:
        """
        Generate output constrained to match a regex pattern.
        
        Args:
            prompt: The input prompt
            pattern: Regex pattern string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with 'generated_text' and other metadata
        """
        prompt_to_send = prompt
        if use_rag:
            try:
                resp = requests.get(self.rag_endpoint, params={"query": prompt[:512]})
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("enabled") and data.get("context"):
                        prompt_to_send = f"Relevant context:\n{data['context']}\n\n{prompt}"
            except Exception:
                pass

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "constraint_type": "regex",
            "constraint": pattern,
        }
        
        print(f"\n🔍 Regex Constraint Request")
        print(f"   Prompt: {prompt[:50]}...")
        print(f"   Pattern: {pattern}")
        
        start_time = time.time()
        response = requests.post(self.generate_endpoint, json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success ({elapsed:.2f}s)")
            return result
        else:
            print(f"   ❌ Error: {response.status_code}")
            return {"error": response.text}
    
    def generate_unconstrained(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_rag: bool = False,
    ) -> dict:
        """Generate without constraints for benchmarking."""
        prompt_to_send = prompt
        if use_rag:
            try:
                resp = requests.get(self.rag_endpoint, params={"query": prompt[:512]})
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("enabled") and data.get("context"):
                        prompt_to_send = f"Relevant context:\n{data['context']}\n\n{prompt}"
            except Exception:
                pass

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.8,
        }
        
        print(f"\n🔓 Unconstrained Request")
        print(f"   Prompt: {prompt[:50]}...")
        
        start_time = time.time()
        response = requests.post(self.generate_endpoint, json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success ({elapsed:.2f}s)")
            return result
        else:
            print(f"   ❌ Error: {response.status_code}")
            return {"error": response.text}


def example_1_sentiment_analysis():
    """Example 1: Sentiment analysis with structured JSON output."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Sentiment Analysis with JSON Schema")
    print("="*70)
    
    client = ConstrainedLLMClient()
    
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            },
            "key_phrases": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["sentiment", "confidence"]
    }
    
    prompt = "Analyze the sentiment: 'I absolutely love this product! Best purchase ever.'"
    result = client.generate_with_json_schema(prompt, schema, max_new_tokens=256)
    
    if "generated_text" in result:
        try:
            parsed = json.loads(result["generated_text"])
            print(f"\n   Parsed Output:")
            print(f"   - Sentiment: {parsed.get('sentiment')}")
            print(f"   - Confidence: {parsed.get('confidence', 'N/A')}")
            print(f"   - Key phrases: {parsed.get('key_phrases', [])}")
        except json.JSONDecodeError:
            print(f"   Raw output: {result['generated_text'][:200]}")


def example_2_structured_list():
    """Example 2: Generate structured list output."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Structured List with Grammar Constraint")
    print("="*70)
    
    client = ConstrainedLLMClient()
    
    # Simple LBNF grammar for bullet list
    grammar = r"""
root   ::= ("- " item "\n")+
item   ::= [^\n]{1,100}
"""
    
    prompt = "List three benefits of machine learning in a bullet list format."
    result = client.generate_with_grammar(prompt, grammar, max_new_tokens=200)
    
    if "generated_text" in result:
        print(f"\n   Output:")
        for line in result["generated_text"].strip().split("\n"):
            if line.strip():
                print(f"   {line}")


def example_3_color_code():
    """Example 3: Generate hex color codes."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Hex Color Code with Regex")
    print("="*70)
    
    client = ConstrainedLLMClient()
    
    # Regex for hex color codes
    pattern = r"^#[0-9a-fA-F]{6}$"
    
    prompt = "Generate a hex color code for professional blue."
    result = client.generate_with_regex(prompt, pattern, max_new_tokens=10)
    
    if "generated_text" in result:
        print(f"   Generated Color Code: {result['generated_text']}")


def example_4_key_value_pairs():
    """Example 4: Generate key-value pair structured output."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Key-Value Pairs with Grammar")
    print("="*70)
    
    client = ConstrainedLLMClient()
    
    # Grammar for key=value format
    grammar = r"""
root   ::= (line "\n")*
line   ::= key " = " value
key    ::= [a-z_]{1,20}
value  ::= [a-zA-Z0-9_ ,\.\!\?]{1,100}
"""
    
    prompt = "Extract information as key=value pairs: John is 25 years old and works as a software engineer."
    result = client.generate_with_grammar(prompt, grammar, max_new_tokens=200)
    
    if "generated_text" in result:
        print(f"\n   Output:")
        print(result["generated_text"])


def example_5_performance_comparison():
    """Example 5: Compare performance with and without constraints."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Performance Comparison")
    print("="*70)
    
    client = ConstrainedLLMClient()
    prompt = "Generate a JSON object with fields: name, age, email"
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        },
        "required": ["name", "age", "email"]
    }
    
    # Benchmark unconstrained
    print("\n   Running 3 iterations for benchmark...")
    times_unconstrained = []
    for i in range(3):
        start = time.time()
        client.generate_unconstrained(prompt, max_new_tokens=200)
        times_unconstrained.append(time.time() - start)
    
    # Benchmark constrained
    times_constrained = []
    for i in range(3):
        start = time.time()
        client.generate_with_json_schema(prompt, schema, max_new_tokens=200)
        times_constrained.append(time.time() - start)
    
    avg_unconstrained = sum(times_unconstrained) / len(times_unconstrained)
    avg_constrained = sum(times_constrained) / len(times_constrained)
    overhead = ((avg_constrained - avg_unconstrained) / avg_unconstrained) * 100
    
    print(f"\n   Results:")
    print(f"   - Unconstrained avg: {avg_unconstrained:.2f}s")
    print(f"   - Constrained avg: {avg_constrained:.2f}s")
    print(f"   - Overhead: {overhead:.1f}%")


def main():
    """Run all examples."""
    print("\n" + "🚀 "*20)
    print("CONSTRAINED DECODING EXAMPLES")
    print("🚀 "*20)
    
    try:
        # Check server is running
        requests.get(f"{SERVER_URL}/")
        print("✅ Server is running and accessible")
    except requests.ConnectionError:
        print("❌ Server not found at", SERVER_URL)
        print("   Please start the server: python server.py")
        return
    
    try:
        example_1_sentiment_analysis()
        example_2_structured_list()
        example_3_color_code()
        example_4_key_value_pairs()
        example_5_performance_comparison()
        
        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
