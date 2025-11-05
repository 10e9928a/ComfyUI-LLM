"""
LLM Node Implementation for ComfyUI
"""

import requests
import json


class LLMNode:
    """
    A ComfyUI node for calling LLM APIs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://api.deepseek.com/v1/chat/completions",
                    "multiline": False
                }),
                "api_token": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "prompt": ("STRING", {
                    "default": "Hello, how are you?",
                    "multiline": True
                }),
                "model": ("STRING", {
                    "default": "deepseek-chat",
                    "multiline": False
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 32000,
                    "step": 1
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "full_json")
    FUNCTION = "call_llm"
    CATEGORY = "ComfyUI-LLM"
    
    def call_llm(self, api_url, api_token, prompt, model, temperature, max_tokens, system_prompt="You are a helpful assistant."):
        """
        Call LLM API and return the response
        
        Args:
            api_url: The API endpoint URL
            api_token: API authentication token
            prompt: User input prompt
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: System prompt for the LLM
            
        Returns:
            tuple: (response_text, full_json_response)
        """
        try:
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
            
            # Prepare request body (OpenAI-compatible format)
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make API request
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            
            # Extract the response text (OpenAI format)
            if "choices" in response_json and len(response_json["choices"]) > 0:
                response_text = response_json["choices"][0]["message"]["content"]
            else:
                response_text = "No response generated"
            
            # Return both the text and full JSON
            full_json = json.dumps(response_json, indent=2, ensure_ascii=False)
            
            return (response_text, full_json)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API Request Error: {str(e)}"
            return (error_msg, json.dumps({"error": error_msg}))
        except json.JSONDecodeError as e:
            error_msg = f"JSON Parse Error: {str(e)}"
            return (error_msg, json.dumps({"error": error_msg}))
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            return (error_msg, json.dumps({"error": error_msg}))


# Node registration
NODE_CLASS_MAPPINGS = {
    "LLMNode": LLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMNode": "LLM API Call"
}
