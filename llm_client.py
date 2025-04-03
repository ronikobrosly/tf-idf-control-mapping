"""
Client for interacting with Llama 3.1 API
"""

import os
from openai import OpenAI

class LlamaClient:
    """Client for interacting with Llama 3.1 API."""
    
    def __init__(
        self,
        base_url="https://api.novita.ai/v3/openai",
        model="meta-llama/llama-3.1-70b-instruct",
        max_tokens=1000
    ):
        """Initialize the Llama client."""
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get('llama_api')
        )
        self.model = model
        self.max_tokens = max_tokens
    
    def generate(self, prompt, stream=False):
        """Generate a response from the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                stream=stream,
                max_tokens=self.max_tokens,
                response_format={"type": "text"},
                extra_body={}
            )
            
            if stream:
                full_response = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                return full_response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return f"Error: {str(e)}" 