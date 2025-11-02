"""
OpenAI Service Module for LeafToHarvest Application

This module handles all OpenAI API interactions with proper error handling
and token usage tracking. It provides a clean interface for making API calls
while automatically logging usage data.
"""

import os
import openai
from typing import Dict, Optional, Tuple
import base64
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Service class for handling OpenAI API interactions with usage tracking.
    
    This class provides methods for making OpenAI API calls while automatically
    tracking token usage, costs, and errors for monitoring purposes.
    """
    
    def __init__(self):
        """Initialize the OpenAI service with API key from environment."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model pricing per 1K tokens (as of 2024)
        # These rates should be updated based on current OpenAI pricing
        self.model_pricing = {
            'gpt-4o': {
                'input': 0.005,   # $0.005 per 1K input tokens
                'output': 0.015   # $0.015 per 1K output tokens
            },
            'gpt-4o-mini': {
                'input': 0.00015,  # $0.00015 per 1K input tokens
                'output': 0.0006   # $0.0006 per 1K output tokens
            },
            'gpt-3.5-turbo': {
                'input': 0.0005,   # $0.0005 per 1K input tokens
                'output': 0.0015  # $0.0015 per 1K output tokens
            }
        }
        
        logger.info(f"OpenAI service initialized with API key: {self.api_key[:20]}...")
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the estimated cost for an API call based on token usage.
        
        Args:
            model: The OpenAI model used
            prompt_tokens: Number of tokens in the input prompt
            completion_tokens: Number of tokens in the response
            
        Returns:
            Estimated cost in USD
        """
        if model not in self.model_pricing:
            logger.warning(f"Unknown model pricing for {model}, using gpt-4o pricing")
            model = 'gpt-4o'
        
        pricing = self.model_pricing[model]
        
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        
        total_cost = input_cost + output_cost
        return round(total_cost, 6)  # Round to 6 decimal places for precision
    
    def analyze_crop_disease(self, image_base64: str, crop_type: str, user_id: int) -> Dict:
        """
        Analyze crop disease using OpenAI's vision capabilities.
        
        Args:
            image_base64: Base64 encoded image data
            crop_type: Type of crop being analyzed
            user_id: ID of the user making the request
            
        Returns:
            Dictionary containing analysis results and usage data
        """
        try:
            # Create crop-specific system prompt
            system_prompt = f"""You are an expert plant pathologist specializing in {crop_type} crops. 
            Analyze this {crop_type} plant image and provide a detailed diagnosis.

            Please provide your response in this exact format:
            DISEASE: [Specific disease name]
            SEVERITY: [Low/Medium/High]
            SYMPTOMS: [Key visible symptoms]
            TREATMENT: [Specific treatment recommendations]
            PREVENTION: [Prevention measures]

            Focus on diseases common to {crop_type} plants and provide specific, actionable advice."""

            # Make the API call
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this {crop_type} plant image for diseases. Provide specific diagnosis and treatment recommendations."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]},
                ],
                max_tokens=600
            )
            
            # Extract usage information
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # Calculate cost
            estimated_cost = self.calculate_cost("gpt-4o", prompt_tokens, completion_tokens)
            
            # Prepare response data
            result = {
                'success': True,
                'response': response.choices[0].message.content,
                'usage_data': {
                    'user_id': user_id,
                    'timestamp': datetime.utcnow(),
                    'model_name': 'gpt-4o',
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'estimated_cost': estimated_cost
                },
                'error': None
            }
            
            logger.info(f"OpenAI API call successful for user {user_id}: {total_tokens} tokens, ${estimated_cost}")
            return result
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded for user {user_id}: {str(e)}")
            return {
                'success': False,
                'response': None,
                'usage_data': {
                    'user_id': user_id,
                    'timestamp': datetime.utcnow(),
                    'model_name': 'gpt-4o',
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'estimated_cost': 0.0
                },
                'error': f"Rate limit exceeded: {str(e)}"
            }
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error for user {user_id}: {str(e)}")
            return {
                'success': False,
                'response': None,
                'usage_data': {
                    'user_id': user_id,
                    'timestamp': datetime.utcnow(),
                    'model_name': 'gpt-4o',
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'estimated_cost': 0.0
                },
                'error': f"API error: {str(e)}"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI service for user {user_id}: {str(e)}")
            return {
                'success': False,
                'response': None,
                'usage_data': {
                    'user_id': user_id,
                    'timestamp': datetime.utcnow(),
                    'model_name': 'gpt-4o',
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'estimated_cost': 0.0
                },
                'error': f"Unexpected error: {str(e)}"
            }
    
    def get_model_pricing(self) -> Dict:
        """
        Get current model pricing information.
        
        Returns:
            Dictionary containing pricing information for all supported models
        """
        return self.model_pricing.copy()
    
    def update_model_pricing(self, model: str, input_price: float, output_price: float):
        """
        Update pricing for a specific model.
        
        Args:
            model: Model name
            input_price: Price per 1K input tokens
            output_price: Price per 1K output tokens
        """
        self.model_pricing[model] = {
            'input': input_price,
            'output': output_price
        }
        logger.info(f"Updated pricing for {model}: input=${input_price}, output=${output_price}")


# Global instance for easy access - initialize lazily
openai_service = None

def get_openai_service():
    """Get or create the OpenAI service instance."""
    global openai_service
    if openai_service is None:
        try:
            openai_service = OpenAIService()
        except ValueError as e:
            print(f"[WARNING] OpenAI service not available: {e}")
            return None
    return openai_service
