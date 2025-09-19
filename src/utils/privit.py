import os

# Get Hugging Face token from environment variable
# Set HUGGINGFACE_TOKEN in your .env file or environment
DONT_SCRAPE_ME = os.getenv('HUGGINGFACE_TOKEN', 'YOUR_HUGGINGFACE_TOKEN_HERE')