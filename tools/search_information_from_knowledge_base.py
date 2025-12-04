import httpx
import json
import os

from dotenv import load_dotenv
from typing import Any, Dict

load_dotenv()

async def search_information_from_knowledge_base(query: str, table_name: str, rerank: int = 2) -> str:
    """Search the knowledge base for relevant information."""
    knowledge_service_url = os.getenv("VECTOR_STORE_URL", "http://localhost:5113") + "/ask"
    payload = {
        "question": query,
        "table_name": table_name,
        "top_k": 5,
        "rerank": rerank,
        "category": None
    }
    headers = {
       "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        try:
            response = await client.post(knowledge_service_url, data=payload, headers=headers)
            response.raise_for_status()
            if response.status_code != 200:
                return f"Error searching knowledge base: {response.status_code} {response.text}"
            
            knowledge_base: Dict[str, Any] = response.json()
            
            if knowledge_base and knowledge_base.get("results"):
                # Format the results for display
                result_strings = []
                for result in knowledge_base["results"]:
                    result_strings.append(json.dumps(result, indent=2, ensure_ascii=False))
                return "\n".join(result_strings)
            else:
                return "No knowledge found."
        except httpx.RequestError as e:
            return f"Error searching knowledge base: {str(e)}"