# """
# Optimized LLM Service Module
# - Singleton pattern for shared LLM instance
# - Environment variable configuration
# - Async-first design
# - Error handling with retry logic
# - Token usage tracking
# """
# import asyncio
# import os
# import time
# from typing import Optional, Any
# from uuid import uuid4
# from logging import getLogger, ERROR

# from langchain.agents import create_agent
# from langchain_openai import ChatOpenAI
# from pydantic import SecretStr
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# from livekit.agents import llm
# from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN
# from dotenv import load_dotenv

# from llm.metrics import LLMMetrics, log_llm_metrics

# load_dotenv()
# logger = getLogger(__name__)
# logger.setLevel(ERROR)


# class LLMService:
#     """
#     Optimized LLM Service with singleton pattern and connection reuse.
#     """
#     _instance: Optional["LLMService"] = None
#     _lock = asyncio.Lock()

#     def __init__(self, api_key: str, model: str, system_prompt: Optional[str] = None):
#         self.api_key = api_key
#         self.model_name = model
        
#         # Initialize ChatOpenAI with optimized settings
#         self.llm = ChatOpenAI(
#             model=model,
#             api_key=SecretStr(api_key),
#             streaming=True,
#             timeout=60,
#             max_retries=2,
#         )
        
#         # Default system prompt for voice assistant
#         default_system_prompt = """
#         Bạn là một bộ não của một robot biết nói tiếng Việt, đảm nhiệm vai trò nhân viên chăm sóc khách hàng.
#         Phải gọi user là anh chị và xưng hô là em.

#         [GOAL]
#         - Hiểu đúng mục đích của người dùng (intent).
#         - Chọn đúng công cụ nếu cần.
#         - Đảm bảo quy trình thực thi hiệu quả, không dư thừa.

#         [OUTPUT]
#         Ngắn gọn nhưng đầy đủ ý nghĩa. Vì bạn sẽ trả lời người dùng bằng giọng nói:
#         - Không chứa ký tự đặc biệt: ( , ), *, &, #, @, ...
#         - Không viết tắt: "Anh/Chị" => "Anh chị"
#         - Số chuyển thành chữ: 1 => Một, 15 => Mười lăm, 850.000 => Tám trăm năm mươi nghìn

#         [IMPORTANT]
#         - Luôn trả lời có cấu trúc, rõ ràng, giữ giọng điệu trung lập và chuyên nghiệp.
#         - Bắt đầu câu trả lời bằng câu thoại ngắn như: "Dạ anh chị!".
#         """
        
#         self.agent = create_agent(
#             self.llm,
#             system_prompt=system_prompt or default_system_prompt,
#         )
        
#         logger.info(f"LLMService initialized with model: {model}")

#     @classmethod
#     async def get_instance(cls) -> "LLMService":
#         """Get or create singleton instance."""
#         async with cls._lock:
#             if cls._instance is None:
#                 # base_url = os.getenv("LLM_BASE_URL", "http://61.206.39.5:16671/v1")
#                 api_key = os.getenv("OPENAI_API_KEY", "")
#                 model = os.getenv("OPENAI_GPT_MODEL", "")
                
#                 if not api_key:
#                     raise ValueError("OPENAI_API_KEY environment variable must be set")
                
#                 cls._instance = cls(api_key, model)
#             return cls._instance

#     @classmethod
#     def reset_instance(cls):
#         """Reset singleton instance (useful for testing or reconfiguration)."""
#         cls._instance = None

#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_exponential(multiplier=1, min=1, max=10),
#         retry=retry_if_exception_type((ConnectionError, TimeoutError)),
#         reraise=True
#     )
#     async def generate(self, messages: list[dict[str, str]], max_tokens: int = 512):
#         """
#         Stream responses from the LLM with retry logic.
        
#         Args:
#             messages: List of message dicts with 'role' and 'content'
#             max_tokens: Maximum tokens to generate
            
#         Yields:
#             str: Text chunks as they arrive
#         """
#         request_id = str(uuid4())[:8]
#         start_time = time.time()
#         first_chunk_time: Optional[float] = None
#         total_tokens = 0
#         prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4  # Rough estimate

#         try:
#             async for chunk in self.agent.astream(  # type: ignore[arg-type]
#                 {"messages": messages},  # type: ignore[arg-type]
#                 stream_mode="messages"
#             ):
#                 # Handle different response types from langgraph
#                 if hasattr(chunk, '__iter__') and not isinstance(chunk, (str, dict)):
#                     token, metadata = chunk
#                 else:
#                     token = chunk
#                     metadata = None
                
#                 # Extract content from response
#                 delta = None
#                 if hasattr(token, 'content_blocks'):
#                     content_blocks = getattr(token, 'content_blocks', None)
#                     if content_blocks and isinstance(content_blocks, list) and len(content_blocks) > 0:
#                         first_block = content_blocks[0]
#                         if isinstance(first_block, dict) and first_block.get("type") == "text":
#                             delta = first_block.get("text")
#                 elif hasattr(token, 'content'):
#                     delta = getattr(token, 'content', None)
#                 elif isinstance(token, str):
#                     delta = token
                
#                 if not delta:
#                     continue
                
#                 if first_chunk_time is None:
#                     first_chunk_time = time.time() - start_time
                
#                 total_tokens += 1
#                 yield delta

#             total_time = time.time() - start_time
#             tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
#             # Log metrics
#             metrics = LLMMetrics(
#                 request_id=request_id,
#                 model=self.model_name,
#                 prompt_tokens=prompt_tokens,
#                 completion_tokens=total_tokens,
#                 total_tokens=prompt_tokens + total_tokens,
#                 first_token_latency_ms=(first_chunk_time or 0) * 1000,
#                 total_latency_ms=total_time * 1000,
#                 tokens_per_second=tokens_per_second,
#                 success=True
#             )
#             log_llm_metrics(metrics)
            
#         except Exception as e:
#             total_time = time.time() - start_time
#             metrics = LLMMetrics(
#                 request_id=request_id,
#                 model=self.model_name,
#                 prompt_tokens=prompt_tokens,
#                 completion_tokens=total_tokens,
#                 total_tokens=prompt_tokens + total_tokens,
#                 first_token_latency_ms=(first_chunk_time or 0) * 1000,
#                 total_latency_ms=total_time * 1000,
#                 tokens_per_second=0,
#                 success=False,
#                 error=str(e)
#             )
#             log_llm_metrics(metrics)
#             logger.error(f"LLM generation error: {e}")
#             raise


# class MyLLM(llm.LLM):
#     """LiveKit-compatible LLM wrapper with optimized performance."""
    
#     _shared_service: Optional[LLMService] = None
#     _service_lock = asyncio.Lock()
    
#     def __init__(self) -> None:
#         super().__init__()
    
#     @classmethod
#     async def _get_service(cls) -> LLMService:
#         """Get shared LLMService instance."""
#         async with cls._service_lock:
#             if cls._shared_service is None:
#                 cls._shared_service = await LLMService.get_instance()
#             return cls._shared_service

#     def chat(
#         self,
#         *,
#         chat_ctx: llm.ChatContext,
#         tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
#         conn_options=DEFAULT_API_CONNECT_OPTIONS,
#         parallel_tool_calls=NOT_GIVEN,
#         tool_choice=NOT_GIVEN,
#         extra_kwargs=NOT_GIVEN,
#     ) -> llm.LLMStream:
#         return MyLLMChunkStream(
#             llm_instance=self,
#             chat_ctx=chat_ctx,
#             tools=tools,
#             conn_options=conn_options
#         )


# class MyLLMChunkStream(llm.LLMStream):
#     """Optimized LLM stream with shared service and proper error handling."""
    
#     def __init__(
#         self,
#         *,
#         llm_instance: MyLLM,
#         chat_ctx: llm.ChatContext,
#         tools: list | None,
#         conn_options
#     ):
#         super().__init__(llm=llm_instance, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)
#         self._llm = llm_instance

#     async def _run(self):
#         """Run LLM inference with optimized streaming."""
#         try:
#             # Get shared service instance
#             service = await MyLLM._get_service()
            
#             # Convert chat context to messages
#             messages = []
#             for msg in self._chat_ctx.items:
#                 # Handle different message types - skip function calls
#                 if not hasattr(msg, 'content') or not hasattr(msg, 'role'):
#                     continue
                    
#                 content = getattr(msg, 'content', '')
#                 if isinstance(content, list) and len(content) > 0:
#                     content = content[0]
#                 role = getattr(msg, 'role', 'user')
#                 messages.append({"role": str(role), "content": str(content)})
            
#             logger.debug(f"LLM request with {len(messages)} messages")
            
#             # Stream response chunks
#             async for chunk_text in service.generate(messages):
#                 chunk = llm.ChatChunk(
#                     id=str(uuid4()),
#                     delta=llm.ChoiceDelta(content=chunk_text, role="assistant"),
#                     usage=llm.CompletionUsage(
#                         completion_tokens=1,
#                         prompt_tokens=0,
#                         prompt_cached_tokens=0,
#                         total_tokens=1,
#                     )
#                 )
#                 self._event_ch.send_nowait(chunk)
                
#         except asyncio.CancelledError:
#             logger.debug("LLM stream cancelled")
#             raise
#         except Exception as e:
#             logger.error(f"LLM stream error: {e}", exc_info=True)
#             raise


