import json
import logging
import httpx
from config.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class GetConvesationInfo:
    async def get_conversation_history_by_phone(self, caller_number: str) -> str:
            """
            Lấy lịch sử hội thoại đã lưu trong CSDL dựa theo số điện thoại để cung cấp context cho LLM hiểu rõ cuộc hội thoại đang diễn ra.
            Sử dụng tool này khi có cuộc gọi đến từ khách hàng đã từng tương tác trước đó.

            Tham số:
            caller_number (str): Số điện thoại của khách hàng đã từng tương tác.
            """
            url = settings.dbnodo.history
            # Các tham số cần thiết cho API của bạn.
            payload = {
                "phone": caller_number,
                "limit": 5,
                #"order": "asc"
            }
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.post(url, data=payload, timeout=20.0)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])

                    unique_notes = list(dict.fromkeys(
                    note.strip()
                    for note in results
                    if note and note.strip()
                    ))

                    context_content = "\n".join(
                    f"- {note}"
                    for note in unique_notes
                    )
                    if context_content:
                        return f"Đã tìm thấy thông tin: {context_content}"
                    else:
                        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
                else:
                    error_details = response.text 
                    return f"Đã xảy ra lỗi khi tìm kiếm thông tin. Mã lỗi: {response.status_code}. Chi tiết từ API: {error_details[:200]}..."

            except httpx.ConnectTimeout:
                return "Đã hết thời gian chờ kết nối đến máy chủ RAG. Vui lòng thử lại sau."
            except httpx.ConnectError:
                return "Không thể kết nối đến máy chủ RAG. Vui lòng kiểm tra lại địa chỉ."


gci = GetConvesationInfo()