import json
import logging
from typing import Optional
import httpx


class GetProjecttInfo:
    async def get_project_info(question: str,project_id:str) -> str:
        """
        Lấy thông tin liên quan từ cơ sở kiến thức bằng cách gọi API RAG.
        Sử dụng tool này khi khách hàng hỏi những câu hỏi về dự án đang giới thiệu.
        Sử dụng tool này khi cần lấy thông tin từ cơ sở dữ liệu.

        Tham số:
        question (str): Câu hỏi của khách hàng cần được tìm kiếm trong cơ sở dữ liệu.
        """
        url = "http://13.251.189.33:8117/ask/category"
        category_filter = {"project_id": project_id}
        category_json_string = json.dumps(category_filter)
        # Các tham số cần thiết cho API của bạn.
        payload = {
            "question": question,
            "top_k": 5,
            "table_name": "nodo_file_emb",
            "category": category_json_string,
            "rerank": 3
        }
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.post(url, data=payload, timeout=20.0)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                context_content = "\n---\n".join([
                    f"Tài liệu {i+1} (score: {r.get('score')}):\n{r.get('content')}"
                    for i, r in enumerate(results)
                    if r.get('content')
                ])
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
    
    
gpji = GetProjecttInfo()