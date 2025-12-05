import httpx
import json
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("nodo-tools")

async def get_info(question: str, duan_id: str) -> str:
    """
    Lấy thông tin dự án từ API RAG dựa trên duan_id.
    """
    if not duan_id:
        return "Không tìm thấy thông tin dự án (thiếu ID dự án)."

    url = "http://13.251.189.33:8117/ask/category"
    category_filter = {"project_id": duan_id}
    
    payload = {
        "question": question,
        "top_k": 5,
        "table_name": "nodo_file_emb",
        "category": json.dumps(category_filter),
        "rerank": 3
    }

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.post(url, data=payload)
        
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
            return f"Lỗi API: {response.status_code} - {response.text[:100]}"
            
    except Exception as e:
        logger.error(f"Error in get_info: {e}")
        return f"Lỗi hệ thống khi lấy thông tin: {str(e)}"

async def ask_other_project(question: str) -> str:
    """
    Hỏi thông tin về các dự án khác (ngoài dự án hiện tại).
    """
    url = "http://13.251.189.33:5113/ask"
    payload = {
        "question": question,
        "top_k": 5,
        "table_name": "NODO0711",
        "category": "string",
        "rerank": 3
    }
    
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.post(url, data=payload)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            context_content = "\n---\n".join([
                f"Tài liệu {i+1} (score: {r.get('score')}):\n{r.get('content')}"
                for i, r in enumerate(results)
                if r.get('content')
            ])
            return f"Đã tìm thấy thông tin: {context_content}" if context_content else "Không tìm thấy thông tin."
        else:
            return f"Lỗi API: {response.status_code}"
    except Exception as e:
        logger.error(f"Error in ask_other_project: {e}")
        return f"Lỗi hệ thống: {str(e)}"

async def get_apartment_info(
    duan_id: str,
    location: str = "",
    property_type: str = "",
    area: str = "",
    direction: str = "",
    purpose: str = "",
    num_rooms: str = "",
    price: str = "",
    num_floors: str = ""
) -> str:
    """
    Tìm kiếm thông tin căn hộ chi tiết.
    """
    # Helper function để parse số
    def parse_number(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            normalized = re.sub(r"[,\s]", "", s)
            m = re.search(r"[-+]?\d*\.?\d+", normalized)
            if m:
                return float(m.group(0))
        except Exception:
            return None
        return None

    area_val = parse_number(area)
    price_val = parse_number(price)
    num_rooms_val = parse_number(num_rooms)
    num_floors_val = parse_number(num_floors)

    url = "http://13.251.189.33:8117/apartments/search/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    
    all_params = {
        "vi_tri": location,
        "loai_hinh": property_type,
        "dien_tich": area_val,
        "huong": direction,
        "muc_dich_mua": purpose,
        "so_phong": int(num_rooms_val) if num_rooms_val is not None else None,
        "gia": price_val,
        "so_tang": int(num_floors_val) if num_floors_val is not None else None,
        "project_id": duan_id
    }
    
    # Lọc bỏ các param None hoặc rỗng, trừ số 0
    payload = {}
    for key, value in all_params.items():
        if isinstance(value, str) and value != "":
            payload[key] = value
        elif value is not None:
            payload[key] = value
            
    # Xóa project_id nếu nó là None (để tránh lỗi API nếu API strict)
    if payload.get("project_id") is None:
        payload.pop("project_id", None)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            mota_apart = ""
            data = response.json()
            mo_ta_list = data.get("mo_ta_list", [])
            for i, mo_ta in enumerate(mo_ta_list, start=1):
                mota_apart += f"Căn hộ {i} có mô tả là {mo_ta} \n"
            
            return mota_apart if mota_apart else "Không tìm thấy căn hộ phù hợp với tiêu chí."
        else:
            return f"Lỗi tìm kiếm căn hộ. Mã lỗi: {response.status_code}"
    except Exception as e:
        logger.error(f"Error in get_apartment_info: {e}")
        return f"Lỗi hệ thống khi tìm căn hộ: {str(e)}"