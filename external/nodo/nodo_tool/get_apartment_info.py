import json
import logging
from typing import Optional
import httpx
import re
import requests
from config.settings import settings


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class GetApartmentInfo:
    async def get_apartment_info(
        self,
        location: str = "",
        property_type: str = "",
        area: str = "",
        direction: str = "",
        purpose: str = "",
        num_rooms: str = "",
        price: str = "",
        num_floors: str = "",
        project_id: str = ""
    ) -> str:
            """
            Lấy thông tin liên quan từ cơ sở kiến thức bằng cách gọi API RAG.
            Sử dụng tool này khi khách hàng hỏi những câu hỏi về chi tiết căn hộ trong dự án hoặc bạn muốn giới thiệu căn hộ cho khách hàng.
            Phải nhập toàn bộ tham số, những tham số nào không được khách hàng nhắc đến thì dùng chuỗi "" thay thế.

            Tham số:
            location: Căn hộ thuộc thành phố nào.
            property_type: Loại hình dự án (ví dụ: thấp tầng hay cao tầng)
            area: Diện tích căn hộ theo m2(ví dụ: 145 )
            direction: Hướng căn hộ (ví dụ: Đông, Tây, Nam , Bắc)
            purpose: Mục đích sử dụng (ví dụ: cho thuê, đầu tư, để ở)
            num_rooms: số phòng trong căn hộ (ví dụ: 1,3,4,...)
            price: giá căn hộ (ví dụ: 64302628767)
            num_floors: số tầng của căn hộ (ví dụ: 1,3,5,2,...)
            """
            def parse_number(s: Optional[str]) -> Optional[float]:
                if not s:
                    return None
                try:
                    normalized = re.sub(r"[,\s]", "", s)
                    m = re.search(r"[-+]?\d*\.?\d+", normalized)
                    if m:
                        val = m.group(0)
                        return float(val)
                except Exception:
                    return None
                return None
            area_val = parse_number(area)
            price_val = parse_number(price)
            num_rooms_val = parse_number(num_rooms)
            num_floors_val = parse_number(num_floors)
            url = settings.dbnodo.apartmentdb
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
                "project_id": project_id
            }
            
            
            payload = {}
            for key, value in all_params.items():
                if isinstance(value, str) and value != "":
                    payload[key] = value
                elif value is not None:
                    payload[key] = value
                    
            if all_params.get("project_id") is None:
                payload.pop("project_id", None)
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    mota_apart=""
                    data = response.json()
                    mo_ta_list = data.get("mo_ta_list", [])
                    for i, mo_ta in enumerate(mo_ta_list, start=1):
                        mota=f"""Căn hộ {i} có mô tả là {mo_ta} \n"""
                        mota_apart+=mota
                    return mota_apart
                else:
                    error_details = response.text 
                    return f"Đã xảy ra lỗi khi tìm kiếm thông tin. Mã lỗi: {response.status_code}. Chi tiết từ API: {error_details[:200]}..."

            except httpx.ConnectTimeout:
                return "Đã hết thời gian chờ kết nối đến máy chủ RAG. Vui lòng thử lại sau."
            except httpx.ConnectError:
                return "Không thể kết nối đến máy chủ RAG. Vui lòng kiểm tra lại địa chỉ."
    
gaitool = GetApartmentInfo()