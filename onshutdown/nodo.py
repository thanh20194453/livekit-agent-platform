from dotenv import load_dotenv
from livekit.plugins import (
    openai
)
import requests
from datetime import datetime, timedelta
import uuid
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timezone, timedelta
from livekit.agents import ChatContext
import asyncio
import json
# from livekit.plugins import noise_cancellation
load_dotenv()
import psycopg2 
from psycopg2 import sql
DB_HOST = "13.251.189.33"
DB_PORT = 33106
DB_DATABASE = "_5e5899d8398b5f7b"
DB_USER = "root"
DB_PASSWORD = "admin"
DB_PASSWORD_PG= "tks39.*AdFkrq3A9"
DB_PG= "svisor"
DB_PORT_PG= "5111"
DB_USER_PG= "postgres"

TEN_BANG_LEAD = "tabLead"
TEN_BANG_OPPORTUNITY= "tabOpportunity"
TEN_BANG_HISTORY="tabCRM Note"
TEN_BANG_EVENT="tabEvent"
TEN_BANG_EVENT_PARTICIPANT="tabEvent Participants"
API_URL = "http://103.178.231.211:8127/latest-call"

BASE_URL = "https://grand.nodo.vn/rest_api/pu/property/ai/query_object_tree"

HEADERS = {
    # Postman của bạn chỉ có 'accept: */*', nhưng ta thêm 'Content-Type'
    # để chỉ rõ rằng ta đang gửi dữ liệu dạng JSON.
    "accept": "*/*",
    "Content-Type": "application/json" 
}
BASE_URL = "https://grand.nodo.vn/rest_api/pr/property/ai/query_object_tree"
LOGIN_URL = "https://grand.nodo.vn/rest_api/pu/property/auth/login"
class Nodo:
    def lay_token_dang_nhap(self):
        """Thực hiện gọi API đăng nhập và lưu trữ Token."""
        access_token = None
        print(">>> Gọi API 0: Lấy Token Đăng nhập")

        # Payload (Body) cho API Đăng nhập
        payload_login = {
            "login": "admin",
            "password": "admin@123"
        }

        try:
            response = requests.post(
                LOGIN_URL, 
                headers=HEADERS, 
                json=payload_login,
                verify=False 
            )
            
            if response.status_code == 200:
                print("Trạng thái: Đăng nhập THÀNH CÔNG (200)")
                full_response_data = response.json()
                
                # Lấy Access Token từ phản hồi
                access_token = full_response_data.get("data").get("access_token")
                print(access_token)
                
                if access_token:
                    # Cập nhật HEADERS với Authorization Token
                    print("Đã lấy và lưu Token thành công.")
                else:
                    print("LỖI: Không tìm thấy 'access_token' trong phản hồi.")
            else:
                print(f"Trạng thái: Đăng nhập LỖI ({response.status_code})")
                print("Nội dung lỗi:", response.text)

            return access_token
                
        except requests.exceptions.RequestException as e:
            print(f"Đã xảy ra lỗi khi gửi yêu cầu đăng nhập: {e}")
            return None,None
    def format_chat_transcript(self,chat_transcript):

        # Khởi tạo một danh sách trống để lưu trữ các tin nhắn đã được định dạng
        formatted_messages = []

        # Lặp qua từng 'item' (tin nhắn) trong transcript
        for item in chat_transcript.get('items', []):
            role = item.get('role')
            content_list = item.get('content', [])
            
            # Nối các phần nội dung thành một chuỗi tin nhắn. 
            # Chúng ta không cần thay thế "\n" bằng "<br>" nữa vì đây là JSON, không phải HTML.
            message = " ".join(content_list)

            # Xác định vai trò đã được dịch
            if role == 'assistant':
                speaker = "agent"
            elif role == 'user':  # Giả định 'user' là 'khách hàng'
                speaker = "khách hàng"
            else:
                # Xử lý các vai trò không xác định (có thể bỏ qua hoặc gán vai trò mặc định)
                speaker = "unknown"
                
            # Tạo một dictionary cho tin nhắn hiện tại
            message_object = {
                "speaker": speaker,
                "message": message
            }

            # Thêm dictionary này vào danh sách
            formatted_messages.append(message_object)

        # Chuyển danh sách các đối tượng tin nhắn thành một chuỗi JSON
        # 'indent=4' giúp chuỗi JSON dễ đọc hơn (tùy chọn)
        json_output = json.dumps(formatted_messages, indent=4, ensure_ascii=False)
        
        return json_output
    
    def doc_du_lieu(self,caller_number, status_call, summary,chat_transcript_v2,time_start_call,lead_id):
        access_token=self.lay_token_dang_nhap()
        tz_plus7 = timezone(timedelta(hours=7))
        time_end_dt = datetime.now(tz_plus7)
        time_end_formatted = time_end_dt.strftime("%d-%m-%Y %H:%M:%S")
        auth='Bearer '+access_token
        url = 'https://grand.nodo.vn//rest_api/pr/property/lead/call_logs'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': auth,
            'Cookie': 'frontend_lang=en_US; session_id=d20f6a6ec83474bc70d2c86bc080c7a483bf3281'
        }

        payload = {
            "status_call": int(status_call),
            "lead_id": lead_id,
            "time_start_call": time_start_call,
            "time_end_call": time_end_formatted,
            "note": summary,
            "context_call_json": chat_transcript_v2, 
            "bot_name": "AI Assistant"
        }
        response = requests.post(url, headers=headers, json=payload)
        print(response)

    def luu_vao_postgres(self,caller_number, status_call, summary,chat_transcript_v2,time_start_call,lead_id):
        """Lưu dữ liệu cuộc gọi vào bảng nodo_outbound.nodo_history."""
        conn = None
        try:
            # 1. Thiết lập kết nối
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT_PG,
                database=DB_PG,
                user=DB_USER_PG,
                password=DB_PASSWORD_PG
            )
            cur = conn.cursor()

            # 2. Xây dựng lệnh SQL INSERT
            # Chú ý sử dụng %s placeholders để tránh SQL Injection
            insert_query = sql.SQL("""
                INSERT INTO nodo_outbound.nodo_history (
                    id, caller_number, status_call, note, context_call_json, 
                    time_start_call, lead_id, time_end_call
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
            """)
            
            # Tạo thời điểm kết thúc cuộc gọi cho bản ghi DB
            tz_plus7 = timezone(timedelta(hours=7))
            time_end_dt = datetime.now(tz_plus7)
            time_end_formatted = time_end_dt.strftime("%Y-%m-%d %H:%M:%S")

            # 3. Chuẩn bị dữ liệu để chèn
            # Đảm bảo thứ tự dữ liệu khớp với thứ tự các cột trong câu lệnh INSERT
            data_to_insert = (
                str(uuid.uuid4()),
                caller_number,
                status_call,
                summary,
                chat_transcript_v2,
                time_start_call,
                lead_id,
                time_end_formatted
            )
            
            # 4. Thực thi và Commit
            cur.execute(insert_query, data_to_insert)
            conn.commit()
            print("\n[DB] Đã lưu dữ liệu vào PostgreSQL thành công.")

        except (Exception, psycopg2.Error) as error:
            print(f"\n[DB] Lỗi khi kết nối hoặc ghi dữ liệu PostgreSQL: {error}")
        finally:
            # 5. Đóng kết nối
            if conn:
                cur.close()
                conn.close()
    async def analyze_and_save_opportunity(self,room, session,caller_number,time_start_call,lead_id):
        """
        Phân tích lịch sử cuộc gọi và lưu thông tin cơ hội vào DB.
        """
        llm = openai.LLM(model="gpt-4o-mini")
        # 1. Đọc lịch sử trò chuyện
        history = session.history
        
        # chat_transcript = str(history.to_dict())
        # print(chat_transcript)
        items_list = history.to_dict().get('items', [])
        filtered_items = []
        for item in items_list:
            # Chúng ta chỉ quan tâm đến các mục có 'type' là 'message'
            if item.get('type') == 'message':
                # 3. Thêm mục tin nhắn vào danh sách đã lọc
                filtered_items.append(item)
        result = {
            'items': filtered_items
        }
        chat_transcript_v2=self.format_chat_transcript(result)
        chat_transcript = str(result)
        # print(chat_transcript)
        utc_now = datetime.now(timezone.utc)

        # Tạo múi giờ +7
        tz_plus7 = timezone(timedelta(hours=7))

        # Chuyển sang múi giờ +7
        local_time = utc_now.astimezone(tz_plus7)
        formatted_time_vn = local_time.strftime("%A, %d/%m/%Y %H:%M:%S")

        # system_prompt = f"""
        # Bạn là một chuyên gia phân tích cuộc gọi.
        # Dưới đây là lịch sử trò chuyện giữa nhân viên tư vấn và khách hàng:
        # ---
        # {chat_transcript}
        # ---

        # Nhiệm vụ:
        # 1. Phân loại mức độ quan tâm của khách hàng:
        #    - **'103'** nếu khách hàng đã thể hiện sự quan tâm đến sản phẩm đang được tiếp thị (ví dụ: đặt câu hỏi sâu, thể hiện ý định mua, hoặc có phản hồi tích cực).
        #    - **'105'** nếu khách hàng không quan tâm đến sản phẩm đang được tiếp thị.
        #    - **'104'** nếu khách hàng có báo gọi lại sau.
        # 2. Tóm tắt những sản phẩm mà khách hàng quan tâm, không quan tâm.
        # Trả về kết quả đúng định dạng JSON sau:
        # {{
        #    "status_call": "103" hoặc "104" hoặc "105",
        #    "summary": Tóm tắt ngắn gọn (ví dụ: "khách hàng quan tâm đến sản phẩm A, không quan tâm đến sản phẩm B").
        # }}

        # Không thêm bất kỳ giải thích nào khác ngoài JSON hợp lệ.
        # """
        system_prompt = f"""
        Bạn là một chuyên gia phân tích cuộc gọi.
        Dưới đây là lịch sử trò chuyện giữa nhân viên tư vấn và khách hàng:
        ---
        {chat_transcript}
        ---

        Nhiệm vụ:
        1. Phân loại mức độ quan tâm của khách hàng:
        - **'103'** nếu khách hàng đã thể hiện sự quan tâm đến sản phẩm đang được tiếp thị (ví dụ: đặt câu hỏi sâu, thể hiện ý định mua, hoặc có phản hồi tích cực).
        - **'105'** nếu khách hàng không quan tâm đến sản phẩm đang được tiếp thị.
        - **'104'** nếu khách hàng có báo gọi lại sau.
        2. Tóm tắt những sản phẩm mà khách hàng quan tâm, không quan tâm.
        Trả về kết quả đúng định dạng JSON sau:
        {{
        "status_call": "103" hoặc "104" hoặc "105",
        "summary": dạng string text, tóm tắt cuộc gọi để lấy được các thông tin về email, số điện thoại, tên dự án, loại hình bất động sản, mục đích sử dụng, vị trí, loại căn hộ, hướng, ngân sách, hình thứ thanh toán, lịch hẹn. Những thông tin không đề cập thì không cần có, nhưng nếu được nhắc đến thì phải tóm tắt thông tin chính xác.
        }}

        Không thêm bất kỳ giải thích nào khác ngoài JSON hợp lệ.
        """


        chat_ctx = ChatContext()
        chat_ctx.add_message(role="system", content=system_prompt)
        llm_stream = llm.chat(chat_ctx=chat_ctx)

        llm_response_full = ""
        async for chunk in llm_stream:
            if chunk and chunk.delta and chunk.delta.content:
                llm_response_full += chunk.delta.content

        print("LLM raw:", llm_response_full)

        try:
            parsed = json.loads(llm_response_full)
            status_call = parsed.get("status_call")
            summary = parsed.get("summary")

        except Exception as e:
            print(f"Lỗi parse JSON từ LLM: {e}")
            status_call = "105"
            summary = None

        if status_call not in ['103', '104','105']:
            print(f"LLM trả về giá trị không hợp lệ: {status_call}. Mặc định là 'Needs Analysis'.")
            status_call = '105'

        print(f"Đã xác định được status_call: {status_call}")
        print(f"Tóm tắt: {summary}")

        # 4. Lưu vào DB
        if caller_number != '':
            print(f"Số điện thoại người gọi là: {caller_number}. Tiến hành lưu vào DB.")
            await asyncio.gather(
                asyncio.to_thread(self.doc_du_lieu, caller_number, status_call, summary, chat_transcript_v2, time_start_call, lead_id),
                asyncio.to_thread(self.luu_vao_postgres, caller_number, status_call, summary, chat_transcript_v2, time_start_call, lead_id)
            )
            # doc_du_lieu(caller_number, status_call, summary,chat_transcript_v2,time_start_call,lead_id)
            # luu_vao_postgres(caller_number, status_call, summary,chat_transcript_v2,time_start_call,lead_id)
            print(time_start_call)
        else:
            print("Không tìm thấy số điện thoại người gọi (sip_identity), không thể lưu vào DB.")
        

    async def get_db_from_lead(phone_number: str) -> str | None:  
        # Hàm đồng bộ để thực hiện kết nối và truy vấn DB
        def sync_db_lookup(num_phone):
            connection = None
            website = "7a24ed17-fb8e-4fbe-a544-a3d4aa322da3"
            doc_id = "1dAWoUb3LghdUI2BZbCcQkWuhxpjEgIjB"
            lead_name= "Không được cung cấp"
            try:
                # 1. Kết nối DB
                connection = mysql.connector.connect(
                    host=DB_HOST,
                    port=DB_PORT,
                    database=DB_DATABASE,
                    user=DB_USER,
                    password=DB_PASSWORD
                )
                
                if connection and connection.is_connected():
                    cursor = connection.cursor()
                    
                    # 2. Câu truy vấn SQL
                    query = f"SELECT website,fax,lead_name FROM {TEN_BANG_LEAD} WHERE phone = %s LIMIT 1"
                    
                    # 3. Thực thi truy vấn
                    cursor.execute(query, (num_phone,))
                    
                    # 4. Lấy kết quả đầu tiên
                    result = cursor.fetchone()
                    
                    if result :
                        # Lấy giá trị cột 'website' (result là tuple, giá trị ở index 0)
                        website = result[0]
                        doc_id = result[1]
                        lead_name = result[2]
                        print(f"Tìm thấy website '{website}' cho số điện thoại {num_phone}.")
                        print(f"Tìm thấy fax '{doc_id}' cho số điện thoại {num_phone}.")
                        print(f"Tìm thấy name '{lead_name}' cho số điện thoại {num_phone}.")
                    else:
                        print(f"Không tìm thấy bản ghi Lead với số điện thoại {num_phone}.")
                        
            except Error as e:
                print(f"Lỗi MySQL khi đọc website: {e}")
            except Exception as e:
                print(f"Lỗi không xác định khi truy vấn DB: {e}")
            finally:
                # 5. Đóng kết nối
                if connection and connection.is_connected():
                    cursor.close()
                    connection.close()
                    
            return website,doc_id,lead_name

        # Chạy hàm đồng bộ trong một luồng riêng để không chặn asyncio event loop
        return await asyncio.to_thread(sync_db_lookup, phone_number)
    
nodo =Nodo()