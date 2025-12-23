from datetime import datetime, timezone, timedelta
import uuid
import logging
import asyncio
import boto3
import time
import requests
from typing import Dict, Any, Optional
from livekit.plugins import (
    openai
)
from livekit.agents import ChatContext
import re
import json

from config.settings import settings
import psycopg2
from psycopg2 import sql
from external.ultils.transcribe import TranscribeUtils


logger = logging.getLogger("nodo-manager")
logger.setLevel(logging.ERROR)



class NodoManager:
    """
    Create, analyze and save transcribe.
    
    Features:
    - Create transcribe from recording file
    - Save transcribe to s3
    - Analyze transcribe by llm
    - Save analyze result to postgres and nodo api
    """
    
    def __init__(self, bot_config: Any) -> None:
        self.bot_config = bot_config
        self.transcribeutil: TranscribeUtils = TranscribeUtils(bot_config)
        # Get S3 configuration from bot config
        s3_config: Dict[str, Any] = bot_config.get_s3_config() or {}

        # Extract predefined S3 URLs from bot config
        self.s3_recording_url = self.transcribeutil.http_s3_to_uri(s3_config.get("recording_url", ""))
        self.s3_transcript_url = self.transcribeutil.http_s3_to_uri(s3_config.get("transcript_url", ""))
        self.s3_uri = s3_config.get("uri", "")
        
        # S3 credentials for validation and operations
        self.s3_bucket: str = settings.aws.s3_bucket_name
        self.s3_region: str = settings.aws.region
        self.s3_access_key: str = settings.aws.access_key.get_secret_value()
        self.s3_secret_key: str = settings.aws.secret_key.get_secret_value()

        # nodo history db
        self.DB_HOST: str = settings.dbnodo.db_host
        self.DB_PORT_PG: int = int(settings.dbnodo.db_port) if isinstance(settings.dbnodo.db_port, str) else settings.dbnodo.db_port
        self.DB_PG: str = settings.dbnodo.db_name
        self.DB_USER_PG: str = settings.dbnodo.db_user
        self.DB_PASSWORD_PG: str = settings.dbnodo.db_password
        # call status
        self.call_status_num: str = ""
        self.call_status_text: str = ""
        self.transcribe: Any = boto3.client(
                "transcribe",
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                region_name=settings.aws.region,
            )
        # call_info:
        self.phone_number: str = bot_config.call_config.get("phone_number", "")
        self.lead_id: str = bot_config.call_config.get("lead_id", "")
        self.summary: str = bot_config.call_config.get("summary", "")
        self.time_start_call: str = ""

    async def _transcribe_s3_uri(
        self,
        caller_number: str = "",
        time_start_call: str = "", 
        lead_id: str = "",
        poll_interval: int = 5,
        max_wait_time: int = 600,
    ) -> Dict[str, str]:
        """
        Start an Amazon Transcribe Job and wait until job done. Save result to db.
        """
        if not self.s3_recording_url.lower().startswith("s3://"):
            raise ValueError(f"S3 URI is invalid: {self.s3_recording_url}. Must start by 's3://'")

        job_name = f"transcription-job-{uuid.uuid4()}"
        
        try:
            self.transcribeutil.start_transcribe_job(job_name)
        except Exception as e:
            raise RuntimeError(f"Error in create Transcribe Job: {e}")


        async def wait_transcribe():
            """Asynchronous polling function to check job status"""
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                response = self.transcribe.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                status = response["TranscriptionJob"]["TranscriptionJobStatus"]

                if status == "COMPLETED":
                    return "DONE"

                if status == "FAILED":
                    reason = response["TranscriptionJob"]["FailureReason"]
                    fail_detail = f"Transcribe FAILED: {reason}"
                    logger.error(fail_detail)
                    raise RuntimeError(fail_detail)

                await asyncio.sleep(poll_interval)

            raise TimeoutError(f"Timed out (>{max_wait_time}s) job {job_name}")

        try:
            await asyncio.wait_for(wait_transcribe(), timeout=1800)

            transcribed_text = self.transcribeutil.format_transcribe_result(job_name)

            await self._analyze_and_save_opportunity(transcribed_text,caller_number, time_start_call, lead_id)

            return {"status": "COMPLETED", "job_name": job_name}

        except asyncio.TimeoutError:
            raise TimeoutError("Request timeout after 30 phút")
        
    def _lay_token_dang_nhap(self) -> Optional[str]:
        """Get nodo sign in token"""
        access_token: Optional[str] = None
        HEADERS = {
            "accept": "*/*",
            "Content-Type": "application/json" 
        }
        LOGIN_URL = settings.dbnodo.apiloginurl
        payload_login = {
            "login": settings.dbnodo.apiloginuser,
            "password": settings.dbnodo.apiloginpw
        }

        try:
            response = requests.post(
                LOGIN_URL, 
                headers=HEADERS, 
                json=payload_login,
                verify=False 
            )
            
            if response.status_code == 200:
                full_response_data = response.json()

                access_token = full_response_data.get("data").get("access_token")
            else:
                logger.error(f"Trạng thái: Đăng nhập LỖI ({response.status_code})")
                logger.error("Nội dung lỗi:", response.text)

            return access_token
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Erorr in sending sign in request: {e}")
            return None

        
    def _save_in_postgres(
        self,
        caller_number: str,
        status_call: str,
        summary: Optional[str],
        chat_transcript_v2: str,
        time_start_call: str,
        lead_id: str
    ) -> None:
        """Save data to table nodo_outbound.nodo_history."""
        conn: Optional[Any] = None
        cur: Optional[Any] = None
        try:
            conn = psycopg2.connect(
                host=self.DB_HOST,
                port=self.DB_PORT_PG,
                database=self.DB_PG,
                user=self.DB_USER_PG,
                password=self.DB_PASSWORD_PG
            )
            cur = conn.cursor()
            assert cur is not None

            insert_query = sql.SQL("""
                INSERT INTO nodo_outbound.nodo_history (
                    id, caller_number, status_call, note, context_call_json, 
                    time_start_call, lead_id, time_end_call
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
            """)
            
            tz_plus7 = timezone(timedelta(hours=7))
            time_end_dt = datetime.now(tz_plus7)
            time_end_formatted = time_end_dt.strftime("%Y-%m-%d %H:%M:%S")
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
            

            cur.execute(insert_query, data_to_insert)
            conn.commit()

        except (Exception, psycopg2.Error) as error:
            logger.error(f"\n[DB] Error in connecting or saving data to PostgreSQL: {error}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def _save_in_nodo_api(
        self,
        caller_number: str,
        status_call: str,
        summary: Optional[str],
        chat_transcript_v2: str,
        time_start_call: str,
        lead_id: str
    ) -> None:
        """Save data to Nodo API"""
        access_token: Optional[str] = self._lay_token_dang_nhap()
        if not access_token:
            logger.error("Failed to get access token")
            return
        tz_plus7 = timezone(timedelta(hours=7))
        time_end_dt = datetime.now(tz_plus7)
        time_end_formatted = time_end_dt.strftime("%d-%m-%Y %H:%M:%S")
        auth: str = 'Bearer ' + access_token
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
        try:
            response = requests.post(url, headers=headers, json=payload)
        except Exception as e:
            logger.error("Error write data to API nodo", exc_info=True)
            raise
    async def _analyze_and_save_opportunity(
        self,
        transcribed_text: str,
        caller_number: str,
        time_start_call: str,
        lead_id: str
    ) -> None:
        """
        Analyze and save transcribe
        """
        try:
            llm = openai.LLM(model="gpt-4o-mini")
            chat_transcript = str(transcribed_text)
            self.transcribeutil.upload_string_to_s3(chat_transcript)
            utc_now = datetime.now(timezone.utc)

            tz_plus7 = timezone(timedelta(hours=7))

            local_time = utc_now.astimezone(tz_plus7)
            formatted_time_vn = local_time.strftime("%A, %d/%m/%Y %H:%M:%S")
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
            2. Tóm tắt thông tin:
            - Trích xuất các thông tin sau: {self.summary}.
            - **YÊU CẦU QUAN TRỌNG VỀ KEY**: Các key trong JSON summary phải giữ nguyên dấu tiếng Việt và khoảng trắng chính xác như danh sách đã cung cấp.
            - Nếu thông tin không có trong cuộc gọi, hãy để giá trị là null hoặc chuỗi rỗng.
            Yêu cầu định dạng trả về duy nhất một JSON như sau:
            {{
            "status_call": "103" hoặc "104" hoặc "105",
            "summary": "một chuỗi JSON đã được escape (stringified JSON) chứa các thông tin: {self.summary}"
            }}

            Ví dụ kết quả mong muốn: {{ "status_call": "103", "summary": "{{\\\"họ và tên\\\": \\\"Lê Văn Minh\\\", \\\"mục tiêu\\\": \\\"xây nhà\\\", \\\"giá tiền\\\": null}}" }}
            Không thêm bất kỳ giải thích nào khác ngoài JSON hợp lệ.
            """


            chat_ctx = ChatContext()
            chat_ctx.add_message(role="system", content=system_prompt)
            llm_stream = llm.chat(chat_ctx=chat_ctx)

            llm_response_full = ""
            async for chunk in llm_stream:
                if chunk and chunk.delta and chunk.delta.content:
                    llm_response_full += chunk.delta.content

            clean_response = re.sub(r"```(json)?", "", llm_response_full, flags=re.IGNORECASE).strip()
            clean_response = re.sub(r"```", "", clean_response).strip()
            try:
                parsed = json.loads(clean_response)
                status_call = parsed.get("status_call")
                summary = parsed.get("summary")

            except Exception as e:
                logger.error(f"Error parse JSON from LLM: {e}")
                status_call = "105"
                summary = None

            if status_call not in ['103', '104','105']:
                logger.error(f"Ivalid LLM reponse: {status_call}. Default is '105'.")
                status_call = '105'

            if caller_number != '':
                await asyncio.gather(
                    asyncio.to_thread(self._save_in_nodo_api, caller_number, status_call, summary, chat_transcript, time_start_call, lead_id),
                    asyncio.to_thread(self._save_in_postgres, caller_number, status_call, summary, chat_transcript, time_start_call, lead_id)
                )
            else:
                logger.error("Can not find phone number, Cannot save to DB.")
        except Exception as e:
            logger.error(f"Error in analyze transcribe: {e}", exc_info=True)
            raise

    async def wait_and_process(self) -> None:
        "Wait analyze and save transcribe"
        try:
            if self.call_status_num=="":
                logger.info(f"Start check file S3: {self.s3_recording_url}")

                for _ in range(120):
                    if await self.transcribeutil.s3_exists():
                        await self._transcribe_s3_uri(self.phone_number,self.time_start_call,self.lead_id)
                        return
                    
                    await asyncio.sleep(3)

                logger.error("[BG] Timeout waiting S3 — file is not uploaded in time.")
            else:
                await asyncio.gather(
                    asyncio.to_thread(self._save_in_nodo_api, self.phone_number, self.call_status_num,self.call_status_text,"",self.time_start_call,self.lead_id),
                    asyncio.to_thread(self._save_in_postgres, self.phone_number, self.call_status_num,self.call_status_text,"",self.time_start_call,self.lead_id)
                )
        except Exception as e:
            logger.error(f"✗ Fail to transcribe s3: {e}", exc_info=True)
            raise