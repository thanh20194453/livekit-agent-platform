from datetime import datetime, timezone, timedelta
import uuid
import logging
import asyncio
import boto3
import time
import requests
from typing import Dict, Any, Optional, Tuple
from livekit.plugins import (
    openai
)
from livekit.agents import ChatContext
import re
import json
import httpx

from config.settings import settings
import mysql.connector
from mysql.connector import Error

from external.ultils.transcribe import TranscribeUtils

logger = logging.getLogger("nodo-manager")
logger.setLevel(logging.ERROR)


class CRMManager:
    """
    Create, analyze and save transcribe.
    
    Features:
    - Create transcribe from recording file
    - Save transcribe to s3
    - Analyze transcribe by llm
    - Save analyze result to crm
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

        # crm db
        self.CRM_TOKEN: str = settings.crm.token
        self.CRM_URL: str = settings.crm.url

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
        self.time_start_call: str = ""
        self.user_name: str = bot_config.call_config.get("user_name", "")
        self.person_id: str = bot_config.call_config.get("person_id", "")


    def _format_chat_transcript_crm(self, chat_transcript_string: str) -> str:
        """
        Reformat transcript to save to crm
        """
        COLOR_SPK0 = "rgb(86, 182, 194)"
        LABEL_SPK0 = "SPK0"
        COLOR_SPK1 = "rgb(209, 154, 102)"
        LABEL_SPK1 = "SPK1"

        html_output = '<div class="ql-editor read-mode" style="line-height: 1.4; font-size: 14px;">'

        lines = chat_transcript_string.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("SPK_0:"):
                label = LABEL_SPK0
                color = COLOR_SPK0
                message = line[len("SPK_0:"):].strip()
            elif line.startswith("SPK_1:"):
                label = LABEL_SPK1
                color = COLOR_SPK1
                message = line[len("SPK_1:"):].strip()
            else:
                continue 
            formatted_message = message.replace("\n", "<br>")

            html_output += f'''
            <span style="background-color: rgb(40, 44, 52); color: rgb(152, 195, 121); font-weight: bold;">[{label}]</span>
            <span style="background-color: rgb(40, 44, 52); color: {color};"> {formatted_message}</span>
            '''
            
        html_output += '</div>'
        return html_output
    
    async def _transcribe_s3_uri(
        self,
        caller_number: str = "",
        time_start_call: str = "", 
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

            await self._analyze_and_save_opportunity(transcribed_text,caller_number)

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
        BASE_URL = "https://grand.nodo.vn/rest_api/pr/property/ai/query_object_tree"
        LOGIN_URL = "https://grand.nodo.vn/rest_api/pu/property/auth/login"
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
                full_response_data = response.json()

                access_token = full_response_data.get("data").get("access_token")
            else:
                logger.error(f"Status: error login ({response.status_code})")
                logger.error("Error content:", response.text)

            return access_token
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Erorr in sending sign in request: {e}")
            return None

    async def _save_to_crm(
        self,
        sales_stage: str,
        chat_transcript: str,
        appointment_datetime: Optional[str]
    ) -> None:
        try:
            utc_now = datetime.now(timezone.utc)
            tz_plus7 = timezone(timedelta(hours=7))
            local_time = utc_now.astimezone(tz_plus7)
            time_str = local_time.strftime('%y%m%d%H%M%S')
            
            note_content = f"Cuộc trò chuyện kết thúc lúc: {local_time}. Đánh giá cuộc gọi: {sales_stage}\n{chat_transcript}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.CRM_TOKEN}"
            }

            async with httpx.AsyncClient() as client:
                tasks = []
                #Opportunity
                payload_opp = {
                    "name": f"{self.user_name}_{time_str}",
                    "stage": sales_stage,
                    "pointOfContactId": self.person_id
                }
                tasks.append(client.post(f"{self.CRM_URL}opportunities", headers=headers, json=payload_opp))

                # Note
                payload_note = {
                    "title": f"note_{self.user_name}_{time_str}",
                    "bodyV2": {"markdown": note_content}
                }
                tasks.append(client.post(f"{self.CRM_URL}notes", headers=headers, json=payload_note))

                # Task
                if appointment_datetime:
                    appointment_datetime = f"{appointment_datetime}+07:00"
                    payload_task = {
                        "title": f"task_{self.user_name}_{time_str}",
                        "bodyV2": {"markdown": note_content},
                        "dueAt": appointment_datetime
                    }
                    tasks.append(client.post(f"{self.CRM_URL}tasks", headers=headers, json=payload_task))

                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                opp_id, note_id, task_id = None, None, None

                def get_id_from_res(res, key: str):
                    if isinstance(res, httpx.Response) and res.status_code == 201:
                        try:
                            return res.json()['data'][key]['id']
                        except (KeyError, TypeError):
                            return None
                    return None

                opp_id = get_id_from_res(responses[0], 'createOpportunity')
                note_id = get_id_from_res(responses[1], 'createNote')
                
                if opp_id: logger.info("Create opp successfully")
                if note_id: logger.info("Create note successfully")

                if appointment_datetime and len(responses) > 2:
                    task_id = get_id_from_res(responses[2], 'createTask')
                    if task_id: logger.info("Create task successfully")

                # --- BƯỚC 2: TẠO TARGETS (PARALLEL) ---
                target_tasks = []

                if note_id and opp_id:
                    payload_note_target = {
                        "noteId": note_id,
                        "personId": self.person_id,
                        "opportunityId": opp_id
                    }
                    target_tasks.append(client.post(f"{self.CRM_URL}noteTargets", headers=headers, json=payload_note_target))

                if task_id and opp_id:
                    payload_task_target = {
                        "taskId": task_id,
                        "personId": self.person_id,
                        "opportunityId": opp_id
                    }
                    target_tasks.append(client.post(f"{self.CRM_URL}taskTargets", headers=headers, json=payload_task_target))

                if target_tasks:
                    await asyncio.gather(*target_tasks, return_exceptions=True)
                    logger.info("Targets created in parallel")
        except Exception as e:
            logger.error(f"Error in saving to crm: {e}")

    async def _analyze_and_save_opportunity(
        self,
        transcribed_text: str,
        caller_number: str
    ) -> None:
        """
        Analyze and save transcribe
        """
        try:
            llm = openai.LLM(model="gpt-4o-mini")
            chat_transcript = str(transcribed_text)
            chat_transcript_crm=self._format_chat_transcript_crm(transcribed_text)
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
            - **'PROPOSAL'** nếu khách hàng đã thể hiện sự quan tâm đến ít nhất MỘT sản phẩm được tiếp thị (ví dụ: đặt câu hỏi sâu, thể hiện ý định mua, hoặc có phản hồi tích cực).
            - **'SCREENING'** nếu khách hàng KHÔNG quan tâm đến tất cả sản phẩm được tiếp thị (ví dụ: từ chối thẳng thừng, chỉ trả lời qua loa, hoặc không quan tâm đến bất cứ sản phẩm nào).
            2. Kiểm tra xem có lịch hẹn gặp (appointment) hợp lệ hay không. Hiện tại là {formatted_time_vn}. Bạn phải dựa vào lịch sử trò chuyện và thời gian hiện tại để tính ngày mà khách hàng muốn hẹn gặp. Lịch hẹn phải nằm sau thời gian hiện tại mới tính là hợp lệ.
            - Nếu có, trích xuất thời gian lịch hẹn thành định dạng ISO 8601 (YYYY-MM-DDTHH:MM:SS), ví dụ: "2025-10-21T14:00:00". Nếu không nếu rõ thời gian hẹn trong ngày thì đặt là 00:00:00 của ngày hẹn.
            - Nếu không có, trả về null.

            Trả về kết quả đúng định dạng JSON sau:
            {{
            "opportunity_stage": "PROPOSAL" hoặc "SCREENING",
            "appointment_datetime": "YYYY-MM-DDTHH:MM:SS" hoặc null
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

            clean_response = re.sub(r"```(json)?", "", llm_response_full, flags=re.IGNORECASE).strip()
            clean_response = re.sub(r"```", "", clean_response).strip()
            try:
                parsed = json.loads(clean_response)
                opportunity_stage: str = parsed.get("opportunity_stage", "SCREENING")
                appointment_datetime: Optional[str] = parsed.get("appointment_datetime")

            except Exception as e:
                logger.error(f"Error parse JSON from LLM: {e}")
                opportunity_stage = "SCREENING"
                appointment_datetime = None

            if opportunity_stage not in ['SCREENING', 'PROPOSAL']:
                logger.error(f"Ivalid LLM reponse: {opportunity_stage}. Default is 'SCREENING'.")
                opportunity_stage = 'SCREENING'

            if self.person_id != '':
                await self._save_to_crm(opportunity_stage,chat_transcript,appointment_datetime)
            else:
                logger.error("Can not find person_id, Cannot save to DB.")
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
                        await self._transcribe_s3_uri(self.phone_number,self.time_start_call)
                        return
                    
                    await asyncio.sleep(3)

                logger.error("[BG] Timeout waiting S3 — file is not uploaded in time.")
            else:
                logger.info(f"Call ended with status {self.call_status_num}: {self.call_status_text}")
        except Exception as e:
            logger.error(f"✗ Fail to transcribe s3: {e}", exc_info=True)
            raise