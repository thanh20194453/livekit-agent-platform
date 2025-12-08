from datetime import datetime, timedelta, timezone
import uuid
from livekit.plugins import (
    openai
)
from livekit.agents import ChatContext
import mysql.connector
from mysql.connector import Error
import re
import json
import os
import boto3
from dotenv import load_dotenv
import requests
import time
import asyncio
DB_HOST = "13.251.189.33"
DB_PORT = 33106
DB_DATABASE = "_5e5899d8398b5f7b"
DB_USER = "root"
DB_PASSWORD = "admin"

TEN_BANG_LEAD = "tabLead"
TEN_BANG_OPPORTUNITY= "tabOpportunity"
TEN_BANG_HISTORY="tabCRM Note"
TEN_BANG_EVENT="tabEvent"
TEN_BANG_EVENT_PARTICIPANT="tabEvent Participants"
load_dotenv()


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
PERMANENT_BUCKET_NAME = "record-engress" 

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
    raise EnvironmentError("Thiếu các biến môi trường AWS. Vui lòng kiểm tra file .env")

# Khởi tạo client S3 và Transcribe
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

transcribe = boto3.client(
    "transcribe",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def create_s3_bucket(bucket_name: str):
    """Kiểm tra và tạo bucket S3 nếu chưa tồn tại."""
    try:
        s3.head_bucket(Bucket=bucket_name)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            s3.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' đã được tạo thành công.")
        else:
            print(f"Lỗi khi kiểm tra hoặc tạo bucket: {e}")
            raise

def upload_local_file_to_s3(bucket_name: str, local_file_path: str) -> str:
    """Đọc file cục bộ và tải lên S3."""
    file_name = os.path.basename(local_file_path)
    if not file_name.lower().endswith(".wav"):
        raise ValueError("Chỉ chấp nhận file âm thanh định dạng .wav")

    # file_extension = os.path.splitext(file_name)[1]
    # object_name = f"{uuid.uuid4()}{file_extension}"
    
    # Tải lên S3 bằng file name
    s3.upload_file(local_file_path, bucket_name, file_name)
    
    return f"s3://{bucket_name}/{file_name}"

# Đảm bảo bucket S3 được tạo khi API khởi động
create_s3_bucket(PERMANENT_BUCKET_NAME)


def start_transcribe_job(job_name: str, s3_uri: str, language_code: str = "vi-VN"):
    """Bắt đầu một job phiên âm Transcribe với nhận diện người nói."""
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat="wav", 
            LanguageCode=language_code,
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 2
            }
        )
    except transcribe.exceptions.ConflictException:
        print(f"Job {job_name} đã tồn tại, tiếp tục kiểm tra trạng thái.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo Transcribe Job: {e}")
        raise
        
def format_transcribe_result(job_name: str) -> str:
    # ... (Hàm này giữ nguyên như code trước) ...
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    response = requests.get(transcript_uri)
    results = response.json()["results"]
    
    items = results["items"]
    speaker_transcripts = []
    current_speaker = None
    current_text = ""
    
    for item in items:
        if "speaker_label" in item:
            speaker_label = item["speaker_label"]
            if speaker_label != current_speaker:
                if current_speaker is not None:
                    speaker_transcripts.append(f"{current_speaker.upper()}: {current_text.strip()}")
                current_speaker = speaker_label
                current_text = item["alternatives"][0]["content"]
            else:
                current_text += " " + item["alternatives"][0]["content"]
        else:
            current_text += item["alternatives"][0]["content"]

    if current_speaker is not None:
        speaker_transcripts.append(f"{current_speaker.upper()}: {current_text.strip()}")
    
    return "\n".join(speaker_transcripts)

def format_chat_transcript(chat_transcript_string):
    """
    Định dạng một chuỗi transcript trò chuyện thành đầu ra HTML, phân chia theo SPK_0 và SPK_1.

    Tham số:
    - chat_transcript_string (str): Chuỗi chứa transcript trò chuyện,
      ví dụ: "SPK_0: tin nhắn 1\nSPK_1: tin nhắn 2..."

    Trả về:
    - str: Chuỗi HTML đã được định dạng.
    """
    
    # Định nghĩa các hằng số màu và nhãn cho từng người nói
    # SPK_0 sẽ lấy màu của 'assistant' cũ (Agent)
    COLOR_SPK0 = "rgb(86, 182, 194)"
    LABEL_SPK0 = "SPK0"
    
    # SPK_1 sẽ lấy màu của 'customer' cũ (Khách hàng)
    COLOR_SPK1 = "rgb(209, 154, 102)"
    LABEL_SPK1 = "SPK1"

    html_output = '<div class="ql-editor read-mode" style="line-height: 1.4; font-size: 14px;">'

    # 1. Tách chuỗi thành các dòng tin nhắn
    lines = chat_transcript_string.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 2. Phân tích người nói và nội dung
        if line.startswith("SPK_0:"):
            label = LABEL_SPK0
            color = COLOR_SPK0
            # Cắt bỏ "SPK_0: "
            message = line[len("SPK_0:"):].strip()
        elif line.startswith("SPK_1:"):
            label = LABEL_SPK1
            color = COLOR_SPK1
            # Cắt bỏ "SPK_1: "
            message = line[len("SPK_1:"):].strip()
        else:
            # Bỏ qua các dòng không khớp với định dạng SPK_0: hoặc SPK_1:
            continue 

        # Thay thế ký tự xuống dòng (nếu có) bằng <br> cho định dạng HTML
        # Tuy nhiên, trong trường hợp này, mỗi tin nhắn đã là một dòng, 
        # nên việc thay thế này có thể không cần thiết nhưng vẫn giữ để phòng trường hợp 
        # nội dung tin nhắn có ký tự xuống dòng bên trong.
        formatted_message = message.replace("\n", "<br>")

        # 3. Tạo chuỗi HTML
        html_output += f'''
        <span style="background-color: rgb(40, 44, 52); color: rgb(152, 195, 121); font-weight: bold;">[{label}]</span>
        <span style="background-color: rgb(40, 44, 52); color: {color};"> {formatted_message}</span>
        '''
        
    html_output += '</div>'
    return html_output

#def doc_du_lieu(num_phone, sales_stage,chat_transcript,appointment_datetime,name,phone_number):
def doc_du_lieu(num_phone, sales_stage,chat_transcript,appointment_datetime,name,phone_number):
    utc_now = datetime.now(timezone.utc)

    # Tạo múi giờ +7
    tz_plus7 = timezone(timedelta(hours=7))

    # Chuyển sang múi giờ +7
    local_time = utc_now.astimezone(tz_plus7)
    comment="Cuộc trò chuyện kết thúc lúc: "+str(local_time) +" . Đánh giá cuộc gọi: "+sales_stage +"\n"+ chat_transcript
    connection = None
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_DATABASE,
            user=DB_USER,
            password=DB_PASSWORD
        )
        if connection.is_connected():
            cursor = connection.cursor()
            check_query = f"SELECT name FROM {TEN_BANG_OPPORTUNITY} WHERE phone = %s"
            cursor.execute(check_query, (num_phone,))
            existing = cursor.fetchone()
            opp_name=''
            if existing:
                opp_name=existing[0]
                update_query = f"""
                    UPDATE {TEN_BANG_OPPORTUNITY}
                    SET sales_stage = %s, modified = %s, transaction_date = %s
                    WHERE phone = %s
                """
                cursor.execute(update_query, (sales_stage, local_time,local_time.strftime("%Y-%m-%d") ,num_phone))
                print(f"Đã cập nhật sales_stage cho số {num_phone}.")
            else:
                query = f"SELECT * FROM {TEN_BANG_LEAD} WHERE phone ={num_phone}"
                cursor.execute(query)
                records = cursor.fetchall()
                opp_name="CRM-OPP-2025-"+str(uuid.uuid4())


                query2 = f"""
                    INSERT INTO {TEN_BANG_OPPORTUNITY} 
                    (name, creation, modified_by,naming_series,opportunity_from,party_name,customer_name,sales_stage,language,title,contact_person,contact_mobile,phone,contact_display, transaction_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # 3. Chuẩn bị dữ liệu để truyền vào (Phải cùng thứ tự với các cột trong query)
                data = (opp_name, local_time, local_time, "CRM-OPP-.YYYY.-","Lead",records[0][0],records[0][12],sales_stage,"en",records[0][12],records[0][12],records[0][25],records[0][25],records[0][12],datetime.now().strftime("%Y-%m-%d"))
                
                # 4. Thực thi câu lệnh
                cursor.execute(query2, data)
                
                # 5. Xác nhận giao dịch (lưu thay đổi vào DB)
                print(f"Thành công: Đã thêm bản ghi vào bảng.")

            cursor.execute(f"SELECT MAX(name) FROM `{TEN_BANG_HISTORY}`")
            max_id = cursor.fetchone()[0]
            new_note_id = (max_id if max_id is not None else 0) + 1
            query3 = f"""
                INSERT INTO `{TEN_BANG_HISTORY}` 
                (name, creation, modified_by,note,parent,parentfield,parenttype,owner,added_by,added_on)
                VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s)
            """
            
            # 3. Chuẩn bị dữ liệu để truyền vào (Phải cùng thứ tự với các cột trong query)
            data = (new_note_id, local_time, local_time, comment,opp_name,"notes","Opportunity","thienvd@sphinxjsc.com","thienvd@sphinxjsc.com",local_time)
            
            # 4. Thực thi câu lệnh
            cursor.execute(query3, data)

            if appointment_datetime:
                ev_parti_id=str(uuid.uuid4())
                ev_id=str(uuid.uuid4())
                query4 = f"""
                    INSERT INTO `{TEN_BANG_EVENT}` 
                    (name, creation, modified,modified_by,owner,subject,event_category,event_type,starts_on,status,description,_comments,_seen)
                    VALUES (%s, %s, %s, %s,%s, %s, %s, %s,%s,%s,%s,%s,%s)
                """
                discription=f'''<div class="ql-editor read-mode"><p>Tên: {name}. Số điện thoại: {phone_number}</p></div>'''
                comments='''[{"comment": "<div class=\\"ql-editor read-mode\\"><p>sgdfscdxs</p></div>", "by": "thienvd@sphinxjsc.com", "name": "thanh@12589"}]'''
                new_content_html = f"<p>Tên: {name}. Số điện thoại: {phone_number}</p>"
                old_content = "<p>sgdfscdxs</p>"
                final_comments = comments.replace(old_content, new_content_html)
                new_content_html = f"{ev_parti_id}"
                old_content = "thanh@12589"
                final_comments = final_comments.replace(old_content, new_content_html)
                # 3. Chuẩn bị dữ liệu để truyền vào (Phải cùng thứ tự với các cột trong query)
                data = (ev_id, local_time, local_time, "thienvd@sphinxjsc.com","thienvd@sphinxjsc.com","lịch hẹn","Event","Private",appointment_datetime,"Open",discription,final_comments,'''["thienvd@sphinxjsc.com"]''')
                
                # 4. Thực thi câu lệnh
                cursor.execute(query4, data)

                query5 = f"""
                    INSERT INTO `{TEN_BANG_EVENT_PARTICIPANT}` 
                    (name, creation, modified,modified_by,owner,reference_doctype,reference_docname,parent,parentfield,parenttype)
                    VALUES (%s, %s, %s, %s,%s, %s, %s, %s,%s,%s)
                """
                # 3. Chuẩn bị dữ liệu để truyền vào (Phải cùng thứ tự với các cột trong query)
                data = (ev_parti_id, local_time, local_time, "thienvd@sphinxjsc.com","thienvd@sphinxjsc.com","Opportunity",opp_name,ev_id,"event_participants","Event")
                
                # 4. Thực thi câu lệnh
                cursor.execute(query5, data)
                
            # 5. Xác nhận giao dịch (lưu thay đổi vào DB)
            connection.commit()
            print(f"Thành công: Đã thêm bản ghi history vào bảng.")



    except Error as e:
        print(f"Lỗi MySQL: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
async def analyze_and_save_opportunity(transcribed_text,caller_number):
    """
    Phân tích lịch sử cuộc gọi và lưu thông tin cơ hội vào DB.
    """
    llm = openai.LLM(model="gpt-4o-mini")
    chat_transcript_v2=format_chat_transcript(transcribed_text)
    chat_transcript = str(transcribed_text)
    utc_now = datetime.now(timezone.utc)

    # Tạo múi giờ +7
    tz_plus7 = timezone(timedelta(hours=7))

    # Chuyển sang múi giờ +7
    local_time = utc_now.astimezone(tz_plus7)
    formatted_time_vn = local_time.strftime("%A, %d/%m/%Y %H:%M:%S")
    print(formatted_time_vn)
    system_prompt = f"""
    Bạn là một chuyên gia phân tích cuộc gọi.
    Dưới đây là lịch sử trò chuyện giữa nhân viên tư vấn và khách hàng:
    ---
    {chat_transcript}
    ---

    Nhiệm vụ:
    1. Phân loại mức độ quan tâm của khách hàng:
       - **'Value Proposition'** nếu khách hàng đã thể hiện sự quan tâm đến ít nhất MỘT sản phẩm được tiếp thị (ví dụ: đặt câu hỏi sâu, thể hiện ý định mua, hoặc có phản hồi tích cực).
       - **'Needs Analysis'** nếu khách hàng KHÔNG quan tâm đến tất cả sản phẩm được tiếp thị (ví dụ: từ chối thẳng thừng, chỉ trả lời qua loa, hoặc không quan tâm đến bất cứ sản phẩm nào).
    2. Kiểm tra xem có lịch hẹn gặp (appointment) hợp lệ hay không. Hiện tại là {formatted_time_vn}. Bạn phải dựa vào lịch sử trò chuyện và thời gian hiện tại để tính ngày mà khách hàng muốn hẹn gặp. Lịch hẹn phải nằm sau thời gian hiện tại mới tính là hợp lệ.
       - Nếu có, trích xuất thời gian lịch hẹn thành định dạng ISO 8601 (YYYY-MM-DDTHH:MM:SS), ví dụ: "2025-10-21T14:00:00". Nếu không nếu rõ thời gian hẹn trong ngày thì đặt là 00:00:00 của ngày hẹn.
       - Nếu không có, trả về null.
    3. Tìm kiếm các thông tin sau trong lịch sử trò truyện: (trả về null nếu không có)
       - Tên khách hàng
       - Số điện thoại

    Trả về kết quả đúng định dạng JSON sau:
    {{
       "opportunity_stage": "Needs Analysis" hoặc "Value Proposition",
       "appointment_datetime": "YYYY-MM-DDTHH:MM:SS" hoặc null
       "name": Tên khách hàng hoặc null
       "phone_number": số điện thoại hoặc null
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
    clean_response = re.sub(r"```(json)?", "", llm_response_full, flags=re.IGNORECASE).strip()
    clean_response = re.sub(r"```", "", clean_response).strip()
    try:
        parsed = json.loads(clean_response)
        opportunity_stage = parsed.get("opportunity_stage", "Needs Analysis")
        appointment_datetime = parsed.get("appointment_datetime")
        name = parsed.get("name")
        phone_number = parsed.get("phone_number")

    except Exception as e:
        print(f"Lỗi parse JSON từ LLM: {e}")
        opportunity_stage = "Needs Analysis"
        appointment_datetime = None

    if opportunity_stage not in ['Needs Analysis', 'Value Proposition']:
        print(f"LLM trả về giá trị không hợp lệ: {opportunity_stage}. Mặc định là 'Needs Analysis'.")
        opportunity_stage = 'Needs Analysis'

    print(f"Đã xác định được sales_stage/opportunity: {opportunity_stage}")
    print(f"Lịch hẹn: {appointment_datetime}")

    # 4. Lưu vào DB
    if caller_number != '':
        print(f"Số điện thoại người gọi là: {caller_number}. Tiến hành lưu vào DB.")
        doc_du_lieu(caller_number, opportunity_stage, chat_transcript_v2, appointment_datetime,name,phone_number)
    else:
        print("Không tìm thấy số điện thoại người gọi (sip_identity), không thể lưu vào DB.")

async def transcribe_s3_uri(
    caller_number: str = "",
    s3_uri: str = "",
    poll_interval: int = 5,
    max_wait_time: int = 600,
):
    """
    Bắt đầu một Amazon Transcribe Job bằng đường dẫn S3 URI đã có sẵn, 
    chờ cho đến khi hoàn thành, sau đó phân tích và lưu kết quả vào DB.
    """
    # 1. Kiểm tra định dạng S3 URI (Tùy chọn, nhưng nên làm)
    if not s3_uri.lower().startswith("s3://"):
        raise RuntimeError(400, f"Đường dẫn S3 URI không hợp lệ: {s3_uri}. Phải bắt đầu bằng 's3://'")

    # 2. Tạo tên job và bắt đầu Transcribe Job
    job_name = f"transcription-job-{uuid.uuid4()}"
    
    try:
        start_transcribe_job(job_name, s3_uri)
    except Exception as e:
        # Xử lý các lỗi khởi tạo job (ví dụ: file không tồn tại, định dạng sai,...)
        raise RuntimeError(500, detail=f"Lỗi khi khởi tạo Transcribe Job: {e}")

    print(f"Đã tạo job {job_name} cho S3 URI: {s3_uri}")

    async def wait_transcribe():
        """Hàm chạy polling bất đồng bộ để kiểm tra trạng thái job"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            response = transcribe.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                return "DONE"

            if status == "FAILED":
                reason = response["TranscriptionJob"]["FailureReason"]
                # Lấy thêm thông tin chi tiết về lỗi
                fail_detail = f"Transcribe FAILED: {reason}"
                print(fail_detail)
                raise RuntimeError(400, fail_detail)

            print(f"{job_name} đang {status}, chờ {poll_interval}s...")
            await asyncio.sleep(poll_interval)

        # Nếu thoát khỏi vòng lặp do hết thời gian chờ
        raise TimeoutError(504, f"Timed out (>{max_wait_time}s) job {job_name}")

    try:
        # 3. Chờ cho job hoàn thành (Timeout tối đa 30 phút/1800 giây cho toàn bộ request)
        await asyncio.wait_for(wait_transcribe(), timeout=1800)

        # 4. Lấy transcript
        transcribed_text = format_transcribe_result(job_name)

        # 5. Phân tích và lưu DB
        await analyze_and_save_opportunity(transcribed_text,caller_number)

        return {"status": "COMPLETED", "job_name": job_name}

    except asyncio.TimeoutError:
        # Lỗi timeout tổng request FastAPI
        raise TimeoutError(504, "Request timeout sau 30 phút")
