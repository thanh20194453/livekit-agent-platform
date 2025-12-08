from datetime import datetime, timedelta
from datetime import datetime, timezone, timedelta
import os
from livekit import api
import boto3
from botocore.exceptions import ClientError
import httpx
import asyncio
def get_current_time():
    utc_now = datetime.now(timezone.utc)

    # Tạo múi giờ +7
    tz_plus7 = timezone(timedelta(hours=7))

    # Chuyển sang múi giờ +7
    local_time = utc_now.astimezone(tz_plus7)
    formatted_time_vn = local_time.strftime("%A, %d/%m/%Y %H:%M:%S")
    time_start_formatted = local_time.strftime("%d-%m-%Y %H:%M:%S")
    return time_start_formatted

async def wait_and_process(caller_number:str ,s3_uri: str,handler: callable):
    print(f"[BG] Bắt đầu kiểm tra file S3: {s3_uri}")

    # tối đa 30 lần, mỗi lần 2 giây = ~60 giây
    for _ in range(120):
        print("test lần ")
        if await s3_exists(s3_uri):
            print(f"[BG] File S3 đã sẵn sàng: {s3_uri}")
            await handler(caller_number,s3_uri)
            return
        
        await asyncio.sleep(3)

    print("[BG] Hết thời gian chờ S3 — file chưa xuất hiện.")
async def process_recording_api(s3_uri: str):
    """
    Thực hiện cuộc gọi POST đến API để bắt đầu quá trình Transcribe và phân tích.
    """
    # Thay đổi địa chỉ API của bạn nếu cần
    API_ENDPOINT = "http://54.255.208.182:8129/process_and_wait"
    
    # Các tham số cho API
    params = {
        "s3_uri": s3_uri,
        "poll_interval": 5,
        "max_wait_time": 1800, # 5 phút chờ đợi
    }
    
    # Sử dụng httpx.AsyncClient để thực hiện cuộc gọi bất đồng bộ
    async with httpx.AsyncClient(timeout=30) as client:
        print(f"Bắt đầu gọi API xử lý ghi âm: {API_ENDPOINT} với S3 URI: {s3_uri}")
        try:
            # Cuộc gọi POST
            response = await client.post(API_ENDPOINT, params=params)
            
            # Kiểm tra mã trạng thái
            if response.status_code == 200:
                print(f"API xử lý ghi âm thành công! Phản hồi: {response.json()}")
            else:
                print(f"API xử lý ghi âm THẤT BẠI. Mã lỗi: {response.status_code}. Phản hồi: {response.text}")
                
        except httpx.RequestError as e:
            print(f"LỖI HTTP khi gọi API xử lý ghi âm: {e}")
        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi gọi API xử lý ghi âm: {e}")

async def s3_exists(s3_uri: str):
    # parse s3://bucket/path/file
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
async def start_recording(room_name:str):
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

    # 1. Khởi tạo LiveKit API Client
    # Cần có LIVEKIT_API_KEY và LIVEKIT_API_SECRET trong biến môi trường
    lkapi = api.LiveKitAPI()

    # 2. Cấu hình và Bắt đầu Ghi âm (Egress)
    s3_output = api.S3Upload(
        bucket=S3_BUCKET_NAME, 
        region=AWS_REGION,
        access_key=AWS_ACCESS_KEY_ID,
        secret=AWS_SECRET_ACCESS_KEY,
        # Nếu bạn dùng dịch vụ S3 tương thích, hãy thêm force_path_style=True
    )
    s3_uri_of_recording = f"s3://{S3_BUCKET_NAME}/recordings/{datetime.now().strftime('%Y%m%d%H%M%S')}_audio.ogg"
    s3_url=f"s3://{S3_BUCKET_NAME}/recordings/{room_name}/{datetime.now().strftime('%Y%m%d%H%M%S')}_audio.ogg"
    # Cấu hình đầu ra là một file audio MP4
    file_output = api.EncodedFileOutput(
        # Đặt tên file theo tên phòng và thời gian hiện tại
        filepath=f"recordings/{datetime.now().strftime('%Y%m%d%H%M%S')}_audio.mp4",
        s3=s3_output,
    )
    
    egress_request = api.RoomCompositeEgressRequest(
        room_name=room_name,
        audio_only=True, # QUAN TRỌNG: Chỉ ghi âm thanh
        file_outputs=[file_output],
        # Preset cơ bản, vì chỉ ghi âm thanh
        preset=api.EncodingOptionsPreset.H264_720P_30, 
    )
    
    egress_info = None
    try:
        print(f"Bắt đầu ghi âm phòng LiveKit: {room_name}")
        egress_info = await lkapi.egress.start_room_composite_egress(egress_request)
        print(f"Ghi âm đã bắt đầu thành công. ID Egress: {egress_info.egress_id}")
    except Exception as e:
        print(f"LỖI: Không thể bắt đầu ghi âm. Vui lòng kiểm tra LiveKit API Key/Secret và cấu hình Egress. Chi tiết: {e}")
    return egress_info,s3_uri_of_recording
async def stop_recording(egress_info):
    if egress_info:
        try:
            await api.LiveKitAPI().egress.stop_egress(api.StopEgressRequest(egress_id=egress_info.egress_id))
        except Exception as e:
            print(f"Warning khi dừng egress: {e}")