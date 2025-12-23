from typing import Any
import time
import os
import logging
import ESL
from livekit.api import ListParticipantsRequest
from livekit.api import LiveKitAPI
from dotenv import load_dotenv
import json
import asyncio

load_dotenv()

from utils.async_cache_file import AsyncStaticExcelCache
async_cache = AsyncStaticExcelCache(cache_dir="./temps/excel")

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)



class RoutingAgent:
    def __init__(self, **kwargs: Any):
        self.con=ESL.ESLconnection(os.getenv("FS_IP", "localhost"), os.getenv("FS_PORT", "8000"), os.getenv("FS_PASSWORD", "123456789"))
        
        super().__init__(**kwargs)
    
    def _get_call_uuid(self,call_to_transfer,caller_num):
        """
        Get call uuid by two phone numbers in curent call.
        """
        
        response = self.con.api('show calls as json')
        try:
            calls_data = json.loads(response.getBody())
            for call in calls_data.get('rows', []):
                if (call.get('dest') == call_to_transfer and call.get('cid_num') == caller_num):
                    uuid = call.get('uuid')
                    return uuid
                elif (call.get('b_dest') == call_to_transfer and call.get('b_cid_num') == caller_num):
                    uuid = call.get('b_uuid')
                    return uuid
        except Exception as e:
            logger.error(f"Error in JSON call data: {e}")
            return None
            
        logger.info(f"Not found UUID for call from {call_to_transfer}.")
        return None


    async def _init_participants(self,room):
        async with LiveKitAPI() as lkapi:
            res = await lkapi.room.list_participants(
                ListParticipantsRequest(room=room.name)
            )
            for participant in res.participants:
                if participant.identity.startswith("sip_"):
                    sip_identity = participant.identity.split("_")[1]
                    return sip_identity

            logger.info("Not found participant have identity format 'sip_'.")
            return None
    
    async def routing_agent(self,ctx,phone_number,transfer_number):
        """
        Routing from curent agent to agent in other phone number .
        """

        if not self.con.connected():
            logger.error('Can not connect to FreeSWITCH.')

        caller_num = await self._init_participants(ctx.room)
        call_uuid=None
        for i in range(5):
            call_uuid= self._get_call_uuid(phone_number,caller_num)
            if call_uuid:
                break
            else:
                await asyncio.sleep(2)
        if not call_uuid:
            logger.error("Can not find UUID of current call to redirect.")

        transfer_command = f"uuid_transfer {call_uuid} {transfer_number}"

        response = self.con.api(transfer_command)

        if response and response.getBody().startswith('+OK'):
            logger.info(f"Redirect call from {phone_number} to {transfer_number} successfully!")
        else:
            logger.error(f"Routing call fail. Response from FreeSWITCH: {response.getBody()}")


routingagent = RoutingAgent()