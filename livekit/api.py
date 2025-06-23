# livekit/api.py
# Stub file to avoid ImportError
# livekit/api.py

import jwt

class AccessToken:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.grants = {}

    def with_grants(self, grants):
        self.grants.update(grants)
        return self

    def to_jwt(self):
        return jwt.encode(self.grants, self.api_secret, algorithm="HS256")

class VideoGrants(dict):
    def __init__(self, room_join=False, room=None, agent=False):
        super().__init__()
        self["room_join"] = room_join
        self["room"] = room
        self["agent"] = agent


class CreateRoomRequest:
    def __init__(self, name):
        self.name = name

class DeleteRoomRequest:
    def __init__(self, room):
        self.room = room

class RoomParticipantIdentity:
    def __init__(self, room, identity):
        self.room = room
        self.identity = identity

class CreateSIPParticipantRequest:
    def __init__(self, **kwargs):
        self.params = kwargs

class TransferSIPParticipantRequest:
    def __init__(self, **kwargs):
        self.params = kwargs

class SIPParticipantInfo:
    pass

class DeleteRoomResponse:
    pass

class LiveKitAPI:
    def __init__(self, ws_url=None, api_key=None, api_secret=None, session=None):
        self.room = self.RoomAPI()
        self.sip = self.SIPAPI()

    class RoomAPI:
        async def create_room(self, request):
            print(f"[stub] create_room({request.name})")
            return {}

        async def delete_room(self, request):
            print(f"[stub] delete_room({request.room})")
            return DeleteRoomResponse()

        async def get_participant(self, identity):
            print(f"[stub] get_participant(room={identity.room}, identity={identity.identity})")
            return {}

    class SIPAPI:
        async def create_sip_participant(self, request):
            print(f"[stub] create_sip_participant({request.params})")
            return SIPParticipantInfo()

        async def transfer_sip_participant(self, request):
            print(f"[stub] transfer_sip_participant({request.params})")
            return SIPParticipantInfo()

    async def aclose(self):
        print("[stub] LiveKitAPI aclose() called")
