from motor.motor_asyncio import AsyncIOMotorClient

from .config import MONGODB_DB, MONGODB_URI

client = AsyncIOMotorClient(MONGODB_URI)
db = client[MONGODB_DB]
