import os
import asyncio
import logging
import sys
import argparse
from pathlib import Path
from twilio.rest import Client
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.responses import Response

# Extend import path for livekit modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "livekit" / "agents"))

from livekit.agents import Agent
from livekit.agents.llm import OpenAILLM, ChatContext
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.tts.elevenlabs import ElevenLabsTTS
from livekit.agents.stt.openai import OpenAIWhisperSTT
from livekit.agents.voice.vad.silero import SileroVAD
from livekit.agents.llm.openai import FunctionTool

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

# ─── Logging ───
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load Env ───
env_path = Path(__file__).resolve().parent / "file.env"
if not load_dotenv(dotenv_path=env_path, override=True):
    logger.warning("⚠️ Could not load .env file from path: %s", env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is missing. Check file.env")

# API Keys
os.environ["OPENAI_API_KEY"] = api_key
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_API_KEY", "")
os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY", "")
os.environ["DEEPGRAM_API_KEY"] = os.getenv("DEEPGRAM_API_KEY", "")

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")
message_url = f"{PUBLIC_URL}/twiml/intro"

# LiveKit keys
for key in ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"]:
    if val := os.getenv(key):
        os.environ[key] = val

logger.info("✅ API keys loaded successfully.")

# ─── Sample Customer ───
test_customer = {
    "name": "Shreya",
    "phone": "+919634056866",
    "destination": "Meghalaya"
}

# ─── Knowledge Base ───
data_dir = Path("data")
persist_dir = Path("query-engine-storage")

try:
    if not persist_dir.exists():
        logger.info("📄 No index found. Creating new index from PDFs...")
        documents = SimpleDirectoryReader(str(data_dir)).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        logger.info("📁 Loading existing knowledge base...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
except Exception as e:
    logger.exception("❌ Failed to initialize knowledge base: %s", e)
    raise

# ─── Tools ───
async def query_kb(query: str) -> str:
    query_engine = index.as_query_engine(use_async=True)
    result = await query_engine.aquery(query)
    return str(result)

query_kb_tool = FunctionTool.from_defaults(fn=query_kb).to_tool()

# ─── Prompt ───
PROMPT = """
You are a polite and helpful Indian travel consultant who assists young Indian travelers...
... [prompt truncated for brevity]
"""

# ─── Travel Agent ───
class TravelAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=PROMPT,
            llm=OpenAILLM(model="gpt-4o", api_key=api_key),
            stt=OpenAIWhisperSTT(model="whisper-1"),
            tts=ElevenLabsTTS(voice_id="DpnM70iDHNHZ0Mguv6GJ"),
            vad=SileroVAD(),
            tools=[query_kb_tool],
        )

# ─── Twilio Call Function ───
def call_customer_via_twilio(to_number: str, message_url: str = message_url):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        logger.error("❌ Missing Twilio credentials. Cannot place call.")
        return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    try:
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=message_url
        )
        logger.info(f"📞 Call initiated to {to_number}. Call SID: {call.sid}")
    except Exception as e:
        logger.exception(f"❌ Failed to initiate call to {to_number}: {e}")

# ─── FastAPI ───
app = FastAPI()

@app.get("/twiml/intro")
def twiml_intro():
    twiml = f"""
    <Response>
        <Say voice=\"Polly.Raveena-Neural\" language=\"en-IN\">
            Hi {test_customer['name']}! Thanks for showing interest in our Meghalaya group trip.
            We’ve sent you some exciting package options on WhatsApp. If you'd like to talk to a trip specialist, just press 1.
        </Say>
        <Pause length=\"2\"/>
        <Say>
            Have a great day!
        </Say>
    </Response>
    """
    return Response(content=twiml.strip(), media_type="application/xml")

@app.get("/")
def root():
    return {"message": "LiveKit Travel Voicebot is running."}

@app.on_event("startup")
async def startup_event():
    if os.getenv("RUN_AGENT", "true").lower() == "true":
        logger.info("🚀 FastAPI app started. Launching background services...")
        asyncio.create_task(initialize_everything())
    else:
        logger.info("✅ FastAPI app started without agent (RUN_AGENT=false).")

async def initialize_everything():
    try:
        logger.info("🧠 Initializing LlamaIndex and LiveKit agent...")
        session = AgentSession()
        await session.start(agent=TravelAgent())
        logger.info("✅ Travel agent session started.")
        await session.say("Hi! I hope you're doing well. Is this a good time to chat about your travel plans?")
        logger.info("💬 Initial message sent.")
    except Exception as e:
        logger.exception("❌ Error during agent startup: %s", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--call", help="Phone number to call")
    args = parser.parse_args()

    if args.call:
        call_customer_via_twilio(to_number=args.call, message_url=message_url)
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
