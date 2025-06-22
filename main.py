import os
import asyncio
import logging
import sys
from pathlib import Path
from twilio.rest import Client
from fastapi import FastAPI
from dotenv import load_dotenv
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent / "livekit" / "agents"))


from livekit.agents import Agent
from livekit.agents.llm import OpenAILLM, ChatContext
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.tts.elevenlabs import ElevenLabsTTS
from livekit.agents.stt.openai import OpenAIWhisperSTT
from livekit.agents.voice.vad.silero import SileroVAD




from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

# ─── Set up Logging ───
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load Environment Variables ───
env_path = Path(__file__).resolve().parent / "file.env"
if not load_dotenv(dotenv_path=env_path, override=True):
    logger.warning("⚠️ Could not load .env file from path: %s", env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is missing. Check file.env")

# Set API keys for global usage
os.environ["OPENAI_API_KEY"] = api_key
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_API_KEY", "")
os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY", "")
os.environ["DEEPGRAM_API_KEY"] = os.getenv("DEEPGRAM_API_KEY", "")
# Assign Twilio credentials from env
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")



for key in ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"]:
    if val := os.getenv(key):
        os.environ[key] = val

logger.info("✅ API keys loaded successfully.")

# ─── Test Customer Details ───
test_customer = {
    "name": "Shreya",
    "phone": "+919634056866",  # Use E.164 format for Twilio
    "destination": "Meghalaya"
}

# ─── PDF Knowledge Base ───
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

def call_customer_via_twilio(to_number: str, message_url: str = "http://localhost:8000/twiml/intro"):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        logger.error("❌ Missing Twilio credentials. Cannot place call.")
        return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    try:
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=message_url  # <- now calling your FastAPI endpoint
        )
        logger.info(f"📞 Call initiated to {to_number}. Call SID: {call.sid}")
    except Exception as e:
        logger.exception(f"❌ Failed to initiate call to {to_number}: {e}")


    """Initiate a call to the customer using Twilio and play message at URL."""
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



from livekit.agents.llm.openai import FunctionTool  # or correct path

async def query_kb(query: str) -> str:
    """Answer a travel-related question from the knowledge base."""
    query_engine = index.as_query_engine(use_async=True)
    result = await query_engine.aquery(query)
    return str(result)

query_kb_tool = FunctionTool.from_defaults(fn=query_kb).to_tool()


print("✅ FunctionTool created:", query_kb_tool)

# ─── Prompt ───

PROMPT = """
You are a polite and helpful Indian travel consultant who assists young Indian travelers (Gen Z and Millennials) in planning group tours. You communicate in clear, simple English and use our travel knowledge base to recommend suitable packages based on user preferences.

"Thanks for reaching out about our group tours. I see you're interested in a trip to Meghalaya, planning to travel around July 6th with three companions, and your overall budget is ₹10,000. Does that sound right?"

(wait)

"I will ask you a few quick questions so I can find the best options for you."

(wait)

Ask these questions one at a time. Acknowledge each answer briefly before moving to the next:

1. "What kind of experience are you looking for — adventure, culture, or relaxation?"
2. "What type of hotel do you prefer — 3-star, 4-star, or 5-star?"
3. "What kind of vehicle would you be more comfortable in — SUV or sedan?"
4. "Do you prefer vegetarian or non-vegetarian meals?"

Once the preferences are collected:

"Thanks for sharing that. Let me find the best options from our travel packages."

(Use the knowledge base to fetch and suggest a package based on destination, travel dates, budget, and preferences.)

If a relevant package is found:

"Based on what you've told me, here's a package that might suit you well: [insert package details from the knowledge base — name, price (like 10000 rupees), hotel category, vehicle, meal plan, and key highlights]. Would you like to know more about this option?"

(wait)

If the customer is interested:

"Great! There are limited spots left. Can our booking specialist give you a call today to help you secure your spot? They are available until 8 PM."

(wait)

"Do you have any specific questions you had like the specialist to answer?"

If the customer says they are not interested:

"No worries at all. If you ever want to explore again, we will be here with exciting group trips. Take care!"

If the customer goes off-topic:

"That's interesting! Just to stay on track — I am here to help you with your travel plans. Shall we continue finding your package?"

If there are no more questions:

"Thanks for chatting with me! I have noted your preferences and passed them along to our team."
"""

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

# ─── FastAPI ───
app = FastAPI()

from fastapi.responses import Response

@app.get("/twiml/intro")
def twiml_intro():
    twiml = f"""
    <Response>
        <Say voice="Polly.Raveena-Neural" language="en-IN">
            Hi {test_customer['name']}! Thanks for showing interest in our Meghalaya group trip.
            We’ve sent you some exciting package options on WhatsApp. If you'd like to talk to a trip specialist, just press 1.
        </Say>
        <Pause length="2"/>
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
                # Test Twilio Call to Customer

    logger.info(f"📞 Initiating test call to {test_customer['name']} about {test_customer['destination']} trip...")
    call_customer_via_twilio(test_customer["phone"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

