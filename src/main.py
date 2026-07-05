import os
import asyncio
import threading
import random
import uvicorn
from dotenv import load_dotenv

# FastAPI & LiveKit Token Auth
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from livekit.api import AccessToken, VideoGrants

# LiveKit Agents & Plugins (v1.x imports)
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, Agent, AgentSession, function_tool
from livekit.plugins import deepgram, elevenlabs, openai

# Import your existing RAG engine
from rag_engine import initialize_support_rag_pipeline

load_dotenv()

# Boot up the RAG Pipeline globally so it is ready in memory
rag_pipeline = initialize_support_rag_pipeline()

# In LiveKit v1.x, we subclass Agent and use @function_tool
class ZendeskSupportAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful, conversational Zendesk Voice Support Agent. "
                "When the user asks a technical or support question, you MUST use the "
                "`ask_knowledgebase` tool to get the accurate answer. Read the tool's response "
                "naturally to the user. Keep your responses concise and conversational."
            )
        )

    @function_tool(description="Use this tool to find answers in the Zendesk Support manual.")
    async def ask_knowledgebase(self, user_query: str):
        """
        Passes the user's question to your existing RAG engine.
        Now fully async-compatible to prevent audio freezing!
        """
        print(f"🔍 Passing query to RAG Engine: {user_query}")
        
        # Run the synchronous RAG pipeline in a background thread
        result = await asyncio.to_thread(
            rag_pipeline.invoke,
            {
                "original_question": user_query,
                "chat_history": [],
                "session_id": "livekit_call"
            }
        )
        
        # Extract the answer and FORCE it to be a plain string
        raw_answer = result.get("generation", "I couldn't find an answer to that.")
        return str(raw_answer)


# The main entrypoint for the LiveKit Agent
async def entrypoint(ctx: JobContext):
    # Connect to the LiveKit Room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print("📞 Agent connected to LiveKit Room!")

    # Instantiate your new Agent subclass
    agent = ZendeskSupportAgent()
    
    # Build the new v1.x AgentSession
    session = AgentSession(
        stt=deepgram.STT(), 
        tts=elevenlabs.TTS(), 
        llm=openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )
    )

    # Start the session
    await session.start(agent=agent, room=ctx.room)
    
    # Greet the user when they connect
    await asyncio.sleep(1)
    await session.generate_reply(
        instructions="Say exactly this: 'Hi there! I am the Zendesk Support Assistant. How can I help you today?'"
    )


app = FastAPI()

@app.get("/get_token")
def generate_token():
    """Securely generates a LiveKit token using your hidden .env variables."""
    
    # Generate a unique room name every time to prevent Zombie Rooms
    unique_room = f"support-room-{random.randint(1000, 9999)}"
    
    token = (
        AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
        .with_identity("browser-user")
        .with_name("Customer")
        .with_grants(VideoGrants(room_join=True, room=unique_room))
    ).to_jwt()
    
    return {
        "url": os.getenv("LIVEKIT_URL"),
        "token": token
    }

@app.get("/")
def serve_html():
    """Serves your frontend interface."""
    html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

def run_fastapi():
    """Runs Uvicorn server in a background thread."""
    # CHANGED PORT TO 8080
    print("🌐 Starting FastAPI Web Server on http://localhost:8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")



if __name__ == "__main__":
    # Start FastAPI in a background daemon thread so it runs alongside LiveKit
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    import sys
    if len(sys.argv) == 1:
        sys.argv.append("dev") # Default to dev mode if no args are passed

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))