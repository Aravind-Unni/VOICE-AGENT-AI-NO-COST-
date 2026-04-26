import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Import your STT, RAG, and NEW TTS modules
from speech2text import LocalSTT
from rag_engine import initialize_support_rag_pipeline
from text2speech import LocalTTS

app = FastAPI()

# Global variables to hold our heavy models in memory
stt = None
rag_pipeline = None
tts = None

@app.on_event("startup")
async def startup_event():
    global stt, rag_pipeline, tts
    print("⚙️ Booting up Voice AI Server...")

    try:
        # 1. Initialize the local STT model
        stt = LocalSTT(model_size="base.en", device="cpu")
        
        # 2. Initialize the Support RAG Pipeline
        rag_pipeline = initialize_support_rag_pipeline()
        
        # 3. Initialize Edge TTS
        tts = LocalTTS(voice="en-US-AvaNeural")
        
        print("✅ Server successfully connected to STT, RAG, and TTS Models!")
        
    except Exception as e:
        print(f"❌ Critical Error during Pipeline Setup: {e}")

@app.get("/")
async def get_frontend():
    """Reads the local index.html file and serves it to the browser."""
    html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
    
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found!</h1><p>Make sure index.html is in the same folder as main.py.</p>", 
            status_code=404
        )

@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📞 User connected to the voice stream.")
    
    # --- IN-MEMORY CHAT HISTORY ---
    # This list exists only for the duration of this specific WebSocket connection.
    conversation_history = []
    
    try:
        while True:
            # 1. Wait for audio bytes from the user's microphone
            audio_bytes = await websocket.receive_bytes()
            
            # 2. Transcribe the audio
            transcription = await stt.transcribe(audio_bytes)
            
            if transcription:
                print(f"🗣️ User said: {transcription}")
                
                # 3. Pass the transcription and the history to the RAG Engine
                result = rag_pipeline.invoke({
                    "original_question": transcription,
                    "chat_history": conversation_history,
                    "session_id": "active_voice_call"
                })
                
                ai_response = result.get("generation", "I'm having trouble processing that right now.")
                print(f"🤖 AI Response: {ai_response}")
                
                # 4. Update the in-memory history for the next conversational turn
                conversation_history.append({"role": "User", "content": transcription})
                conversation_history.append({"role": "AI", "content": ai_response})
                
                # 5. Send TEXT to the frontend instantly (updates the UI)
                await websocket.send_text(f"<b>You:</b> {transcription}<br><b>Agent:</b> {ai_response}")
                
                # 6. Generate AUDIO and send the binary bytes to the frontend
                ai_audio_bytes = await tts.synthesize(ai_response)
                
                # --- NEW DEBUG LOG ---
                print(f"🎵 Generated {len(ai_audio_bytes)} bytes of audio data!") 
                
                await websocket.send_bytes(ai_audio_bytes)
                
    except WebSocketDisconnect:
        print("🛑 User hung up / disconnected.")
    except Exception as e:
        print(f"❌ Error processing audio or RAG: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)