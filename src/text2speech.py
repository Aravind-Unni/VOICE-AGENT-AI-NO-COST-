import edge_tts
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalTTS:
    def __init__(self, voice="en-US-AvaNeural"):
        """
        Initializes the Edge TTS wrapper.
        'en-US-AvaNeural' is a highly natural, fast female voice.
        """
        logger.info(f"Initialized Edge TTS with voice: {voice}")
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        """
        Asynchronously generates MP3 audio bytes.
        """
        communicate = edge_tts.Communicate(text, self.voice)
        audio_bytes = bytearray()
        
        # Stream the audio chunks into memory
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
                
        return bytes(audio_bytes)