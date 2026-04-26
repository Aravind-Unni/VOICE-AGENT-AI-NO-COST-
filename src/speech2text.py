import io
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

# Set up logging for observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalSTT:
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8", max_threads=2):
        """
        Initializes the faster-whisper model and a thread pool.
        
        Args:
            model_size (str): 'tiny.en' or 'base.en' for lowest latency.
            device (str): 'cpu' or 'cuda'.
            compute_type (str): 'int8' for CPU, 'float16' for GPU.
            max_threads (int): Limits concurrent transcription threads to prevent CPU overload.
        """
        logger.info(f"Loading faster-whisper '{model_size}' on {device}...")
        
        # Load the model into RAM/VRAM once
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # Initialize a dedicated thread pool for CPU-bound transcription
        self.executor = ThreadPoolExecutor(max_workers=max_threads)
        
        logger.info("STT model loaded and thread pool initialized.")

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        """
        The core synchronous logic that runs in the background thread.
        """
        audio_file = io.BytesIO(audio_bytes)
        
        # vad_filter=True prevents hallucinating text from static or breathing
        segments, info = self.model.transcribe(
            audio_file, 
            beam_size=5, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500) 
        )
        
        # Assemble the segments into a single string
        transcription = " ".join([segment.text for segment in segments])
        return transcription.strip()

    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Asynchronous wrapper for the FastAPI endpoint to call.
        Yields control back to the event loop while transcription happens.
        """
        loop = asyncio.get_running_loop()
        
        # Execute the blocking transcription logic in the thread pool
        transcription = await loop.run_in_executor(
            self.executor, 
            self._transcribe_sync, 
            audio_bytes
        )
        
        return transcription