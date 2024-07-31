import io
import wave
import json
import logging
from pytube import YouTube
from vosk import Model, KaldiRecognizer
import time
import random
from pydub import AudioSegment


class YouTubeTranscriber:
    def __init__(self, model_path, log_level=logging.INFO):
        self.model = Model(model_path)
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def transcribe_url(self, url, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self._try_transcribe(url)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = random.uniform(1, 5)
                    self.logger.info(f"Waiting {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All attempts failed for {url}")
                    return None

    def _try_transcribe(self, url):
        self.logger.info(f"Processing video from: {url}")
        try:
            yt = YouTube(url)
            self.logger.info(f"YouTube object created for {url}")
            
            # Get all available audio streams
            self.logger.info("Retrieving available audio streams...")
            streams = yt.streams.filter(only_audio=True)
            
            self.logger.info(f"Number of audio streams found: {len(streams)}")
            
            if not streams:
                self.logger.error("No audio streams found for this video.")
                raise ValueError("No audio streams found for this video.")

            # Log details of available audio streams
            for stream in streams:
                self.logger.info(f"Stream: itag={stream.itag}, abr={stream.abr}, mime_type={stream.mime_type}")

            # Choose the best audio stream (highest bitrate)
            selected_stream = streams.order_by('abr').desc().first()
            
            self.logger.info(f"Selected audio stream: itag={selected_stream.itag}, abr={selected_stream.abr}")
            
            self.logger.info(f"Downloading audio: {selected_stream.itag}")
            
            # Download to a BytesIO object
            buffer = io.BytesIO()
            selected_stream.stream_to_buffer(buffer)
            buffer.seek(0)
            
            self.logger.info("Audio download complete. Processing for transcription...")

            # Convert to WAV format for VOSK (if needed)
            wav_buffer = self._convert_to_wav(buffer, selected_stream.mime_type)
            
            self.logger.info("Transcribing audio...")
            rec = KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)

            results = []
            while True:
                data = wav_buffer.read(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.append(part_result)

            part_result = json.loads(rec.FinalResult())
            results.append(part_result)

            full_text = ' '.join([r['text'] for r in results if 'text' in r])

            self.logger.info("Transcription complete.")
            return full_text
        except Exception as e:
            self.logger.error(f"Error in _try_transcribe: {str(e)}")
            raise

    def _convert_to_wav(self, audio_buffer, mime_type):
        self.logger.info(f"Converting audio from {mime_type} to WAV...")
        
        # Determine the format from the mime type
        audio_format = mime_type.split('/')[-1]
        
        # Load the audio data
        audio = AudioSegment.from_file(audio_buffer, format=audio_format)
        
        # Export as WAV to a new buffer
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        self.logger.info("Audio conversion complete.")
        return wav_buffer
