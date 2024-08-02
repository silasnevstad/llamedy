import logging
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import pytube.exceptions

class YouTubeTranscriber:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_captions(self, url, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self._try_get_captions(url)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying...")
                else:
                    self.logger.error(f"All attempts failed for {url}")
                    return None

    def _try_get_captions(self, url):
        self.logger.info(f"Processing video from: {url}")
        try:
            yt = YouTube(url)
            self.logger.info(f"YouTube object created for {url}")
            
            video_id = yt.video_id
            self.logger.info(f"Fetching captions for video ID: {video_id}")
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            
            if not transcript:
                self.logger.error("No English captions found for this video.")
                return None

            full_text = ' '.join([entry['text'] for entry in transcript])
            
            self.logger.info("Captions fetched successfully.")
            return full_text
        except Exception as e:
            self.logger.error(f"Error in _try_get_captions: {str(e)}")
            raise

    def transcribe_url(self, url):
        captions = self.get_captions(url)
        if captions:
          return captions
        else:
          self.logger.ERROR("Failed to transcribe video: {url}")
