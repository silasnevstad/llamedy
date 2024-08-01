import logging
from UrlParser import ComedianParser
from Transcribe import YouTubeTranscriber

class TranscriptionManager:
    def __init__(self, log_level, url_file_path, vosk_model_path):
        self.logger = self._setup_logger(log_level)
        self.url_file_path = url_file_path
        self.parser = ComedianParser(log_level=log_level)
        self.transcriber = YouTubeTranscriber(log_level=log_level)
        self.comedians_dict = {}

    def _setup_logger(self, log_level):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def parse_urls(self):
        self.logger.info(f"Parsing URLs from file: {self.url_file_path}")
        self.parser.parse_file(self.url_file_path)
        self.comedians_dict = self.parser.get_comedians()
        self.logger.info(f"Parsed {len(self.comedians_dict)} comedians")

    def transcribe_all(self):
        self.logger.info("Starting transcription for all comedians")
        for name, urls in self.comedians_dict.items():
            self.logger.info(f"Transcribing videos for {name}")
            for url in urls:
                self.logger.info(f"Transcribing: {url}")
                transcription = self.transcriber.transcribe_url(url)
                if transcription:
                    print(f"\nTranscription for {name} - {url}:")
                    print(transcription)
                    print("\n" + "="*50 + "\n")
                else:
                    print(f"\nTranscription failed for {name} - {url}\n")
        self.logger.info("Transcription complete for all comedians")

    def run(self):
        self.parse_urls()
        self.transcribe_all()

# Usage
if __name__ == "__main__":
    log_level = logging.INFO
    url_file_path = 'scrape/youtube_urls.txt'
    vosk_model_path = "models/vosk/vosk-model-en-us-0.42-gigaspeech"  # Replace with your VOSK model path
    
    manager = TranscriptionManager(log_level, url_file_path, vosk_model_path)
    manager.run()
