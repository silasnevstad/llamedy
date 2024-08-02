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
        self.transcription_dict = {}

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
                    self.transcription_dict[url] = transcription
                    print(f"\nTranscription for {name} - {url}:")
                    print(transcription)
                    print("\n" + "="*50 + "\n")
                else:
                    print(f"\nTranscription failed for {name} - {url}\n")
        self.logger.info("Transcription complete for all comedians")

    def get_transcriptions(self):
        return self.transcription_dict

    def load_transcriptions_from_file(self, file_path):
        self.logger.info(f"Loading transcriptions from file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    url, transcription = line.split(' - ')
                    self.transcription_dict[url] = transcription
            self.logger.info(f"Loaded {len(self.transcription_dict)} transcriptions")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading transcriptions: {str(e)}")

    def save_transcriptions_to_file(self, file_path):
        self.logger.info(f"Saving transcriptions to file: {file_path}")
        try:
            with open('scrape/' + file_path, 'w') as file:
                for url, transcription in self.transcription_dict.items():
                    file.write(f"{url} - {transcription}\n")
            self.logger.info(f"Saved {len(self.transcription_dict)} transcriptions")
        except Exception as e:
            self.logger.error(f"Error saving transcriptions: {str(e)}")

    def run(self):
        self.parse_urls()
        self.transcribe_all()
        self.save_transcriptions_to_file('transcriptions.txt')

# Usage
if __name__ == "__main__":
    log_level = logging.INFO
    url_file_path = 'scrape/youtube_urls.txt'
    vosk_model_path = "models/vosk/vosk-model-en-us-0.42-gigaspeech"  # Replace with your VOSK model path
    
    manager = TranscriptionManager(log_level, url_file_path, vosk_model_path)
    manager.run()
