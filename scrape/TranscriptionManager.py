import logging
from UrlParser import ComedianParser
from Transcribe import YouTubeTranscriber
from DBManager import DBManager

class TranscriptionManager:
    def __init__(self, log_level, url_file_path, db_path):
        self.logger = self._setup_logger(log_level)
        self.url_file_path = url_file_path
        self.parser = ComedianParser(log_level=log_level)
        self.transcriber = YouTubeTranscriber(log_level=log_level)
        self.db_manager = DBManager(db_path, log_level=log_level)
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

        urls_added = 0
        urls_existing = 0
        urls_error = 0

        for name, urls in self.comedians_dict.items():
            for url in urls:
                result = self.db_manager.add_comedian(name, url)
                if result == "added":
                    urls_added += 1
                elif result == "exists":
                    urls_existing += 1
                    self.logger.info(f"URL already in database (not added): {url}")
                else:
                    urls_error += 1

        self.logger.info(f"URL processing complete:")
        self.logger.info(f"  - Added: {urls_added}")
        self.logger.info(f"  - Already existed: {urls_existing}")
        self.logger.info(f"  - Failed due to error: {urls_error}")

    def transcribe_all(self):
        self.logger.info("Starting transcription for untranscribed videos")
        untranscribed = self.db_manager.get_untranscribed_urls()
        total_untranscribed = len(untranscribed)
        successful_transcriptions = 0
        failed_transcriptions = 0

        for name, url in untranscribed:
            self.logger.info(f"Transcribing: {url}")
            transcription = self.transcriber.transcribe_url(url)
            if transcription:
                self.db_manager.update_transcription(url, transcription)
                successful_transcriptions += 1
                print(f"\nTranscription for {name} - {url}:")
                print(transcription)
                print("\n" + "="*50 + "\n")
            else:
                failed_transcriptions += 1
                print(f"\nTranscription failed for {name} - {url}\n")

        self.logger.info("Transcription process complete:")
        self.logger.info(f"  - Total untranscribed videos: {total_untranscribed}")
        self.logger.info(f"  - Successfully transcribed: {successful_transcriptions}")
        self.logger.info(f"  - Failed transcriptions: {failed_transcriptions}")

    def run(self):
        self.parse_urls()
        self.transcribe_all()
        self.db_manager.close()

# Usage
if __name__ == "__main__":
    log_level = logging.INFO
    url_file_path = 'scrape/youtube_urls.txt'
    db_path = "comedians.db"
    
    manager = TranscriptionManager(log_level, url_file_path, db_path)
    manager.run()