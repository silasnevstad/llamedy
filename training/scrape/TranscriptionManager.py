import logging
import csv
from UrlParser import ComedianParser
from Transcribe import YouTubeTranscriber

class TranscriptionManager:
    def __init__(self, log_level, url_file_path, csv_file_path):
        self.logger = self._setup_logger(log_level)
        self.url_file_path = url_file_path
        self.csv_file_path = csv_file_path
        self.parser = ComedianParser(log_level=logging.ERROR)
        self.transcriber = YouTubeTranscriber(log_level=log_level)
        self.comedians_dict = {}
        self.new_urls = []  # List to store newly added URLs

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
        self.new_urls.clear()  # Clear any previously stored new URLs

        urls_added = 0
        urls_existing = 0

        with open(self.csv_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            existing_urls = set(row[1] for row in reader)

        for name, urls in self.comedians_dict.items():
            for url in urls:
                if url not in existing_urls:
                    self.new_urls.append((name, url))
                    urls_added += 1
                else:
                    urls_existing += 1

        self.logger.info(f"URL processing complete:")
        self.logger.info(f" - Added: {urls_added}")
        self.logger.info(f" - Already existed: {urls_existing}")

    def transcribe_all(self):
        self.logger.info(f"Starting transcription for {len(self.new_urls)} newly added videos")
        successful_transcriptions = 0
        failed_transcriptions = 0

        with open(self.csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for name, url in self.new_urls:
                self.logger.info(f"Transcribing: {url}")
                transcription = self.transcriber.transcribe_url(url)
                if transcription:
                    writer.writerow([name, url, transcription])
                    successful_transcriptions += 1
                    print(f"\nTranscription for {name} - {url}:")
                    print(transcription)
                    print("\n" + "="*50 + "\n")
                else:
                    failed_transcriptions += 1
                    self.logger.warning(f"Transcription failed for {name} - {url}")

        self.logger.info("Transcription process complete:")
        self.logger.info(f" - Successfully transcribed: {successful_transcriptions}")
        self.logger.info(f" - Failed transcriptions: {failed_transcriptions}")
        self.new_urls.clear()  # Clear the new_urls list after processing

    def run(self):
        self.parse_urls()
        self.transcribe_all()

# Usage
if __name__ == "__main__":
    log_level = logging.INFO
    url_file_path = 'scrape/youtube_urls.txt'
    csv_file_path = "comedians_transcriptions.csv"

    manager = TranscriptionManager(log_level, url_file_path, csv_file_path)
    manager.run()
