import logging
from collections import defaultdict

class ComedianParser:
    def __init__(self, log_level=logging.INFO):
        self.comedians = defaultdict(set)  # Using set to automatically handle duplicates
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def parse_file(self, file_path):
        self.logger.info(f"Parsing file: {file_path}")
        current_comedian = None
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('& '):
                        current_comedian = line[2:]
                    elif line.startswith('http'):
                        if current_comedian:
                            self.comedians[current_comedian].add(line)
                        else:
                            self.logger.warning(f"Found URL without associated comedian: {line}")

            self.log_statistics()
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
        except Exception as e:
            self.logger.error(f"Error parsing file: {str(e)}")

    def log_statistics(self):
        total_comedians = len(self.comedians)
        self.logger.info(f"Total comedians scraped: {total_comedians}")
        for comedian, urls in self.comedians.items():
            self.logger.info(f"Comedian: {comedian}, Number of unique videos: {len(urls)}")
        total_videos = sum(len(urls) for urls in self.comedians.values())
        self.logger.info(f"Total unique videos scraped: {total_videos}")

    def get_comedians(self):
        return {comedian: list(urls) for comedian, urls in self.comedians.items()}

