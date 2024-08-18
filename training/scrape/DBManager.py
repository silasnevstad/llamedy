import sqlite3
import logging

class DBManager:
    def __init__(self, db_path, log_level=logging.INFO):
        self.db_path = db_path
        self.logger = self._setup_logger(log_level)
        self.conn = None
        self.cursor = None
        self.url_set = set()
        self._initialize_db()
        self._load_urls()

    def _setup_logger(self, log_level):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _initialize_db(self):
        self.logger.info(f"Initializing database: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS comedians (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                transcription TEXT
            )
        ''')
        self.conn.commit()

    def _load_urls(self):
        self.logger.info("Loading existing URLs into memory")
        self.cursor.execute("SELECT url FROM comedians")
        self.url_set = set(row[0] for row in self.cursor.fetchall())
        self.logger.info(f"Loaded {len(self.url_set)} URLs")
       
    def get_all_transcripts(self, tokenizer, max_tokens):
      self.logger.info(f"Retrieving all transcripts, trimmed to {max_tokens} tokens")
      self.cursor.execute("SELECT name, transcription FROM comedians WHERE transcription IS NOT NULL")
      results = self.cursor.fetchall()
      
      trimmed_transcripts = []
      for name, transcription in results:
          if transcription:
              # Tokenize the transcription
              tokens = tokenizer.encode(transcription, add_special_tokens=False)
              
              # Trim to max_tokens
              if len(tokens) > max_tokens:
                  tokens = tokens[:max_tokens]
                  trimmed_text = tokenizer.decode(tokens)
                  self.logger.warning(f"Trimmed transcript for {name} to {max_tokens} tokens")
              else:
                  trimmed_text = transcription
              
              trimmed_transcripts.append({
                  "name": name,
                  "text": trimmed_text
              })
      
      self.logger.info(f"Retrieved {len(trimmed_transcripts)} transcripts")
      return trimmed_transcripts 

    def add_comedian(self, name, url):
        if url in self.url_set:
            self.logger.info(f"URL already exists in database: {url}")
            return "exists"
        
        try:
            self.cursor.execute(
                "INSERT INTO comedians (name, url) VALUES (?, ?)",
                (name, url)
            )
            self.conn.commit()
            self.url_set.add(url)
            self.logger.info(f"Added new entry: {name} - {url}")
            return "added"
        except sqlite3.IntegrityError:
            self.logger.warning(f"Failed to add duplicate URL: {url}")
            return "error"
          
    def update_transcription(self, url, transcription):
        if url not in self.url_set:
            self.logger.warning(f"URL not found in database: {url}")
            return False

        if not transcription:
            self.logger.warning(f"Transcription is empty for URL: {url}")
            return False
        
        self.cursor.execute(
            "UPDATE comedians SET transcription = ? WHERE url = ?",
            (transcription, url)
        )
        self.conn.commit()
        self.logger.info(f"Updated transcription for URL: {url}")
        return True

    def get_untranscribed_urls(self):
        self.cursor.execute("SELECT name, url FROM comedians WHERE transcription IS NULL")
        return self.cursor.fetchall()

    def url_exists(self, url):
        return url in self.url_set
      
    def close(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
