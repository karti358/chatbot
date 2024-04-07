"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base_tokenizer import Tokenizer, get_stats, merge
import os
# import pyspark.pandas as pd
import sqlite3
from numba import njit

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizerLarge(Tokenizer):

    def __init__(self, vocab_size = 276,special_tokens = {}, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        assert vocab_size >= 256
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        print(f"The tokenizer will require {self.num_merges} iteraions of merges")

        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = special_tokens if len(special_tokens) > 0 else {"<|newlinechar|>" : self.vocab_size,
                               "<|start|>" : self.vocab_size + 1,
                               "<|end|>" : self.vocab_size + 2,
                               "<|padding|>": self.vocab_size + 3
                              }
        self.inverse_special_tokens = self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
    def chunkify(self, files: list, data_dir: str):
        self.sql_transaction = []
        self.counter = 0

        self.chunk_db_path = data_dir + "/" + "full_chunks.db"
        connection = sqlite3.connect(self.chunk_db_path)
        cursor = connection.cursor()

        def create_chunk_ids_table(cursor):
            sql = f"""CREATE TABLE IF NOT EXISTS chunk_ids_{0}
                     (
                        id INTEGER PRIMARY KEY,
                        text TEXT
                     )"""
            return cursor.execute(sql)
        
        def transaction_builder(sql,cursor, conn):
            self.sql_transaction.append(sql)
            if len(self.sql_transaction) > 100000:
                cursor.execute("BEGIN TRANSACTION;")
                for s in self.sql_transaction:
                    try:
                        cursor.execute(s)
                    except:
                        pass
                print(f"Inserted {self.counter} into chunk_ids_{0}...")
                conn.commit()
                self.sql_transaction = []
        
        def flush_pending(cursor, conn):
            cursor.execute("BEGIN TRANSACTION;")
            for s in self.sql_transaction:
                try:
                    cursor.execute(s)
                except:
                    pass
            print(f"Flushed pending !! ...")
            conn.commit()
            self.sql_transaction = []

        def insert_into_chunk_ids(chunks, cursor, connection):
            try:
                ids = map(lambda x: ",".join( map(lambda y: str(int(y)), x)), map(lambda x: x.encode("utf-8"), chunks))
                sql = f"INSERT INTO chunk_ids_{0} (text) VALUES (" +"),(".join( map(lambda x: f"'{x}'", ids)) + ");"
                self.counter += 1
                transaction_builder(sql,cursor, connection)
            except:
                print(f"Problem in inserting {chunks}")
                print(list(map(lambda x: ",".join( map(lambda y: str(int(y)), x)), map(lambda x: x.encode("utf-8"), chunks))))
                print("\n\n\n\n\n")
        
        cursor = create_chunk_ids_table(cursor)
        for file in files:
            print("Processing-->", file)
            #Open the input file
            with open(file, "r", encoding = "utf-8") as f:
                #Iterate through the input file
                for line in f:
                    #Find the chunks in each line
                    chunks = re.findall(self.compiled_pattern, line)
                    #Write each chunk to chunks file
                    insert_into_chunk_ids(chunks, cursor, connection)
            
            if len(self.sql_transaction) > 0:
                flush_pending(cursor, connection)
            print("Done processing-->", file)
            print("\n\n\n")     

        connection.close() 
    
    def chunk_train(self, verbose = False):
        connection = sqlite3.connect(self.chunk_db_path)
        old_cursor = connection.cursor()

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        def create_chunk_ids_table(i, cursor):
            sql = f"""CREATE TABLE IF NOT EXISTS chunk_ids_{i}
                     (
                        id INTEGER PRIMARY KEY,
                        text TEXT
                     )"""
            return cursor.execute(sql)
        
        self.sql_transaction = []
        self.counter = 0

        def transaction_builder(i, sql, cursor, conn):
            self.sql_transaction.append(sql)
            if len(self.sql_transaction) > 100000:
                cursor.execute("BEGIN TRANSACTION;")
                for s in self.sql_transaction:
                    try:
                        cursor.execute(s)
                    except:
                        pass
                print(f"Inserted {self.counter} completed in chunk_ids_{i}")
                conn.commit()
                self.sql_transaction = []

        def insert_into_chunk_ids(i, ids, cursor, connection):
            try:
                temp = ",".join(map(lambda x: str(x), ids))
                sql = f"INSERT INTO chunk_ids_{i} (text) VALUES (" + f"'{temp}'" + ")"
                # print(sql)
                self.counter += 1
                transaction_builder(i, sql,cursor, connection)
            except:
                print(f"Problem in inserting {ids}")
        
        def flush_pending(i, cursor, conn):
            cursor.execute("BEGIN TRANSACTION;")
            for s in self.sql_transaction:
                try:
                    cursor.execute(s)
                except:
                    pass
            print("Flushed the pending...")
            conn.commit()
            self.sql_transaction = []
        
        # @njit
        def process_stats(old_cursor):
            stats = dict()

            print(old_cursor.execute(f"SELECT count(*) FROM chunk_ids_{i-1}").fetchall())
            old_cursor.execute(f"SELECT * FROM chunk_ids_{i-1}")

            rows = old_cursor.fetchmany(100000)
            rows_counter = len(rows)
            while len(rows) > 0:
                for row in rows:
                    #Convert str to ints
                    chunk_ids = list(map(int, row[1].split(",")))
                    #Get stats for the present chunk ids
                    get_stats(chunk_ids, stats)
                    # print(row)
                print(f"Processed {rows_counter} rows")
                rows = old_cursor.fetchmany(100000)
                rows_counter += len(rows)
            return stats
        
        # @njit
        def update_ids(i, old_cursor, connection):
            new_cursor = connection.cursor()
            new_cursor = create_chunk_ids_table(i, new_cursor)

            self.counter = 0

            old_cursor.execute(f"SELECT * FROM chunk_ids_{i-1}")
            rows = old_cursor.fetchmany(100000)
            rows_counter = len(rows)
            while len(rows) > 0:
                for row in rows:
                    #Convert str to ints
                    chunk_ids = list(map(int, row[1].split(",")))
                    #Update the chunk ids
                    temp_ids = merge(chunk_ids, pair, idx)
                    #Wrtite the updated chunk ids
                    insert_into_chunk_ids(i, temp_ids, new_cursor, connection)
                print(f"Processed {rows_counter} rows")
                rows = old_cursor.fetchmany(100000)
                rows_counter += len(rows)
            if len(self.sql_transaction) > 0:
                flush_pending(i, new_cursor, connection)
            
            return new_cursor

        for i in range(1, self.num_merges + 1):
            print(f"Starting merge {i}")

            #get stats
            stats = process_stats(old_cursor)            

            #Get the most occuring pair
            pair = max(stats, key=stats.get)
            #Define new index
            idx = 256 + i

            #Open the old chunk_ids file and new_chunk_ids file
            new_cursor = update_ids(i, old_cursor, connection)

            #Update merges dictionary
            merges[pair] = idx
            #Update the vocab dictionary
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            #Delete the old chunk file
            # os.remove(self.chunk_ids)
            old_cursor.execute(f"DROP TABLE chunk_ids_{i-1}")
        
            #Update the chunk_ids file to the new chunk ids file
            old_cursor = new_cursor

            if verbose:
                print(f"merge {i+1}/{self.num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences \n\n\n")

        connection.close()
        self.merges = merges
        self.vocab = vocab
        
    def train(self, files: str,data_dir: str, verbose=False):
        self.chunkify(files, data_dir)
        self.chunk_train(verbose = verbose)

        print(self.merges, self.vocab)

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids