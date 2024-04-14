// use sqlite;
use regex;
use std::collections::HashMap;
// use std::io::Error;
use std::vec::Vec;
// use std::rc::Rc;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::format;

const GPT2_SPLIT_PATTERN: &str = r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;
//const GPT4_SPLIT_PATTERN: &str = r#"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"#;
const GPT4_SPLIT_PATTERN: &str = r#"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+"#;

#[derive(Debug, Clone)]
struct Tokenizer {
    vocab_size: i32,
    num_merges: i32,
    pattern: String,
    compiled_pattern: regex::Regex,
    special_tokens: HashMap<String, i32>,
    inverse_special_tokens: HashMap<String, i32>,

    chunk_db_path: String,
    sql_transaction: Vec<String>,
    counter: i32
}

impl Tokenizer {
    fn chunkify(mut_self: &RefCell<Self>, files: Vec<String>, data_dir: String) {
        mut_self.borrow_mut().chunk_db_path = data_dir + "/chunks.db";
        mut_self.borrow_mut().sql_transaction = Vec::new();
        mut_self.borrow_mut().counter = 0;


        let create_chunk_ids_table = |mut_self_: &mut RefCell<Self>, conn: &RefCell<&sqlite::Connection>| {
            let sql = "CREATE TABLE IF NOT EXISTS chunk_ids_0
                     (
                        id INTEGER PRIMARY KEY,
                        text TEXT
                     );";
            conn.borrow_mut().execute(sql).unwrap();            
        };

        let transaction_builder = |mut_self_: &mut RefCell<Self>, sql: String, conn: &RefCell<&sqlite::Connection>| -> &mut RefCell<Self> {
            println!("Inside builder");
            mut_self_.borrow_mut().sql_transaction.push(sql.clone());
            println!("{:?}", mut_self_.borrow().sql_transaction);

            if mut_self_.borrow().sql_transaction.len() > 100 {
                println!("Hmmm Fuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuck");
                conn.borrow().execute("BEGIN TRANSATION;").unwrap();
                for s in mut_self_.borrow_mut().sql_transaction.iter() {
                   conn.borrow().execute(s).unwrap();
                }
                // conn.commit()
                println!("Inserted {} into chunk_ids_0", mut_self_.borrow().counter);
                mut_self_.borrow_mut().sql_transaction.clear();
            }    
            // conn.borrow_mut().execute(sql).unwrap();  
            return mut_self_;
        };

        let flush_pending = |mut_self_: &RefCell<Self>, conn: &RefCell<&sqlite::Connection>| {
            let _ = conn.borrow().execute("BEGIN TRANSACTION;");

            for s in mut_self_.borrow_mut().sql_transaction.iter(){
                conn.borrow().execute(s).unwrap();
            }
            println!("Flushed pending !! ...");
            // conn.commit()
            mut_self_.borrow_mut().sql_transaction.clear();
        };

        let insert_into_chunk_ids = |mut_self_: &mut RefCell<Self>, chunks: Vec<String>,conn: &RefCell<&sqlite::Connection>| {
            let ids: Vec<String> = chunks.into_iter().map(|x| {
                let temp_ids = x.as_bytes();
                let temp_str: Vec<String> = temp_ids.into_iter().map(|y| {
                    y.to_string()
                }).collect();
                // let final_str = temp_str.join(",");
                let ret_str = format!("'{}'", temp_str.join(","));
                ret_str
            }).collect();

            let sql = "INSERT INTO chunk_ids_0 (text) VALUES (".to_string() + &ids.join("),(") + ");";
            // println!("{}", sql);
            mut_self_.borrow_mut().counter += 1;

            // let new_mut_self = mut_self.clone();
            // let new_conn = RefCell::new(conn.borrow_mut().clone());
            // let new_conn = conn.clone();
            mut_self_ = transaction_builder(mut_self_, sql, conn);
            // println!("SQL TRANS -> {:?}", mut_self.borrow().sql_transaction);
        };

        let connection = sqlite::open(mut_self.borrow().chunk_db_path.clone()).unwrap();

        let mut temp_mut_self = &mut mut_self.clone();
        let temp_conn = &RefCell::new(&connection);
        create_chunk_ids_table(temp_mut_self, temp_conn);
        
        for file in files {
            println!("Processing--> {file}");

            let f = File::open(file.clone()).unwrap();
            let reader = BufReader::new(f);

            let mut local_cnt = 0;
            for line in reader.lines() {
                local_cnt += 1;
                let chunks: Vec<_> = mut_self.borrow().compiled_pattern.find_iter(line.unwrap().as_str()).map(|x| {
                    x.as_str().to_string()
                }).collect();

                let mut new_mut_self = &mut mut_self.clone();
                let conn = &RefCell::new(&connection);
                insert_into_chunk_ids(new_mut_self, chunks, conn);

                println!("for current line -> {}", mut_self.borrow().sql_transaction.len());
            }
            if mut_self.borrow().sql_transaction.len() > 0 {
                let new_mut_self = &mut_self.clone();
                let conn = &RefCell::new(&connection);
                flush_pending(new_mut_self, conn);
            }
            println!("Done processing ... {}, lines -> {local_cnt}", file);
            print!("\n\n\n");
        }

    }
}



fn init(vocab_size: i32 , special_tokens: Option<HashMap<String, i32>> , pattern: Option<String>) -> Tokenizer{
    let vocab_size_ = vocab_size;
    assert!(vocab_size > 256);
    let num_merges_ = vocab_size - 256;

    let special_tokens_ = if special_tokens.is_none() {
        HashMap::from([ (String::from("<|newlinechar|>"),  vocab_size),
        (String::from("<|start|>") , vocab_size + 1),
        (String::from("<|end|>") , vocab_size + 2),
        (String::from("<|padding|>"), vocab_size + 3)
        ])
    }else {
        special_tokens.unwrap().clone()
    };

    let mut inverse_special_tokens_: HashMap<String, i32> = HashMap::new();
    for (k, v) in special_tokens_.iter() {
        inverse_special_tokens_.insert( (*k).clone(),  *v);
    }

    let pattern_ = if pattern.is_none() {
        GPT4_SPLIT_PATTERN.to_string()
    } else {
        pattern.unwrap()
    };

    let compiled_pattern_: regex::Regex = regex::Regex::new(pattern_.as_str()).unwrap();

   return Tokenizer {
        vocab_size: vocab_size_,
        num_merges : num_merges_,
        pattern: pattern_,
        compiled_pattern: compiled_pattern_,
        special_tokens: special_tokens_,
        inverse_special_tokens: inverse_special_tokens_,
        chunk_db_path: "".to_string(),
        sql_transaction: Vec::new(),
        counter: 0 as i32
    };

}

fn main () {
    let mut tok: Tokenizer = init(276, None, None);
    tok.sql_transaction.push("Hello".to_string());
    println!("{:?}", tok);
    
    let files = Vec::from([String::from("/home/thrasher/Class/chatbot/data/validation_data/RC_2017-01.txt")]);
    let data_dir = String::from("/home/thrasher/Class/chatbot/data");

    let mut_self = &RefCell::new(tok);
    Tokenizer::chunkify(mut_self, files, data_dir);
}





        
//         cursor = create_chunk_ids_table(cursor)
//         for file in files:
//             print("Processing-->", file)
//             #Open the input file
//             with open(file, "r", encoding = "utf-8") as f:
//                 #Iterate through the input file
//                 for line in f:
//                     #Find the chunks in each line
//                     chunks = re.findall(self.compiled_pattern, line)
//                     #Write each chunk to chunks file
//                     insert_into_chunk_ids(chunks, cursor, connection)
            
//             if len(self.sql_transaction) > 0:
//                 flush_pending(cursor, connection)
//             print("Done processing-->", file)
//             print("\n\n\n")     

//         connection.close() 
    
//     def chunk_train(self, verbose = False):
//         connection = sqlite3.connect(self.chunk_db_path)
//         old_cursor = connection.cursor()

//         merges = {}
//         vocab = {idx: bytes([idx]) for idx in range(256)}

//         def create_chunk_ids_table(i, cursor):
//             sql = f"""CREATE TABLE IF NOT EXISTS chunk_ids_{i}
//                      (
//                         id INTEGER PRIMARY KEY,
//                         text TEXT
//                      )"""
//             return cursor.execute(sql)
        
//         self.sql_transaction = []
//         self.counter = 0

//         def transaction_builder(i, sql, cursor, conn):
//             self.sql_transaction.append(sql)
//             if len(self.sql_transaction) > 100000:
//                 cursor.execute("BEGIN TRANSACTION;")
//                 for s in self.sql_transaction:
//                     try:
//                         cursor.execute(s)
//                     except:
//                         pass
//                 print(f"Inserted {self.counter} completed in chunk_ids_{i}")
//                 conn.commit()
//                 self.sql_transaction = []

//         def insert_into_chunk_ids(i, ids, cursor, connection):
//             try:
//                 temp = ",".join(map(lambda x: str(x), ids))
//                 sql = f"INSERT INTO chunk_ids_{i} (text) VALUES (" + f"'{temp}'" + ")"
//                 # print(sql)
//                 self.counter += 1
//                 transaction_builder(i, sql,cursor, connection)
//             except:
//                 print(f"Problem in inserting {ids}")
        
//         def flush_pending(i, cursor, conn):
//             cursor.execute("BEGIN TRANSACTION;")
//             for s in self.sql_transaction:
//                 try:
//                     cursor.execute(s)
//                 except:
//                     pass
//             print("Flushed the pending...")
//             conn.commit()
//             self.sql_transaction = []
        
//         # @njit
//         def process_stats(old_cursor):
//             stats = dict()

//             print(old_cursor.execute(f"SELECT count(*) FROM chunk_ids_{i-1}").fetchall())
//             old_cursor.execute(f"SELECT * FROM chunk_ids_{i-1}")

//             rows = old_cursor.fetchmany(100000)
//             rows_counter = len(rows)
//             while len(rows) > 0:
//                 for row in rows:
//                     #Convert str to ints
//                     chunk_ids = list(map(int, row[1].split(",")))
//                     #Get stats for the present chunk ids
//                     get_stats(chunk_ids, stats)
//                     # print(row)
//                 print(f"Processed {rows_counter} rows")
//                 rows = old_cursor.fetchmany(100000)
//                 rows_counter += len(rows)
//             return stats
        
//         # @njit
//         def update_ids(i, old_cursor, connection):
//             new_cursor = connection.cursor()
//             new_cursor = create_chunk_ids_table(i, new_cursor)

//             self.counter = 0

//             old_cursor.execute(f"SELECT * FROM chunk_ids_{i-1}")
//             rows = old_cursor.fetchmany(100000)
//             rows_counter = len(rows)
//             while len(rows) > 0:
//                 for row in rows:
//                     #Convert str to ints
//                     chunk_ids = list(map(int, row[1].split(",")))
//                     #Update the chunk ids
//                     temp_ids = merge(chunk_ids, pair, idx)
//                     #Wrtite the updated chunk ids
//                     insert_into_chunk_ids(i, temp_ids, new_cursor, connection)
//                 print(f"Processed {rows_counter} rows")
//                 rows = old_cursor.fetchmany(100000)
//                 rows_counter += len(rows)
//             if len(self.sql_transaction) > 0:
//                 flush_pending(i, new_cursor, connection)
            
//             return new_cursor

//         for i in range(1, self.num_merges + 1):
//             print(f"Starting merge {i}")

//             #get stats
//             stats = process_stats(old_cursor)            

//             #Get the most occuring pair
//             pair = max(stats, key=stats.get)
//             #Define new index
//             idx = 256 + i

//             #Open the old chunk_ids file and new_chunk_ids file
//             new_cursor = update_ids(i, old_cursor, connection)

//             #Update merges dictionary
//             merges[pair] = idx
//             #Update the vocab dictionary
//             vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

//             #Delete the old chunk file
//             # os.remove(self.chunk_ids)
//             old_cursor.execute(f"DROP TABLE chunk_ids_{i-1}")
        
//             #Update the chunk_ids file to the new chunk ids file
//             old_cursor = new_cursor

//             if verbose:
//                 print(f"merge {i+1}/{self.num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences \n\n\n")

//         connection.close()
//         self.merges = merges
//         self.vocab = vocab
        
//     def train(self, files: str,data_dir: str, verbose=False):
//         self.chunkify(files, data_dir)
//         self.chunk_train(verbose = verbose)

//         print(self.merges, self.vocab)

//     def decode(self, ids):
//         # given ids (list of integers), return Python string
//         part_bytes = []
//         for idx in ids:
//             if idx in self.vocab:
//                 part_bytes.append(self.vocab[idx])
//             elif idx in self.inverse_special_tokens:
//                 part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
//             else:
//                 raise ValueError(f"invalid token id: {idx}")
//         text_bytes = b"".join(part_bytes)
//         text = text_bytes.decode("utf-8", errors="replace")
//         return text

//     def _encode_chunk(self, text_bytes):
//         # return the token ids
//         # let's begin. first, convert all bytes to integers in range 0..255
//         ids = list(text_bytes)
//         while len(ids) >= 2:
//             # find the pair with the lowest merge index
//             stats = get_stats(ids)
//             pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
//             # subtle: if there are no more merges available, the key will
//             # result in an inf for every single pair, and the min will be
//             # just the first pair in the list, arbitrarily
//             # we can detect this terminating case by a membership check
//             if pair not in self.merges:
//                 break # nothing else can be merged anymore
//             # otherwise let's merge the best pair (lowest merge index)
//             idx = self.merges[pair]
//             ids = merge(ids, pair, idx)
//         return ids

//     def encode_ordinary(self, text):
//         """Encoding that ignores any special tokens."""
//         # split text into chunks of text by categories defined in regex pattern
//         text_chunks = re.findall(self.compiled_pattern, text)
//         # all chunks of text are encoded separately, then results are joined
//         ids = []
//         for chunk in text_chunks:
//             chunk_bytes = chunk.encode("utf-8") # raw bytes
//             chunk_ids = self._encode_chunk(chunk_bytes)
//             ids.extend(chunk_ids)
//         return ids

//     def encode(self, text, allowed_special="none_raise"):
//         """
//         Unlike encode_ordinary, this function handles special tokens.
//         allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
//         if none_raise, then an error is raised if any special token is encountered in text
//         this is the default tiktoken behavior right now as well
//         any other behavior is either annoying, or a major footgun
//         """
//         # decode the user desire w.r.t. handling of special tokens
//         special = None
//         if allowed_special == "all":
//             special = self.special_tokens
//         elif allowed_special == "none":
//             special = {}
//         elif allowed_special == "none_raise":
//             special = {}
//             assert all(token not in text for token in self.special_tokens)
//         elif isinstance(allowed_special, set):
//             special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
//         else:
//             raise ValueError(f"allowed_special={allowed_special} not understood")
//         if not special:
//             # shortcut: if no special tokens, just use the ordinary encoding
//             return self.encode_ordinary(text)
//         # otherwise, we have to be careful with potential special tokens in text
//         # we handle special tokens by splitting the text
//         # based on the occurrence of any exact match with any of the special tokens
//         # we can use re.split for this. note that surrounding the pattern with ()
//         # makes it into a capturing group, so the special tokens will be included
//         special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
//         special_chunks = re.split(special_pattern, text)
//         # now all the special characters are separated from the rest of the text
//         # all chunks of text are encoded separately, then results are joined
//         ids = []
//         for part in special_chunks:
//             if part in special:
//                 # this is a special token, encode it separately as a special case
//                 ids.append(special[part])
//             else:
//                 # this is an ordinary sequence, encode it normally
//                 ids.extend(self.encode_ordinary(part))
//         return ids