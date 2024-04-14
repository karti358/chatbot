package main

import (
	"bufio"
	"database/sql"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

// var GPT2_SPLIT_PATTERN string = "'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
// var GPT4_SPLIT_PATTERN string = "'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

// var GPT2_SPLIT_PATTERN string = `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
// var GPT4_SPLIT_PATTERN string = `(?i:'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)`
var GPT4_SPLIT_PATTERN string = `(?i:'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+\r?\n*|\s*\r?\n|\s+([^\S\r\n])|[\s\r\n]+)
`

type Tokenizer struct {
	vocab_size             int32
	num_merges             int32
	pattern                string
	compiled_pattern       *regexp.Regexp
	special_tokens         map[string]int32
	inverse_special_tokens map[int32]string
	buf_size               int32
	sql_transaction        []string
	counter                int32
	database_path          string
	merges                 map[Tuple[int32, int32]]int32
	vocab                  map[int32]int32
}

func initialize(vocab_size int32, special_tokens map[string]int32, pattern string) Tokenizer {
	vocab_size_ := vocab_size
	// assert.Assert()
	num_merges_ := vocab_size - 256

	pattern_ := ""
	if len(pattern) == 0 {
		pattern_ = GPT4_SPLIT_PATTERN
	} else {
		pattern_ = pattern
	}

	compiled_pattern_ := regexp.MustCompile(pattern_)
	// fmt.Println(compiled_pattern_)

	var special_tokens_ map[string]int32
	if special_tokens == nil {
		special_tokens_ = map[string]int32{"<|newlinechar|>": vocab_size, "<|start|>": vocab_size + 1, "<|end|>": vocab_size + 2, "<|padding|>": vocab_size + 3}
	} else {
		special_tokens_ = special_tokens
	}

	var inverse_special_tokens_ map[int32]string
	for k, v := range special_tokens_ {
		inverse_special_tokens_[v] = k
	}

	var buf_size_ int32 = 100

	// var database_path_ string

	var merges_ map[Tuple[int32, int32]]int32

	var vocab_ map[int32]int32

	return Tokenizer{
		vocab_size:             vocab_size_,
		num_merges:             num_merges_,
		pattern:                pattern_,
		compiled_pattern:       compiled_pattern_,
		special_tokens:         special_tokens_,
		inverse_special_tokens: inverse_special_tokens_,
		buf_size:               buf_size_,
		sql_transaction:        make([]string, 0, buf_size_),
		counter:                0,
		merges:                 merges_,
		vocab:                  vocab_,
	}
}

func (self Tokenizer) chunkify(files []string, data_dir string) *sql.DB {
	self.sql_transaction = make([]string, 0, self.buf_size)
	self.counter = 0

	create_chunk_ids_table := func(conn *sql.DB) {
		sql := "CREATE TABLE IF NOT EXISTS chunk_ids_0 (id INTEGER PRIMARY KEY,text TEXT);"
		statement, err := conn.Prepare(sql)
		if err != nil {
			fmt.Println(err)
			panic("Problem creating table")
		}
		statement.Exec()
	}

	transaction_builder := func(sql string, conn *sql.DB) {
		self.sql_transaction = append(self.sql_transaction, sql)
		if len(self.sql_transaction) == int(self.buf_size) {

			statement, err := conn.Prepare("BEGIN TRANSACTION;")
			if err != nil {
				fmt.Println(err)
				panic("Cannot begin transaction panicked ...")
			}
			statement.Exec()

			for _, s := range self.sql_transaction {
				statement, err = conn.Prepare(s)
				if err != nil {
					fmt.Println(err)
					panic("Inside Chunkify loop of builder panicked...")
				}
				statement.Exec()
			}
			fmt.Printf("Inserted %v into chunk_ids_0...", self.counter)
			self.sql_transaction = make([]string, 0, self.buf_size)
		}
	}

	flush_pending := func(conn *sql.DB) {
		statement, err := conn.Prepare("BEGIN TRANSACTION;")
		if err == nil {
			statement.Exec()
		}

		for _, s := range self.sql_transaction {
			statement, err = conn.Prepare(s)
			if err == nil {
				statement.Exec()
			}
		}
		self.sql_transaction = make([]string, 0, self.buf_size)
	}

	insert_into_chunk_ids := func(chunks []string, conn *sql.DB) {
		var builder strings.Builder
		builder.WriteString("INSERT INTO chunk_ids_0 (text) VALUES (")
		if len(chunks) == 0 {
			fmt.Println("Fuck empty")
			builder.WriteString("'');")
			self.counter += 1
			transaction_builder(builder.String(), conn)
			return
		}

		for i, temp_str := range chunks {
			var builder2 strings.Builder
			for j, rn := range []rune(temp_str) {
				builder2.WriteString(strconv.Itoa(int(rn)))

				if j < len(temp_str)-1 {
					builder2.WriteString(",")
				}

			}
			builder.WriteString(fmt.Sprintf("'%v'", builder2.String()))

			if i < len(chunks)-1 {
				builder.WriteString("),(")
			}
		}
		builder.WriteString(");")

		self.counter += 1
		transaction_builder(builder.String(), conn)
	}

	conn, _ := sql.Open("sqlite3", "file:"+data_dir+"/chunks.db")
	fmt.Println(conn)
	// defer conn.Close()

	create_chunk_ids_table(conn)
	for _, file := range files {
		f_obj, err := os.Open(file)
		defer f_obj.Close()

		if err != nil {
			panic(err)
		}

		scanner := bufio.NewScanner(f_obj)
		for scanner.Scan() {
			line := scanner.Text()
			// chunks := self.compiled_pattern.FindAllString(line, -1)
			// fmt.Println(chunks)
			chunks := strings.Split(line, "<||>")
			// fmt.Println(chunks)
			// fmt.Println(chunks)
			insert_into_chunk_ids(chunks, conn)
		}

		if len(self.sql_transaction) > 0 {
			flush_pending(conn)
		}

		fmt.Println("Done Processing ... -> ", file)
		fmt.Println("\n\n\n")
	}

	// sample_rows, ok := conn.Query("SELECT * FROM chunk_ids_0;")
	// if ok != nil {
	// 	panic("Cannot query from table...panicking")
	// }
	// defer sample_rows.Close()

	// temp_ctr := 0
	// for sample_rows.Next() {
	// 	var (
	// 		id   int32
	// 		text string
	// 	)
	// 	sample_rows.Scan(&id, &text)

	// 	if len(text) > 0 {
	// 		fmt.Println(id, text)
	// 	} else {
	// 		temp_ctr += 1
	// 	}
	// }
	// fmt.Println(temp_ctr)

	return conn
}

func (self Tokenizer) chunk_train(conn *sql.DB, verbose bool) {
	create_chunk_ids_table := func(i int32, conn *sql.DB) {
		sql := fmt.Sprintf("CREATE TABLE IF NOT EXISTS chunk_ids_%v (id INTEGER PRIMARY KEY,text TEXT);", i)
		statement, err := conn.Prepare(sql)
		if err != nil {
			fmt.Println("Problem creating table")
			panic(err)
		}
		statement.Exec()
	}

	transaction_builder := func(i int32, sql string, conn *sql.DB) {
		self.sql_transaction = append(self.sql_transaction, sql)
		if len(self.sql_transaction) == int(self.buf_size) {

			statement, err := conn.Prepare("BEGIN TRANSACTION;")
			if err != nil {
				fmt.Println(err)
				panic("Cannot begin transaction panicked ...")
			}
			statement.Exec()

			for _, s := range self.sql_transaction {
				statement, err = conn.Prepare(s)
				if err != nil {
					fmt.Println("\n\n", err)
					fmt.Println(s)
					panic("Inside loop of builder panicked...")
				}
				statement.Exec()
			}

			if self.counter%100000 == 0 {
				fmt.Printf(fmt.Sprintf("Inserted %v into chunk_ids_%v...", self.counter, i))
			}
			self.sql_transaction = make([]string, 0, self.buf_size)
		}
	}

	flush_pending := func(i int32, conn *sql.DB) {
		statement, err := conn.Prepare("BEGIN TRANSACTION;")
		if err == nil {
			statement.Exec()
		}

		for _, s := range self.sql_transaction {
			statement, err = conn.Prepare(s)
			if err == nil {
				statement.Exec()
			}
		}
		fmt.Printf("Flushed pending...")
		self.sql_transaction = make([]string, 0, self.buf_size)
	}

	insert_into_chunk_ids := func(i int32, chunk_ids *[]int32, conn *sql.DB) {
		var builder strings.Builder
		builder.WriteString(fmt.Sprintf("INSERT INTO chunk_ids_%v (text) VALUES (", i))
		if len((*chunk_ids)) == 0 {
			fmt.Println("Fuck empty")
			builder.WriteString("'');")
			self.counter += 1
			transaction_builder(i, builder.String(), conn)
			return
		}
		builder.WriteString("'")
		for i, temp_id := range *chunk_ids {
			builder.WriteString(strconv.Itoa(int(temp_id)))

			if i < len(*chunk_ids)-1 {
				builder.WriteString(",")
			}
		}
		builder.WriteString("');")

		self.counter += 1
		transaction_builder(i, builder.String(), conn)
	}

	process_stats := func(i int32, conn *sql.DB) map[Tuple[int32, int32]]int32 {
		stats := map[Tuple[int32, int32]]int32{}

		rows, ok := conn.Query(fmt.Sprintf("SELECT * FROM chunk_ids_%v;", i-1))
		if ok != nil {
			fmt.Println(ok)
			panic(fmt.Sprintf("Cannot query from table chunk_ids_%v...panicking", i-1))
		}
		defer rows.Close()

		var (
			id        int32
			text      string
			temp_id   int
			chunk_ids []int32
		)

		row_counter := 0
		for rows.Next() {
			rows.Scan(&id, &text)
			for _, st := range strings.Split(text, ",") {
				temp_id, _ = strconv.Atoi(st)
				chunk_ids = append(chunk_ids, int32(temp_id))
			}
			get_stats(&chunk_ids, &stats)
			row_counter += 1
			// fmt.Printf("Processed %v rows", row_counter)
			clear(chunk_ids)
		}
		fmt.Printf("Processed stats of chunk_ids_%v... ", i-1)
		return stats
	}

	update_ids := func(i int32, pair Tuple[int32, int32], idx int32, conn *sql.DB) {
		create_chunk_ids_table(i, conn)

		self.counter = 0
		rows, ok := conn.Query(fmt.Sprintf("SELECT * FROM chunk_ids_%v;", i-1))
		if ok != nil {
			panic(fmt.Sprintf("Cannot query from table chunk_ids_%v...panicking", i-1))
		}

		var (
			id        int32
			text      string
			temp_ids  []int32
			chunk_ids []int32
			temp_id   int
		)

		row_counter := 0
		for rows.Next() {
			rows.Scan(&id, &text)
			for _, st := range strings.Split(text, ",") {
				temp_id, _ = strconv.Atoi(st)
				chunk_ids = append(chunk_ids, int32(temp_id))
			}

			temp_ids = merge(&chunk_ids, pair, idx)
			insert_into_chunk_ids(i, &temp_ids, conn)
			row_counter += 1
		}
		fmt.Printf("Done updating tokens into chunk_ids_%v", i)
		if len(self.sql_transaction) > 0 {
			flush_pending(i, conn)
		}
	}

	// conn, err := sql.Open("sqlite3", "file:"+self.database_path)
	// if err != nil {
	// 	panic("Hmm cannot open database ... panicked")
	// }
	// defer conn.Close()

	var merges map[Tuple[int32, int32]]int32
	var vocab map[int32]int32

	for i := int32(1); i <= self.num_merges; i++ {
		fmt.Printf("Starting merge %v", i)

		stats := process_stats(i, conn)

		var pair Tuple[int32, int32]
		var maxi_val int32 = -1
		for k, v := range stats {
			if maxi_val < v {
				maxi_val = v
				pair = k
			}
		}
		if maxi_val == -1 {
			panic("Could not get max tuple...")
		}

		idx := 256 + i

		update_ids(i, pair, idx, conn)

		merges[pair] = idx
		vocab[idx] = vocab[pair.id1] + vocab[pair.id2]

		statement, ok := conn.Prepare(fmt.Sprintf("DROP TABLE chunk_ids_%v", i-1))
		if ok != nil {
			panic(fmt.Sprintf("Not able to drop chunk_ids_%v", i-1))
		}
		statement.Exec()

		if verbose {
			fmt.Printf("Merge %v / %v : %v -> %v (%v) had %v occurences \n\n\n", i, self.num_merges, idx, vocab[idx], stats[pair])
		}

		self.merges = merges
		self.vocab = vocab
	}

	conn.Close()
}

func (self Tokenizer) train(files []string, data_dir string, verbose bool) {
	conn := self.chunkify(files, data_dir)
	self.chunk_train(conn, verbose)

	fmt.Println(self.merges, self.vocab)

}

func main() {
	var tok Tokenizer = initialize(276, map[string]int32{}, "")
	tok.train([]string{"/home/thrasher/Class/chatbot/data/validation_data/RC_2017-03.txt"}, "/home/thrasher/Class/chatbot/data", true)

}
