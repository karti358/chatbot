import argparse
import sqlite3
import json
from datetime import datetime
import time
from pathlib import Path

def create_table():
    sql = """CREATE TABLE IF NOT EXISTS reddit_comments
            (
                parent_id TEXT PRIMARY KEY, 
                comment_id TEXT UNIQUE, 
                parent TEXT, 
                comment TEXT, 
                subreddit TEXT,
                unix INT,
                score INT)"""
    return c.execute(sql)

def find_parent(pid):
    try:
        sql = f"""SELECT comment FROM reddit_comments 
                 WHERE comment_id = '{pid}' LIMIT 1"""
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        return False
    except Exception as e:
        #print(str(e))
        return False

def format_data(data):
    newlinechar = " <|newlinechar|> "

    data = data.replace('\n', newlinechar).replace('\r', newlinechar).replace('"',"'")
    return data

def find_existing_score(pid):
    try:
        sql = f"""SELECT score FROM reddit_comments
                 WHERE parent_id = '{pid}' LIMIT 1"""
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        #print(str(e))
        return False

def acceptable(data):
    if len(data.split(' ')) > 1000 or len(data) < 1:
        return False
    elif len(data) > 32000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    return True

def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []

def sql_insert_replace_comment(parent_id, comment_id,parent_data,comment, subreddit, unix, score):
    try:
        sql = f"""UPDATE reddit_comments 
                 SET parent_id = "{parent_id}", comment_id = "{comment_id}", parent = "{parent_data}", comment = "{comment}", subreddit = "{subreddit}", unix = {int(unix)},  score = {int(score)} 
                 WHERE parent_id = "{parent_id}";"""
        transaction_bldr(sql)
    except Exception as e:
        print('Problem in sql_insert_replace_comment',str(e))

def sql_insert_has_parent(parent_id, comment_id,parent_data,comment, subreddit, unix,score):
    try:
        sql = f"""INSERT INTO reddit_comments (parent_id, comment_id, parent, comment,subreddit, unix, score) 
                 VALUES ("{parent_id}","{comment_id}","{parent_data}","{comment}","{subreddit}", {int(unix)}, {int(score)});"""
        transaction_bldr(sql)
    except Exception as e:
        print('Problem in sql_insert_has_parent',str(e))

def sql_insert_no_parent(parent_id, comment_id, comment, subreddit, unix, score):
    try:
        sql = f"""INSERT INTO reddit_comments (parent_id, comment_id,parent, comment,subreddit, unix,  score) 
                 VALUES ("{parent_id}","{comment_id}",NULL, "{comment}", "{subreddit}", {int(unix)} , {int(score)});"""
        transaction_bldr(sql)
    except Exception as e:
        print('Problem in sql_insert_not_parent',str(e))


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str, help = "Give the posix path of the file to load to database.")
    args = parser.parse_args()
    print(args)

    filepath = Path(args.filepath)

    """Defining some global variables"""
    timeframe = filepath.as_posix().split("/")[-1]
    sql_transaction = []
    start_row = 0
    cleanup = 1000000

    main_dir = Path.cwd()
    connection = sqlite3.connect(main_dir.joinpath(f"data/database/{timeframe}.db").as_posix())
    c = connection.cursor()
    """Done"""

    c = create_table()
    row_counter = 0
    paired_rows = 0

    #with open('J:/chatdata/reddit_data/{}/RC_{}'.format(timeframe.split('-')[0],timeframe), buffering=1000) as f:
    with open(filepath.as_posix(), buffering=1000) as f:
        for row in f:
            # print(row)
            #time.sleep(555)
            row_counter += 1

            if row_counter > start_row:
                try:
                    row = json.loads(row)

                    """get data"""
                    parent_id = row['parent_id'].split('_')[1]
                    comment_id = row['id']

                    parent_data = find_parent(parent_id)
                    comment = format_data(row['body'])
                    score = row['score']
                    
                    subreddit = row['subreddit']
                    created_utc = row['created_utc']
                    """Done"""
                    
                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            if acceptable(comment):
                                sql_insert_replace_comment(parent_id, comment_id,parent_data,comment, subreddit, created_utc, score)
                                
                    else:
                        if acceptable(comment):
                            if parent_data:
                                if score >= 2:
                                    sql_insert_has_parent(parent_id, comment_id,parent_data,comment, subreddit, created_utc, score)
                                    paired_rows += 1
                            else:
                                sql_insert_no_parent(parent_id, comment_id,comment, subreddit, created_utc, score)
                except Exception as e:
                    print(str(e))
                            
            if row_counter % 100000 == 0:
                print(f"Total Rows Read: {row_counter}, Paired Rows: {paired_rows}, Time: {datetime.now()}")

            if row_counter > start_row and row_counter % cleanup == 0:
                print("Cleanin up!")
                sql = f"DELETE FROM reddit_comments WHERE parent IS NULL OR parent = 'False'"
                c.execute(sql)
                connection.commit()
                c.execute("VACUUM")
                connection.commit()
                # break