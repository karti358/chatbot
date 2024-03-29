import sqlite3
import argparse
import pandas as pd
import sys
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "The pc_name, c_name specified are used for both training and testing data.")
    parser.add_argument("filename", type=str, help="Specify a filename to store the parent comments. The file will be created if not exists.")
    parser.add_argument("timeframes", type=str, nargs = "*", help = "Specify timeframes to retrieve data from database.")

    args = parser.parse_args()
    print(args)
    
    filename = args.filename
    # create_dirs(filename)
    
    timeframes = args.timeframes

    if len(timeframes) == 0:
        print("Provide valid timeframes.....!")
        print("Exiting execution...")
        sys.exit()
    

    for timeframe in timeframes:
        print(f"data/database/{timeframe}.db")
        connection = sqlite3.connect(f"data/database/{timeframe}.db")
        c = connection.cursor()
        
        """Defining some parameters"""
        limit = 5000
        last_unix = 0
        cur_length = limit
        counter = 0
        test_done = False
        validation_done = False

        while cur_length == limit:

            df = pd.read_sql(f"""SELECT * FROM reddit_comments 
                                 WHERE unix > {last_unix} and parent NOT NULL and score > 0 
                                 ORDER BY unix ASC LIMIT {limit};""",
                                 connection)
            last_unix = df.tail(1)['unix'].values[0]
            cur_length = len(df)

            if not test_done:
                with open(f"data/testing_data/{filename}.txt","a", encoding="utf8") as f:
                    for parent, comment in zip(df["parent"].values, df["comment"].values ):
                        f.write(f"{str(parent)}<||>{str(comment)}" +'\n')

                test_done = True

            elif not validation_done:
                with open(f"data/validation_data/{filename}.txt","a", encoding="utf8") as f:
                    for parent, comment in zip(df["parent"].values, df["comment"].values ):
                        f.write(f"{str(parent)}<||>{str(comment)}" +'\n')

                validation_done = True

            else:
                with open(f"data/training_data/{filename}.txt","a", encoding="utf8") as f:
                    for parent, comment in zip(df["parent"].values, df["comment"].values ):
                        f.write(f"{str(parent)}<||>{str(comment)}" +'\n')

            counter += 1
            if counter % 20 == 0:
                print(counter*limit,'rows completed so far')