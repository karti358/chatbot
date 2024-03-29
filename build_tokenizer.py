
import pyspark
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

from nmt.tokenizer import RegexTokenizer, RegexTokenizerLarge

import sys
import os

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

data_dir = "hdfs://localhost:9000/data"
spark_master = "spark://172.20.33.77:7077"

tokenizer = RegexTokenizerLarge(vocab_size = 15000)

tokenizer.train([
    data_dir + "/training_data/RC_2017-01.txt",
    data_dir + "/training_data/RC_2017-02.txt",
    data_dir + "/training_data/RC_2017-03.txt"
])
