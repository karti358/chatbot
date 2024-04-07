from nmt.tokenizer import RegexTokenizerLarge

tokenizer = RegexTokenizerLarge(vocab_size = 15000)
tokenizer.train(files = ["data/training_data/tok_train_data.txt"], data_dir = "data/", verbose = True)

tokenizer.save("models/tokenizer/bpe_tokenizer_1")

# text = ""
# with open("data/training_data/RC_2017-01.txt", "r", encoding = "utf-8") as fin, open("data/training_data/tok_train_data.txt", "a", encoding = "utf-8") as fout:
#     for i, line in enumerate(fin):
#         temp = " ".join(line.split("<||>"))

#         fout.write(temp)
#         if i == 5000:
#             break




