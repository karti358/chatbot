from dvc.api import DVCFileSystem

fs = DVCFileSystem("./")
print(fs.find("/", detail = False, dvc_only = True))
# fs.get_file("/data/validation_data/RC_2017-01.txt", "data/validation_data/RC_2017-01.txt")
fs.get_file("/data/testing_data/RC_2017-03.txt", "data/validation_data/RC_2017-03.txt")