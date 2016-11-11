import os

from classify import show_trained_data


def test_file_codec():
    rootdir = os.getcwd() + '/DataSet'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdir, file)

            try:
                f = open(path)
                sentence = f.read()
            except:
                print(path.split('/')[-1])


show_trained_data()

# test_file_codec()
