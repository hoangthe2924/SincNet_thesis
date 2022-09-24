import os
import glob
from pathlib import Path

root_path = str(Path(__file__).absolute().parent.parent.parent.parent.parent)
current_path = str(Path(__file__).absolute().parent)

# print(root_path)
# print(current_path)

def makeAllFolderNameToLower(input_path):
    for name in glob.glob(input_path + '/*'):
        temp = name.split('/')
        temp[len(temp) - 1] = temp[len(temp) - 1].lower()
        temp = '/'.join(temp)
        try:
            os.rename(name, temp)
            print('Done: ' + temp)
        except:
            print('Error: ' + name)

def makeAllFolderAndFileToLower(input_path):
    makeAllFolderNameToLower(input_path);

    for parent_folder_name in glob.glob(input_path + '/*'):
        makeAllFolderNameToLower(parent_folder_name)

    for parent_folder_name in glob.glob(input_path + '/*'):
        makeAllFolderNameToLower(parent_folder_name + '/*')


dataset_train_path = root_path + '/datasets/TIMIT/raw_TIMIT/train'
print(dataset_train_path)
makeAllFolderAndFileToLower(dataset_train_path)

dataset_test_path = root_path + '/datasets/TIMIT/raw_TIMIT/test'
print(dataset_test_path)
makeAllFolderAndFileToLower(dataset_test_path)