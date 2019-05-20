import wget
from settings import *
import zipfile
import os
import shutil
import time

def make_directory():
    if not os.path.exists(path_to_extracted):
        os.mkdir(path_to_extracted)


def load_data():
    print("The data for training wasn't found."
          " Loading it from {}".format(path_to_online_data))

    wget.download(path_to_online_data)

def make_crossval():
    for i in os.listdir('train'):
        for z,v in enumerate(os.listdir(os.path.join('train',i))):
            if z==500:
                break
            shutil.move(os.path.join('train',i,v),os.path.join('crossval',i))

def divide_data():
    os.mkdir(path_to_train)
    os.mkdir(path_to_test)
    os.mkdir(path_to_crossval)
    tmp = 'data/kvasir-dataset-v2-folds'
    for i in os.listdir(tmp):
        for z in os.listdir(os.path.join(tmp, i)):
            if z == 'train':
                for v in os.listdir(os.path.join(tmp, i, z)):
                    if not os.path.exists(os.path.join(path_to_train, v)):
                        os.mkdir(os.path.join(path_to_train, v))
                        os.mkdir(os.path.join(path_to_crossval, v))
                    for f,m in enumerate(os.listdir(os.path.join(tmp, i, z, v))):


                        shutil.copy(os.path.join(tmp, i, z, v, m), os.path.join(path_to_train, v))
                       
            else:
                for v in os.listdir(os.path.join(tmp, i, z)):
                    if not os.path.exists(os.path.join(path_to_test, v)):
                        os.mkdir(os.path.join(path_to_test, v))
                        
                    for m in os.listdir(os.path.join(tmp, i, z, v)):
                        shutil.copy(os.path.join(tmp, i, z, v, m), os.path.join(path_to_test, v))
    print('Done with test and train folders')
    
   

def extract__data():
    if not os.path.exists('train') or not os.path.exists('test') or not os.path.exists('crossval'):
        load_data()
        make_directory()
        zip_ref = zipfile.ZipFile(name_of_data, 'r')
        zip_ref.extractall(path_to_extracted)
        zip_ref.close()
        if os.path.exists('kvasir-dataset-v2-folds.zip'):
            os.remove('kvasir-dataset-v2-folds.zip')
        print("Loaded and extracted data,  dividing it into related groups")
        divide_data()
        make_crossval()
        shutil.rmtree('data')
    else:
        print('Found all the mandatory folders')

if __name__ == '__main__':
   extract__data()
