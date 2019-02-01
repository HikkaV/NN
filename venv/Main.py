import NN
from NN import NN
from Settings import *
import os
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    nn = NN()
    nn.fit_nn()
    nn.show_stats()
    model = nn.load_model()
    nn.predict(path='/home/hikkav/environments/my_env/cs.jpg', model=model)
