import NN
from NN import NN
from Settings import *
import os
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    nn = NN()
    # nn.fit_nn()
    # nn.show_stats()
    model = nn.load_model('/home/hikkav/environments/my_env/validCNNS/the _best_MODEL.h5')
    nn.predict(model=model)
