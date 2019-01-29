import NN
from NN import NN

if __name__ == '__main__':
    nn=NN()
    #nn.fit_nn(20)
    nn.load_model()
    nn.predict(path='/home/hikkav/environments/my_env/traindata/https:images.pexels.comphotos39501lamborghini-brno-racing-car-automobiles-39501.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500.jpg')
