import NN
from NN import NN
if __name__ == '__main__':
    nn = NN()
    model = nn.load_model()
    nn.predict(model=model, path = '/home/hikkav/environments/my_env/cat.jpg')