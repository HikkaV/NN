path_to_model = '/home/hikkav/environments/my_env/CNNmodel.h5'
path_to_train = '/home/hikkav/environments/my_env/mytraindata'
path_to_test = '/home/hikkav/environments/my_env/mytestdata'
path_to_history = '/home/hikkav/environments/my_env/history.json'
learning_rate = 0.001
n_classes = 2
n_channels=3
input_neurons = 2500
n_epochs = 1
batch = 64
img_size = 50
last_dropout = 0.5
kernel_size = (3, 3)
space = {'space':[  (10**-5, 10.0, 'log-uniform'),#eta
                     (2, 30) #epochs
                     ]}
n_calls=10
random_state=1