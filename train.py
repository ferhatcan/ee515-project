import torch.optim as optim
import torch.nn as nn

from utils.dataloaders import get_dataloaders
from utils.experiment import Experiment
from utils.parse_ini_file import Options

from model import MNISTmodel


params = Options('config.ini')

mnist_dataloaders = get_dataloaders('mnist', batch_size=params.batch_size, train=True)
svhn_dataloaders = get_dataloaders('svhn', batch_size=params.batch_size, train=True)

mnist_test_dataloaders = get_dataloaders('mnist', batch_size=params.batch_size, train=False)
svhn_test_dataloaders = get_dataloaders('svhn', batch_size=params.batch_size, train=False)

model = MNISTmodel()

if params.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                          momentum=params.momentum, weight_decay=params.weight_decay)
elif params.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
else:
    print('Given optimizer ({}) not found. SGD will be used instead.'.format(params.optimizer))
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                          momentum=params.momentum, weight_decay=params.weight_decay)


criterion = nn.CrossEntropyLoss()
experiment = Experiment(model, optimizer, criterion, params)

experiment.train(source_dataloader=mnist_dataloaders['train'], target_dataloader=svhn_dataloaders['train'])
# experiment.load('./trained_params/MNIST-SVHN_10.pth')

print('Source Domain Results: ')
experiment.test(mnist_test_dataloaders['test'], domain_flag='source')
print('Target Domain results: ')
experiment.test(svhn_test_dataloaders['test'], domain_flag='target')
