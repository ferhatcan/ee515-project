import torch.optim as optim
import torch.nn as nn

from utils.dataloaders import get_dataloaders
from utils.experiment import Experiment
from utils.parse_ini_file import Options

from model import MNISTmodel, CNNModel


params = Options('config.ini')

if params.source_dataset_name == 'mnist':
    source_dataloaders = get_dataloaders('mnist', batch_size=params.batch_size, train=True)
    target_dataloaders = get_dataloaders('svhn', batch_size=params.batch_size, train=True)
    source_dataloaders['test'] = get_dataloaders('mnist', batch_size=params.batch_size, train=False)['test']
    target_dataloaders['test'] = get_dataloaders('svhn', batch_size=params.batch_size, train=False)['test']

    if not params.only_source:
        print('Unsupervised Domain Adaptation From {} To {}'.format('MNIST', 'SVHN'))
    else:
        print('Without domain adaptation, only trained on {}'.format('MNIST'))
else:
    source_dataloaders = get_dataloaders('svhn', batch_size=params.batch_size, train=True)
    target_dataloaders = get_dataloaders('mnist', batch_size=params.batch_size, train=True)
    source_dataloaders['test'] = get_dataloaders('svhn', batch_size=params.batch_size, train=False)['test']
    target_dataloaders['test'] = get_dataloaders('mnist', batch_size=params.batch_size, train=False)['test']

    if not params.only_source:
        print('Unsupervised Domain Adaptation From {} To {}'.format('SVHN', 'MNIST'))
    else:
        print('Without domain adaptation, only trained on {}'.format('SVHN'))


model = MNISTmodel()
# model = CNNModel()
for p in model.parameters():
    p.requires_grad = True

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
# criterion = nn.NLLLoss()
experiment = Experiment(model, optimizer, criterion, params)

experiment.load('./trained_params/MNIST-cnst-LR_100_model_last.pth')
# experiment.train(source_dataloader=source_dataloaders, target_dataloader=target_dataloaders)

print('Source Domain Results: ')
experiment.test(source_dataloaders['test'], domain_flag='source')
print('Target Domain results: ')
experiment.test(target_dataloaders['test'], domain_flag='target')
