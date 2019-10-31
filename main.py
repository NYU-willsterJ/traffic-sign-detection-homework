from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False


### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()
if use_gpu:
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

epoch_plot_list = []
training_plot_list = []
validation_plot_list = []

def train(epoch):
    model.train()

    correct = 0
    training_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if use_gpu:
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        max_index = output.max(dim=1)[1]
        correct += (max_index == target).sum()
        training_loss += loss

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

    training_accuracy = 100. * correct / len(train_loader.dataset)

    #print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
    #    100. * training_accuracy))

    print("Training set: Average loss: %d, Accuracy: %d/%d (%d)" %
          (training_loss / len(train_loader.dataset), correct, len(train_loader.dataset), training_accuracy))

    training_plot_list.append(training_accuracy)



def validation():
    model.eval()
    validation_loss = 0
    correct = 0

    for data, target in val_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)

            if use_gpu:
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)

    scheduler.step(np.around(validation_loss, 2))

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    validation_plot_list.append(100. * correct / len(val_loader.dataset))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        epoch_plot_list.append(epoch)  # for plotting

        train(epoch)
        validation()
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')

# plotting the graphs
import matplotlib.pyplot  as plt
plt.plot(epoch_plot_list, validation_plot_list, label='validation')
plt.plot(epoch_plot_list, training_plot_list, label='training')

plt.title("Deep Convnet model")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()