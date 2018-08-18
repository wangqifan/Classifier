import torch
from network import Net
from LoadData import testloader 
from torch.autograd import Variable


net=torch.load('net.pkl')

def test():
    correct = 0
    total = 0
    for data in testloader:
      images, labels = data
      images=images.cuda()
      labels=labels.cuda()
      outputs = net(Variable(images))
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



if __name__=='__main__':
    test()