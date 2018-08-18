import torch
import torch.optim  as optim
import torch.nn as nn
from torch.autograd import Variable
from LoadData import trainloader
from network import Net

net=Net()
net=net.cuda()
criterion = nn.CrossEntropyLoss()


def trainmodel():
    length=0.001
    for epoch in range(5):
        length=length/2.0
        optimizer=optim.Adam(net.parameters(),lr=length,betas=(0.9,0.99))
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels=data
            inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())

            optimizer.zero_grad()

            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss +=  loss.data[0].item()
            if i % 2000 == 1999:    # 每2000个小批量打印一次
                  print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
                  running_loss = 0.0

    
    
if __name__=='__main__':
    trainmodel() 
    torch.save(net,"net.pkl")
