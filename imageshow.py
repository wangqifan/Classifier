import matplotlib.pyplot as plt
import numpy as np 
import torchvision
import pylab
from LoadData import trainloader


def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


dataiter=iter(trainloader)
images,labes=dataiter.next()
imshow(torchvision.utils.make_grid(images))