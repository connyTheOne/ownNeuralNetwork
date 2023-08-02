import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import statistics as stat

class Netz(nn.Module):
    def __init__(self, in_chn, end_output, fc_layer, learning_rate):
        super(Netz, self).__init__()
        
        # learning rate
        self.lr = learning_rate
        
        # 1st convolutional layer defining dimension
        self.conv1 = nn.Conv2d(in_chn, 6, kernel_size=5, stride=4)
        # 2nd convolutional layer defining dimension, uncomment if using. Be aware to adjust the self.forward()-Function
        #self.conv2 = nn.Conv2d(6, 12, kernel_size=2, stride=1)
        self.conv_dropout = nn.Dropout2d(p=0.2)
        
        # input nodes for fully connected Neural Network
        self.inodes = 6*12*12 # or connect with num_flat_features() somehow
        
        # decide how many layers should have your Neural Network: 3 or 4; anything else doesn't work
        self.layer = fc_layer
        if self.layer == 3:
            # hidden nodes are calculated via quadratic equation; optimised for 3-linear-Layer-Neural network
            self.hnodes = int((self.inodes/4*2**2)+(self.inodes/(-2)*2)+(self.inodes/4+end_output))
        
        elif self.layer == 4:
            # hidden nodes are calculated via quadratic equation; optimised for 4-linear-Layer-Neural network
            self.hnodes = int(((self.inodes-end_output)/9*3**2)+
                              ((self.inodes-end_output)/9*(-2)*3)+
                              (end_output-((self.inodes-end_output)/9+(self.inodes-end_output)/9*(-2))))
            self.hnodes1 = int((self.inodes-end_output)/9*2**2+
                               (2*(-2)*(self.inodes-end_output)/9)+
                               (end_output-((self.inodes-end_output)/9+(self.inodes-end_output)/9*(-2))))
        
        # define linear full connected layers and sizes
        self.fc1 = nn.Linear(self.inodes, self.hnodes, bias=True)
        if self.layer == 3:    
            self.fc2 = nn.Linear(self.hnodes, end_output, bias=True)
        elif self.layer == 4: 
            self.fc2 = nn.Linear(self.hnodes, self.hnodes1, bias=True)
            # for 4-Layer
            self.fc3 = nn.Linear(self.hnodes1, end_output, bias=True)
        
        # create optimiser, using simple stochastic gradient descent
        #self.optimizer = optim.SGD(self.parameters(), self.lr, momentum=0.8)
        self.optimizer = optim.Adam(self.parameters(), self.lr)
        pass
    
    def forward(self, inputs_list):
        
        # convert list to a 2-D FloatTensor then wrap in Variable 
        if inputs_list.type() == 'torch.Tensor':    # doesn't really work
            if torch.cuda.is_available():
                inputs_list = inputs_list.cuda()
                
            inputs_list = Variable(inputs_list)
            
        else:
            if torch.cuda.is_available():
                inputs = Variable(torch.Tensor(inputs_list).cuda())
            else:
                inputs = Variable(torch.Tensor(inputs_list))
        
        # CNN part
        # put your image as Tensor in 1st convolutional layer
        x = self.conv1(inputs)
        
        # apply ReLu activation function
        x = F.relu(x)
        # halve your 1st convolutional layer via MaxPool-Layer
        # If the size is a square you can only specify a single number...maybe
        x = F.max_pool2d(x, 2)
        
        # let forget some nodes of CNN (MaxPool-Layer)
        x = self.conv_dropout(x)
        # apply ReLu activation function
        global y
        y = F.relu(x)
        
        # for Troubleshoot, if you change the size of nodes in the NN. uncomment it for usage
        #print(x.size())
        # out [anz. bilder (batch_size) ,teilbilder output-channels , 4 height bilder(px), 4 wide bilder(px)]
        #print(self.num_flat_features(y))
        #exit()
        
        # convert the output from CNN to 1-dim-Tensor for input in fully connected NN
        x = x.view(-1, self.num_flat_features(y))
        # apply ReLu activation function in 1st fully connected layer
        x = F.relu(self.fc1(x))
        
        output = F.sigmoid(self.fc2(x))
        
        # Only for 4-Layer NN
        if self.layer == 4:
            # combine 1st hidden layer signals into 2nd hidden layer
            output = F.sigmoid(self.fc3(output))
        
        return output                         #in klammern mit schreiben ', dim=1'
        
        pass
    
    def train(self, inputs_list, target_list): 

        # calculate the output of the network
        output = self.forward(inputs_list)
        
        # create a Variable out of the target vector
        if torch.cuda.is_available():
            target_variable = Variable(torch.Tensor(target_list).cuda(), requires_grad=False)
        else:
            target_variable = Variable(torch.Tensor(target_list), requires_grad=False)
        
        # create error function and calculate error
        #criterion = nn.MSELoss(size_average=True)   # can be changed with other error function --> play around with
        #criterion = nn.CrossEntropyLoss()           # need LongTensor type for target
        criterion = F.binary_cross_entropy           # need requires_grad=False in taget_variable
        loss = criterion(output, target_variable, )
        print(loss)
        # print(loss.item()) #-> uncomment to observe the loss during training
        # the less the loss, the more the outputs and targets are identical
        
        # zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass
    
    def num_flat_features(self, inputs):
        # convert the output from CNN to 1-dim-Tensor for input in fully connected NN
        size = inputs.size()[1:]                     # all dimensions except the batch dimension
        num = 1
        for i in size:
            num *= i
        return num
        pass
    pass

RGBorGray = 3    # 1 for grayscale images 3 for RGB-images
end_nodes = 2    # how many end output nodes do you want?
fc_layer = 3     # decide how many layers should have your fully connected linear Neural Network: 3 or 4;
                 # anything else doesn't work
lr = 0.001

# create instance of neural network
model = Netz(RGBorGray, end_nodes, fc_layer, lr)
# take a look of your CNN-model
print(model)

# move neural network to the GPU, if available
if torch.cuda.is_available():
    model.cuda()
    print("cuda is OK")

'''# function to get normalization values for each image seperately for Convolutional Neural Network
def img_normalization(imglist):
    # list comprehension
    red = tuple(r[0] for r in imglist)
    green = tuple(g[1] for g in imglist)
    blue = tuple(b[2] for b in imglist)
    
    norm_red = (stat.mean(red)/255*1.99)-0.99
    normsd_red = (stat.stdev(red)/255*1.99)

    norm_green = (stat.mean(green)/255*1.99)-0.99
    normsd_green = (stat.stdev(green)/255*1.99)

    norm_blue = (stat.mean(blue)/255*1.99)-0.99
    normsd_blue = (stat.stdev(blue)/255*1.99)
            
    normalize = transforms.Normalize(mean=[norm_red, norm_green, norm_blue],
                                        std=[normsd_red, normsd_green, normsd_blue])
    return normalize
'''
# perfect normalize values for CO, NC & R3 Spots
normalize = transforms.Normalize(mean=[0.94057255, 0.71759216, 0.96673725], 
                                 std=[0.0464078, 0.43387451, 0.04229804])

transform = transforms.Compose([transforms.Resize(103),
                                transforms.CenterCrop(102),    
                                transforms.ToTensor(),
                                normalize])
print("Normalization and Transformation is OK")

# load images as batch from folder for train set
train_data_list = []
target_list = []
train_data = []
files = listdir("F:/Dokumente/Arbeit/fehlerhafteSpots/AnaG_train/CO-NC-R3/")
for i in range(len(files)):
    
    f = random.choice(files)
    files.remove(f)
    
    # load image for calculating the normalization for each image and add later as tensor in train_list
    img = Image.open("F:/Dokumente/Arbeit/fehlerhafteSpots/AnaG_train/CO-NC-R3/" + f)
    imglist = img.getdata()
    
    # rotate image for more train data by 90°
    for i in range(0, 358, 90):
        img = img.rotate(i)
        
        img_tensor = transform(img)
        train_data_list.append(img_tensor)
        # label the correct image/Tensor as target
        # use the filename to set the correct label
        label = f[-6:-4]
        
        if label == "_0":
            isOK = 1
            notOK = 0
        else:
            isOK = 0
            notOK = 1
        target = [isOK, notOK]
        target_list.append(target)
    
        if len(train_data_list) >= 512:
            #creates image batch for the batch size
            train_data.append((torch.stack(train_data_list),target_list))
            train_data_list = []
            target_list = []

print(len(train_data))
print(len(target_list))
print(len(train_data_list))

# load trained model with right weights
'''if os.path.isfile('SpotCNN.pt'):
    model = torch.load('SpotCNN.pt')'''

# train the model
# define epochs how many runs over training data
epochs = 32
#for-epoch-loop
for e in range(epochs):
    
    batch_id = 0
    # for-batch-loop
    for data, target in train_data:
        model.train(data, target)
        batch_id += 1
        
        print('Train Epoche: {} von {} [{}/{} ({:.0f}%)]'.format(e, epochs, batch_id * len(data), len(train_data),
                                                                  100. * batch_id / len(train_data)))
        pass
    pass

# save the trained model
#torch.save(model, 'smallSpotCNN.pt')

# test the model
files = listdir("F:/Dokumente/Arbeit/fehlerhafteSpots/test/")
for i in range(len(files)):
    #einzelnes bild; kein batch
    f = random.choice(files)
    files.remove(f)
    img = Image.open("F:/Dokumente/Arbeit/fehlerhafteSpots/test/" + f)
    imglist = img.getdata()

    img_eval_tensor = transform(img)
    img_eval_tensor.unsqueeze_(0)
    # use the filename to set the correct label
    label = f[-6:-4]
    if label == "_0":
        isOK = 1
        notOK = 0
    else:
        isOK = 0
        notOK = 1
    print(model.forward(img_eval_tensor))
    print(isOK, notOK)
    print(f)
    img.show()
    input()
    pass

print("Finished")
exit()
