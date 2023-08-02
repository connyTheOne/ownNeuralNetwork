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

# perfect normalize values for CO, NC & R3 Spots
normalize = transforms.Normalize(mean=[0.96, 0.711, 0.96], 
                                 std=[0.0310, 0.473, 0.0346])

transform = transforms.Compose([transforms.Resize(103),
                                transforms.CenterCrop(102),    
                                transforms.ToTensor(),
                                normalize])

# load images as batch from folder for train set
train_data_list = []
target_list = []
train_data = []
files = listdir("C:/Users/Conrad/Documents/OwnDataSpots/train/NC_CO_R3/")
for i in range(len(files)):
    
    f = random.choice(files)
    files.remove(f)
    
    # load image as Tensor
    img = Image.open("C:/Users/Conrad/Documents/OwnDataSpots/train/NC_CO_R3/" + f)
    
    # rotate image for more train data by 90°
    for i in range(0, 358, 90):
        img = img.rotate(i)
        i += 45
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
    
        if len(train_data_list) >= 256:
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
torch.save(model, 'SpotCNN.pt')