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

# load trained model with right weights
'''if os.path.isfile('SpotCNN.pt'):
    model = torch.load('SpotCNN.pt')'''

# test the model
files = listdir("directory/with/files/of/images/for/testing/")
for i in range(len(files)):
    #einzelnes bild; kein batch
    f = random.choice(files)
    files.remove(f)
    img = Image.open("directory/with/files/of/images/for/testing/" + f)
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
