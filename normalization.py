import statistics as stat
from torchvision import transforms
from PIL import Image
import time
from os import listdir
import random
'''from threading import Thread

def norm_colors(arg):
    color = tuple(c[arg] for c in imagelist)
    norm_color = (stat.mean(color)/255*1.99)-0.99
    normsd_color = (stat.stdev(color)/255*1.99)
    pass

if __name__ == "__main__":
    red = Thread(target=norm_colors, args=(0,))
    red.start()'''

files = listdir("F:/Dokumente/Arbeit/fehlerhafteSpots/AnaG_train/CO-NC-R3/")
cmplte_start = time.time()
for i in range(100):
    f = files[i]
    #f = random.choice(files)
    #files.remove(f)
    start = time.time()
    if not f.endswith('.db'):
        # load image for calculating the normalization for each image and add later as tensor in train_list
        
        img = Image.open(r"F:/Dokumente/Arbeit/fehlerhafteSpots/AnaG_train/CO-NC-R3/" + f) 
        imglist = img.getdata()

        '''#split-method --> seems the slowest
        r,g,b = img.split()
        red = r.getdata()
        green = g.getdata()
        blue = b.getdata()
        '''
        # list comprehension --> seems the fastest
        red = tuple(r[0] for r in imglist)
        green = tuple(g[1] for g in imglist)
        blue = tuple(b[2] for b in imglist)
        '''
        # normal for-loop
        red = []
        green = []
        blue = []

        for i in range(len(imagelist)):
            red.append(imagelist[i][0])
            green.append(imagelist[i][1])
            blue.append(imagelist[i][2])
        '''
        norm_red = (stat.mean(red)/255*1.99)-0.99
        normsd_red = (stat.stdev(red)/255*1.99)

        norm_green = (stat.mean(green)/255*1.99)-0.99
        normsd_green = (stat.stdev(green)/255*1.99)

        norm_blue = (stat.mean(blue)/255*1.99)-0.99
        normsd_blue = (stat.stdev(blue)/255*1.99)
            
        normalize = transforms.Normalize(mean=[norm_red, norm_green, norm_blue],
                                        std=[normsd_red, normsd_green, normsd_blue])
        
    end = time.time()
    print(end-start, normalize, f)
cmplte_end = time.time()
print(cmplte_end-cmplte_start)
