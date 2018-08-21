'''
William Dreese

-train a CNN to give me propeties of a 32x32 pic w one object
- 

'''
from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
import layers
import datetime
import numpy as np

class BasicShapesModel:
    def __init__(self):
        self._epochs = 20
        self._learning_rate = 0.01
        self._batch_size = 20
        self._data = self.getData()
        self._model = layers.Model(lr=self._learning_rate,blr=self._learning_rate)
        
        self._model.add_layer(layers.conv(ems=1,
                                          nodes=20,
                                          kernel_size=3,
                                          padding=1,
                                          activation_function_="relu"))
        self._model.add_layer(layers.conv(ems=20,
                                          nodes=20,
                                          kernel_size=3,
                                          padding=1,
                                          activation_function_="relu"))
        self._model.add_layer(layers.max_pool(kernel_size=2))
        
        self._model.add_layer(layers.conv(ems=20,
                                          nodes=12,
                                          kernel_size=3,
                                          padding=1,
                                          activation_function_="relu"))        
        self._model.add_layer(layers.conv(ems=12,
                                          nodes=12,
                                          kernel_size=3,
                                          padding=1,
                                          activation_function_="relu"))        
        self._model.add_layer(layers.max_pool(kernel_size=2))
                
        self._model.add_layer(layers.conv(ems=12,
                                          nodes=6,
                                          kernel_size=3,
                                          padding=1,
                                          activation_function_="relu"))        
        self._model.add_layer(layers.conv(ems=6,
                                          nodes=6,
                                          kernel_size=3,
                                          padding=1,
                                          activation_function_="relu"))
        self._model.add_layer(layers.max_pool(kernel_size=2))
        
        self._model.add_layer(layers.dense(eis=96,
                                           nodes=48,
                                           act_func="tanh"))
        self._model.add_layer(layers.dense(eis=48,
                                           nodes=3,
                                           act_func="none"))

    def getData(self):
        daters = list()
        ret = list()
        datafile = open("shapeset1_1cs_2p_3o.5000.valid.amat")
        a = datafile.readline()
        for i in range(0,5000):
            daters.append(datafile.readline().split(" "))
        for data in daters:
            bckgrnd = float(data[0])
            image = list()
            l2_sum = 0.0
            for x in range(32):
                line = list()
                for y in range(32):
                    pt = float(data[32*x+y])
                    
                    if pt == bckgrnd: pt = 0.0
                    else: pt = 1.0
                    
                    l2_sum += pt**2
                    line.append(pt)
                image.append(line)
            l2_sum = np.sqrt(l2_sum)
            '''
            for x in range(32):
                for y in range(32):
                    image[x][y] /= l2_sum
                    '''
            label = data[-7]
            ret.append([[image],label])
        return ret

    def easy_view(self,data):
        for d in data:
            for m in d:
                for r in m:
                    for c in r:
                        print(np.around(c,1),end=" ")
                    print("")
                print("")
            print("")
        print("")

    def train(self):
        bat_si = 20
        curtime = datetime.datetime.now()
        for i in range(self._epochs):
            loss = 0.0
            for ex in range(int(5000/bat_si)):
                batch_images = np.array(self._data[ex*bat_si:ex*bat_si+bat_si]).T
                loss += self._model.train(batch_images[0],
                                          batch_images[1],
                                          bat_si)
                #model-specific benchmarking
                eprint("Loss after batch",str(ex),":",str(loss))
                loss = 0.0
                if ex % 10 == 0 and ex != 0:
                    curtime_new = datetime.datetime.now()
                    secs = (curtime_new.second-curtime.second)+(curtime_new.minute-curtime.minute)*60
                    curtime = curtime_new
                    eprint("10 batchs (200 passes) total time (secs): ", str(secs))                    
            print("Epoch ",str(i)," avg loss: ",str(float(loss/float(len(self._data)))))

if __name__ == "__main__":
    m = BasicShapesModel()
    m.train()
        
        
