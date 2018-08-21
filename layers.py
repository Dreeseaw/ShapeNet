'''
William Dreese 2018

Model class
Layer classes

'''
import numpy as np
import math
import random

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(1.0-x))

def relu(net):
    if net>0.0:
        return net
    return 0.0

def leaky_relu(net):
    if net>0.0:
        return net
    return 0.001

def relu_deriv(net):
    if net>0.0:
        return 1.0
    return 0.0

def leaky_relu_deriv(net):
    if net>0.0:
        return 1.0
    return 0.001

class Model:

    def __init__(self,lr=0.01,blr=0.01,bs=20):
        self._layer_flow = list()
        self._lr = lr
        self._blr = blr
        self._bs = bs

        #adam optimizer inits
        self._ms = list()
        self._vs = list()
        self._Bms = list()
        self._Bvs = list()
        self._beta1 = 0.9
        self._beta2 = 0.99
        self._eps = 1e-2
        self._first_optimize = True
        self._iters = 1
        
    def add_layer(self,lay):
        self._layer_flow.append(lay)

    def forward(self,data):
        inp = data
        acts = list()
        acts.append(inp)
        for lay in self._layer_flow:
            inp = lay.forward(inp)
            acts.append(inp)
        return acts

    def backward(self,grads,acts):
        grad_map = list()
        for n in range(len(self._layer_flow)):
            grads,update = self._layer_flow[-1*(n+1)].backward(grads,acts[-1*(n+2)])
            if self._layer_flow[-1*(n+1)]._trainable:
                grad_map.append(update)
        return np.array(grad_map)

    def loss_function(self,preds,real):
        ret,loss = self.cross_entropy_loss(preds,real)
        return ret,loss

    def softmax(self,acts):
        exps = np.exp(acts - np.max(acts))
        return exps / np.sum(exps)

    def cross_entropy_loss(self,preds,real):
        target = np.zeros(np.shape(preds),dtype=float)
        target[int(real)] = 1.0
        preds = self.softmax(preds)
        grads = preds - target
        loss = np.sum(target*np.log(preds)+(1.0-target)*np.log(1.0-preds))
        loss /= -1*len(target)
        return grads,loss

    def save_weights(self):
        return 0
    def load_weights(self):
        return 0

    def optimize(self,grads):
        grads = list(reversed(grads))
        
        if self._first_optimize:
            for g in grads:
                self._ms.append(np.zeros(np.shape(g[0])))
                self._vs.append(np.zeros(np.shape(g[0])))
                self._Bms.append(np.zeros(np.shape(g[1])))
                self._Bvs.append(np.zeros(np.shape(g[1])))
            self._first_optimize = False

        grad_count = 0
        for lay in range(len(self._layer_flow)):
            if self._layer_flow[lay]._trainable:
                #update 1st/2nd moment grids
                self._ms[grad_count] = self._ms[grad_count]*self._beta1 + grads[grad_count][0]*(1.0-self._beta1)
                self._vs[grad_count] = self._vs[grad_count]*self._beta2 + (grads[grad_count][0]**2)*(1.0-self._beta2)
                self._Bms[grad_count] = self._Bms[grad_count]*self._beta1 + grads[grad_count][1]*(1.0-self._beta1)
                self._Bvs[grad_count] = self._Bvs[grad_count]*self._beta2 + (grads[grad_count][1]**2)*(1.0-self._beta2)
                #correct for beta bias
                ms_b = self._ms[grad_count] * (1.0 / (1.0 - self._beta1**self._iters))
                vs_b = self._vs[grad_count] * (1.0 / (1.0 - self._beta2**self._iters))
                vs_b = np.sqrt(vs_b) + self._eps
                Bms_b = self._Bms[grad_count] * (1.0 / (1.0 - self._beta1**self._iters))
                Bvs_b = self._Bvs[grad_count] * (1.0 / (1.0 - self._beta2**self._iters))
                Bvs_b = np.sqrt(Bvs_b) + self._eps
                #update weights
                update = self._lr*(ms_b/vs_b)
                Bupdate = self._blr*(Bms_b/Bvs_b)
                self._layer_flow[lay]._weights -= np.array(update)
                self._layer_flow[lay]._weights = np.around(self._layer_flow[lay]._weights,12)
                #update biases (nothing special)
                self._layer_flow[lay]._bias -= Bupdate
                self._layer_flow[lay]._bias = np.around(self._layer_flow[lay]._bias,12)
                grad_count += 1
        self._iters += 1
                
    def grad_pass(self,image,real):
        acts = self.forward(image)
        grad,loss = self.loss_function(acts[-1],real)
        grads = self.backward(grad,acts)
        return grads,loss

    def train(self,image,real,batch_size):
        grads,loss = self.grad_pass(image[0],real[0])
        for b in range(1,batch_size):
            grads1,loss1 = self.grad_pass(image[b],real[b])
            grads += grads1
            loss += loss1
        grads /= float(batch_size)
        loss /= float(batch_size)
        self.optimize(grads)
        return loss

#need to -convert to numpy lists/splicing
#        -make 3d, since now we know how convos fucking work
class conv:

    def __init__(self,ems=1,nodes=10, kernel_size=5,init_scalar=1,activation_function_="relu",stride=1,padding=0,learning_rate=0.2,gc=False,tt=False):
        self._size = nodes
        self._kernel_size = kernel_size
        self._stride = stride
        self._expected_map_size = ems
        self._padding = padding
        self._act_func = activation_function_
        self._learning_rate = learning_rate
        self._in_shape = None
        self._out_shape = None
        self._grad_clipping = gc
        self._trainable = True
        self._relu_weights = tt
        self._relu_cache = None
        self._im2col_cache = None
        
        std = 1.0 / math.sqrt(self._expected_map_size*(self._kernel_size**2))
        self._weights = np.random.normal(scale=std,size=(self._size,self._expected_map_size,self._kernel_size,self._kernel_size))
        if self._relu_weights:
            for (m,z,x,y),val in np.ndenumerate(self._weights):
                self._weights[m,z,x,y] = relu(val)
        
        self._total_update = np.zeros(np.shape(self._weights))
        self._bias = np.zeros((self._size,))

    def activation_function(self,net):
        if self._act_func == "relu": return relu(net)
        elif self._act_func == "leaky_relu": return leaky_relu(net)
        elif self._act_func == "tanh": return np.tanh(net)
        elif self._act_func == "sigmoid": return sigmoid(net)
        else: return 0.0

    def activation_function_deriv(self,delt):
        if self._act_func == "relu":
            #delt*(relu cache >= 0.0)
            for (x,y,z),val in np.ndenumerate(delt):
                delt[x,y,z] *= (self._relu_cache[x,y,z] > 0.0)
            return delt
        elif self._act_func == "leaky_relu":
            for (x,y,z),val in np.ndenumerate(delt):
                delt[x,y,z] = delt[x,y,z]*(self._relu_cache[x,y,z] > 0.0)+(0.01)
            return delt
        elif self._act_func == "tanh":
            return (1.0-(np.tanh(delt)**2))
        elif self._act_func == "sigmoid":
            return sigmoid(delt)*(1-sigmoid(delt))

    #take 3d numpy array, return numpy feature maps
    def forward(self,inputs):
        inputs = np.array(inputs)
        self._in_shape = np.shape(inputs)

        #check/apply padding
        if self._padding != 0:
            new_inp = list()
            for maps in inputs:
                new_inp.append(np.pad(maps,self._padding,'constant'))
            inputs = np.array(new_inp)
    
        in_size = len(inputs[0])
        
        '''
        #basic conv forward pass
        acts = list()
        for n in range(self._size):
            feature_grid = list()
            for i in range(in_size-self._kernel_size+1):
                row = list()
                for ii in range(in_size-self._kernel_size+1):
                    inp = inputs[:,ii:ii+self._kernel_size,i:i+self._kernel_size]
                    net = np.sum(inp*self._weights[n]) + self._bias[n]
                    row.append(self.activation_function(net))
                feature_grid.append(row)
            acts.append(feature_grid)
        '''
        
        #im2col forward pass
        #flatten all weights and partial inputs

        if self._im2col_cache == None:
            self._im2col_cache = list()
            for i in range(in_size-self._kernel_size+1):
                for ii in range(in_size-self._kernel_size+1):
                    inpcoord = list()
                    for x in range(self._kernel_size):
                        for y in range(self._kernel_size):
                            linecoord = [(n,i+x,ii+y) for n in range(len(inputs))]
                            inpcoord = inpcoord+linecoord
                    self._im2col_cache.append(inpcoord)

        flattened_weights = np.reshape(self._weights,(self._size,np.ma.size(self._weights[0])))

        flattened_input = np.zeros((len(self._im2col_cache),len(self._im2col_cache[1])))
        for x in range(len(flattened_input)):
            for y in range(len(flattened_input[0])):                
                flattened_input[x][y] = inputs[self._im2col_cache[x][y]]

        #matmul
        acts = flattened_weights @ flattened_input.T
        acts += np.tile(self._bias,((in_size-self._kernel_size+1)**2,1)).T
        for (x,y),val in np.ndenumerate(acts):
            acts[x,y] = self.activation_function(val)
        acts = np.reshape(acts,(self._size,in_size-self._kernel_size+1,in_size-self._kernel_size+1))
        
        self._out_shape=np.shape(acts)
        self._relu_cache = acts
        return acts

    def backward(self,deltas,acts):
        #make all placeholders for gradients
        updates = np.zeros(np.shape(self._weights))
        bias_updates = np.zeros(np.shape(self._bias))
        deltas = np.reshape(deltas,self._out_shape)
        acts_M,acts_X,acts_Y = self._in_shape
        dOut = np.zeros((acts_M,acts_X+self._padding*2,acts_X+self._padding*2))
        dNet = self.activation_function_deriv(deltas)
        if self._padding != 0:
            acts = np.array([np.pad(a,self._padding,'constant') for a in acts])

        '''
        #'paint' the gradients onto the dOut map, build weight grads
        for i in range(len(dOut[0])-self._kernel_size+1):
            for ii in range(len(dOut[0])-self._kernel_size+1):
                inp = acts[:,ii:ii+self._kernel_size,i:i+self._kernel_size]
                for n in range(self._size):
                    updates[n] += inp*dNet[n][ii][i]
                    bias_updates[n] += dNet[n][ii][i]
                    dOut[:,ii:ii+self._kernel_size,i:i+self._kernel_size] += self._weights[n]*dNet[n][ii][i]
        '''
        #im2col backprop

        flattened_weights = np.reshape(self._weights,(self._size,np.ma.size(self._weights[0])))

        flattened_acts = list()
        for i in range(len(dOut[0])-self._kernel_size+1):
            for ii in range(len(dOut[0])-self._kernel_size+1):
                flattened_acts.append(np.reshape(acts[:,ii:ii+self._kernel_size,i:i+self._kernel_size],(self._expected_map_size*self._kernel_size**2,)))
        flattened_acts = np.array(flattened_acts)

        flattened_dNet = np.reshape(dNet,(self._size,len(dNet[0])**2))

        updates = flattened_dNet @ flattened_acts
        updates = np.reshape(updates,(self._size,acts_M,self._kernel_size,self._kernel_size))
        bias_update = np.sum(flattened_dNet)
        flat_dOut = flattened_weights.T @ flattened_dNet
        
        for (x,y),val in np.ndenumerate(flat_dOut.T):
            a,b,c = self._im2col_cache[x][y]
            dOut[a,b,c] += val

        if self._padding != 0:
            dOut = dOut[:,self._padding:-1*self._padding,self._padding:-1*self._padding]
        
        return dOut,[updates,bias_updates]

class max_pool:
    
    def __init__(self,kernel_size=2):
        self._kernel_size = kernel_size
        self._recent_pass_map = list()
        self._out_shape = ()
        self._in_shape = ()
        self._trainable = False

    def forward(self,inputs):
        self._in_shape = np.shape(inputs)
        self._recent_pass_map = list()
        maxs = list()
        for n in inputs:
            grid = list()
            max_grid = list()
            for x in range(int(len(n)/self._kernel_size)):
                row = list()
                max_row = list()
                for y in range(int(len(n)/self._kernel_size)):
                    biq = n[x*self._kernel_size:x*self._kernel_size+self._kernel_size,y*self._kernel_size:y*self._kernel_size+self._kernel_size]
                    biq = np.reshape(biq,(self._kernel_size**2,))
                    maxim = np.amax(biq)
                    row.append(maxim)
                    for num in range(len(biq)):
                        if biq[num] == maxim:
                            max_row.append(num)
                grid.append(row)
                max_grid.append(max_row)
            maxs.append(grid)
            self._recent_pass_map.append(max_grid)
        self._out_shape = np.shape(maxs)
        return np.array(maxs)

    #takes deltas, rebuilds larger map with 0s around true gradients
    def backward(self,deltas,intro_acts):
        deltas = np.reshape(deltas,self._out_shape)
        temp = np.zeros(self._in_shape,dtype=float)
        
        for maps in range(len(deltas)):
            for s_y in range(len(deltas[0])):
                for s_x in range(len(deltas[0][0])):
                    spot = self._recent_pass_map[maps][s_y][s_x]
                    new_x = int(spot/self._kernel_size)
                    new_y = spot%self._kernel_size
                    temp[maps][(s_y*self._kernel_size)+new_x][(s_x*self._kernel_size)+new_y] += deltas[maps][s_y][s_x]
        return temp,[0,0]
            

class dense:

    def __init__(self,eis,nodes,init_scalar=1,act_func="sigmoid",lr=0.02,gc=False):
        self._in_size = eis
        self._size = nodes
        self._act_func = act_func
        self._learning_rate = lr
        self._grad_clip = gc
        self._trainable = True

        self._weights = (2*np.random.random((self._size,self._in_size)) -1)*init_scalar
        self._total_update = np.zeros(np.shape(self._weights))
        self._bias = np.zeros((nodes,))

    def activation_function(self,net):
        if self._act_func == "tanh":
            return np.tanh(net)
        elif self._act_func == "sigmoid":
            return sigmoid(net)
        elif self._act_func == "none":
            return net

    def activation_function_deriv(self,grad):
        if self._act_func == "tanh":
            return (1.0-(np.tanh(grad)**2))
        elif self._act_func == "sigmoid":
            return sigmoid(grad)*(1-sigmoid(grad))
        elif self._act_func == "none":
            return grad

    def clip_grads(self,update):
        clipped_update = list()
        node_shape = np.shape(update[0])
        for node_update in update:
            node_update = np.reshape(node_update,(np.ma.size(node_update),))
            norm = np.linalg.norm(node_update)
            if norm != 0.0: node_update /= norm
            clipped_update.append(np.reshape(node_update,node_shape))
        return np.array(clipped_update)

    def forward(self,inputs):
        inputs = np.reshape(inputs,self._in_size)

        acts = list()
        rows = np.dot(self._weights,inputs)
        for r in range(len(rows)):
            acts.append(self.activation_function(rows[r]+self._bias[r]))
        return np.array(acts)

    def backward(self,deltas,forward_acts):
        forward_acts = np.reshape(forward_acts,self._in_size)
        
        dNet = np.array([self.activation_function_deriv(delt) for delt in deltas])
        updates = np.array([forward_acts*d for d in dNet])
        dOut = self._weights.T*dNet
        dOut = [np.sum(r) for r in dOut]
        
        if self._grad_clip: dOut=self.clip_grads(dOut)
        return dOut,[updates,dNet]

class batchnorm_conv:

    def __init__(self,in_feature_maps,lr=0.002):
        self._trainable = False
        self._size = in_feature_maps
        self._cache = ()
        self._out_shape = ()
        self._lr=lr

        self._gammas = np.ones((self._size,))
        self._betas = np.zeros((self._size,))

    def forward(self,acts):
        #flatten each feature map first
        acts_shape = np.shape(acts)
        flattened_acts = np.array([np.reshape(maps,np.ma.size(maps)) for maps in acts])
        maps,map_size = np.shape(flattened_acts)

        mean_acts = 1.0/map_size * np.sum(flattened_acts, axis = 1)
        mean_acts_2d = np.broadcast_to(mean_acts,(map_size,maps))
        xmu = flattened_acts - mean_acts_2d.T

        squares = xmu ** 2
        acts_var = 1.0/map_size * np.sum(squares, axis = 1)
        acts_var += 0.0001
        sqrtvar = np.sqrt(acts_var)
        invar = 1.0/sqrtvar

        xhat = xmu.T * invar
        gammax = xhat * self._gammas
        normed_acts = gammax + self._betas

        self._cache = (xhat.T,xmu,invar,sqrtvar,acts_var)
        ret = np.reshape(normed_acts.T,acts_shape)
        self._out_shape = np.shape(ret)
        return ret

    def backward(self,deltas,acts):
        #unfold the variables stored in cache
        xhat,xmu,invar,sqrtvar,acts_var = self._cache
        self._cache=()
        
        deltas = np.reshape(deltas,self._out_shape)
        deltas_shape = np.shape(deltas)
        flattened_deltas = np.array([np.reshape(maps,np.ma.size(maps)) for maps in deltas])
        grad_maps,map_size = np.shape(flattened_deltas)

        dbeta = np.sum(flattened_deltas, axis=1)
        dgamma = np.sum(flattened_deltas*xhat, axis=1)
        dxhat = flattened_deltas.T * self._gammas

        divar = np.sum(dxhat.T*xmu, axis=1)
        dxmu1 = dxhat * invar
        
        dsqrtvar = -1.0 /(sqrtvar**2) * divar
        dvar = 0.5 * 1.0 /np.sqrt(acts_var) * dsqrtvar
        dsq = 1.0 /map_size * np.ones((grad_maps,map_size))
        dsq = dsq.T * dvar
        dxmu2 = 2 * xmu * dsq.T
        
        dx1 = (dxmu1.T + dxmu2)
        dmu = -1.0 * np.sum(dxmu1.T+dxmu2, axis=0)
        dx2 = 1.0 /map_size * np.ones((grad_maps,map_size)) * dmu
        deltas = dx1 + dx2

        self._gammas -= dgamma*self._lr
        self._betas -= dbeta*self._lr
        
        return deltas,[0,0]

class dropout:

    def __init__(self,p):
        self._trainable = False
        self._prob = p
        self._cache = []
        self._in_shape = ()

    def forward(self,acts):
        self._in_shape = np.shape(acts)
        self._cache = (np.random.random(self._in_shape) < self._prob) / self._prob
        return self._cache*acts

    def backward(self,grads,acts):
        grads = np.reshape(grads,self._in_shape)
        return self._cache*grads,[0,0]

        
        
        
