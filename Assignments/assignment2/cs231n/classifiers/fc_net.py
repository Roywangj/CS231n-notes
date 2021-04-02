from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # randn函数是基于零均值和标准差的一个高斯分布
        W1 = weight_scale*np.random.randn(input_dim,hidden_dim)#(3072，100)
        W2 = weight_scale*np.random.randn(hidden_dim,num_classes)#(100,10)
        b1 = np.zeros((hidden_dim,))
        b2 = np.zeros((num_classes,))
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1=self.params['W1']
        W2=self.params['W2']
        b1=self.params['b1']
        b2=self.params['b2']
        out1,cache1=affine_relu_forward(X,W1,b1)
        scores,cache2=affine_forward(out1,W2,b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #后向传播，计算loss和梯度
        loss,dscores=softmax_loss(scores,y)#softmax_loss在layers.py中
        loss+=0.5*self.reg*(np.sum(self.params['W1']**2)+np.sum(self.params['W2']**2))
        dout1,dw2,db2=affine_backward(dscores,cache2)#第二层反向传播，affine_backward函数在layers.py中
        dX,dw1,db1=affine_relu_backward(dout1,cache1)#第一层反向传播，affine_relu_backward函数在layer_utils.py中
        dw1+=self.reg*self.params['W1']
        dw2+=self.reg*self.params['W2']
        grads['W1']=dw1
        grads['b1']=db1
        grads['W2']=dw2
        grads['b2']=db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #初始化所有隐藏层的参数
        layer_input_dim=input_dim #D
        for i,h_dim in enumerate(hidden_dims): #(1,H1)(2,H2)
          self.params['W%d'%(i+1)]=weight_scale*np.random.randn(layer_input_dim,h_dim)#W1(D，H1) W2(H1,H2)
          self.params['b%d'%(i+1)]=np.zeros((h_dim,))#b1(H1,) b2(H2,)
          if self.normalization=='batchnorm' or self.normalization=='layernorm':
            self.params['gamma%d'%(i+1)]=np.ones((h_dim,))#初始化为1
            self.params['beta%d'%(i+1)]=np.zeros((h_dim,))#初始化为0
          layer_input_dim=h_dim#将该层的列数传给下一层的行数
        
        #初始化所有输出层的参数
        self.params['W%d'%self.num_layers]=weight_scale*np.random.randn(layer_input_dim,num_classes)
        self.params['b%d'%self.num_layers]=np.zeros((num_classes,))



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        #  当开启 dropout 时，我们需要在每一个神经元层中传递一个相同的 dropout 参数字典 self.dropout_param ,
        #  以保证每一层的神经元们 都知晓失活概率p和当前神经网络的模式状态mode(训练／测试)。 

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {} #dropout的参数字典
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed
        

        #当开启批量归一化时，我们要定义一个BN算法的参数列表 self.bn_params ， 以用来跟踪记录每一层的平均值和标准差。
        #其中，第0个元素 self.bn_params[0] 表示前向传播第1个BN层的参数，
        #第1个元素 self.bn_params[1] 表示前向传播 第2个BN层的参数，以此类推。
        
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []#BN的参数字典
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        fc_mix_cache={} #初始化每层前向传播的缓冲字典
        if self.use_dropout:  #如果开启了dropout，初始化其对应的缓冲字典
          dp_cache={}
        #从第一个隐藏层开始循环，每一个隐藏层，传递数据out,保存每一层的缓冲cache        
        out=X
        for i in range(self.num_layers-1):
          w=self.params['W%d'%(i+1)]
          b=self.params['b%d'%(i+1)]
          if self.normalization=='batchnorm':
            gamma=self.params['gamma%d'%(i+1)]
            beta=self.params['beta%d'%(i+1)]
            out,fc_mix_cache[i]=affine_bn_relu_forward(out,w,b,gamma,beta,self.bn_params[i])
          else:
            out,fc_mix_cache[i]=affine_relu_forward(out,w,b)
          if self.use_dropout:
            out,dp_cache[i]=dropout_forward(out,self.dropout_param)
        #最后的输出层
        w=self.params['W%d'%(self.num_layers)]
        b=self.params['b%d'%(self.num_layers)]
        out,out_cache=affine_forward(out,w,b)
        scores=out


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss,dout=softmax_loss(scores,y)
        loss+=0.5*self.reg*np.sum(self.params['W%d'%(self.num_layers)])
        #输出层梯度的反向传播,并把梯度保存在梯度字典grad[]中
        dout,dw,db=affine_backward(dout,out_cache)
        grads['W%d'%(self.num_layers)]=dw+self.reg*self.params['W%d'%(self.num_layers)]
        grads['b%d'%(self.num_layers)]=db
        #在每一个隐藏层处梯度的反向传播，不仅更新了梯度字典grad[]，还迭代计算出了loss
        for i in range(self.num_layers-1):
          ri=self.num_layers-i-2#倒数第ri+1隐藏层
          loss+=0.5*self.reg*np.sum(self.params['W%d'%(ri+1)]**2) #迭代地补上每层的正则项给loss
          if self.use_dropout:
            dout=dropout_backward(dout,dp_cache[ri])
          if self.normalization=='batchnorm':
            dout,dw,db,dgamma,dbeta=affine_bn_relu_backward(dout,fc_mix_cache[ri])
            grads['gamma%d'%(ri+1)]=dgamma
            grads['beta%d'%(ri+1)]=dbeta
          else:
            dout,dw,db=affine_relu_backward(dout,fc_mix_cache[ri])
          grads['W%d' %(ri+1,)] = dw + self.reg * self.params['W%d' %(ri+1,)]
          grads['b%d' %(ri+1,)] = db

          
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
