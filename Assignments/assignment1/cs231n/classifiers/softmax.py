from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in xrange(num_train):
        scores = X[i].dot(W) # X的第i行，也就是第i个图像点乘参数矩阵，得到的矩阵维度为(1,C)
        scores -= max(scores) # 减去最大值防止计算时数值错误
        loss += -np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores))) # 跟下面的公式是一样的，下面的使用了对数化简公式log(a/b)=log(a)-log(b)
        # loss += -score[y[i]] + np.log(np.sum(np.exp(scores)))
        # 对W的更新要放在i的循环中，每轮都要对所有W进行更新，也即每求出一个X的预测矩阵后都要反向更新W
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (np.exp(scores[j]) / np.sum(np.exp(scores))) * X[i] - X[i] # 代入公式，图中有详细推导
            else:
                dW[:, j] += (np.exp(scores[j]) / np.sum(np.exp(scores))) * X[i]
 
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train
    dW += reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W) # NxC  N对应N个实例，C代表每一项的得分
    # 求概率，用指数矩阵除以每一行的指数行和  
    scores_exp = np.exp(scores) #NxC
    scores_exp_sum_axis_1 = np.sum(scores_exp,axis=1,keepdims=True) #Nx1
    scores_probility = scores_exp / scores_exp_sum_axis_1 #NxC
    # 计算loss
    loss = -np.sum(np.log(scores_probility[range(num_train),y]))
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)

    # 构造一个mask(NxC)，使用X.T * mask(DxN x NxC ) ->dW(DxC)
    mask_minus_ones = np.zeros_like(scores)
    mask_minus_ones[range(num_train),y] = -1 #j=y[i]时，-X[i]那一项
    mask_ones = np.ones_like(scores)
    mask = mask_ones * scores_probility + mask_minus_ones
    dW = np.dot(X.T,mask)
    dW /= num_train
    dW += reg*W

    #print(scores_exp.shape)
    #print(scores_exp_sum_axis_1.shape)
    #print(scores_probility.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
