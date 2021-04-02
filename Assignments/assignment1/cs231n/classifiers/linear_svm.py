from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]# 10
    num_train = X.shape[0]# 500
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) #1x10
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        scores = X[i].dot(W) #1x10
        index = y[i]  # 正确的索引
        for j in range(num_classes):
            if j != index and scores[j] >= scores[index]-1:
                dW[:,j] += X[i]#数据分类错误时的梯度
                dW[:,index] -= X[i]#数据分类正确时的梯度，所有非正确的累减
 
    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=X.shape[0]
    num_classes = W.shape[1]# 10
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero


    scores=X.dot(W)#所有样本的得分情况500x10
    #利用np.arange(),correct_class_score变成了 (num_train,y)的矩阵
    correct_class_score = scores[np.arange(num_train),y]
    correct_class_score = np.reshape(correct_class_score,(num_train,-1))
    margins = scores - correct_class_score + 1
    margins = np.maximum(0, margins)
    #然后这里计算了j=y[i]的情形，所以把他们置为0
    margins[np.arange(num_train),y] = 0
    loss += np.sum(margins) / num_train
    loss += reg * np.sum( W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    margins[margins > 0] = 1
    #因为j=y[i]的那一个元素的grad要计算 >0 的那些次数次
    row_sum = np.sum(margins,axis=1)
    margins[np.arange(num_train),y] = -row_sum.T
    #把公式1和2合到一起计算了
    dW = np.dot(X.T,margins)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
