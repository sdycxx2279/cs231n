import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    monitor = (scores - correct_class_score + 1 > 0)
    for j in range(num_classes):
      if j == y[i]:
        dW[:,j] -=  ((np.sum(monitor) - 1) * X[i]).T
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      
      dW[:,j] +=  (monitor[j] * X[i]).T
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= (num_train * 1.0) 

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

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
  num_train = X.shape[0]
    
  score = X.dot(W)
  correct_class_score = score[np.arange(num_train), y].reshape(-1,1)
  #此处所得margin并不是准确的margin，对于正确分类，margin为1，需要减去
  margin = np.maximum(score - correct_class_score + 1, 0)
  loss = (np.sum(margin) - num_train) / (num_train * 1.0) + reg * np.sum(W * W)
  #原始做法，错误原因：产生的矩阵类型为bool，无法作为整型进行修改，令bool=5，结果仍为true（1）
  #monitor = (margin > 0)
    
  monitor = margin
  monitor[margin > 0] = 1
  monitor[np.arange(num_train), y] = -(np.sum(monitor, axis = 1) - 1)
  dW = (X.T).dot(monitor) / (num_train * 1.0) + 2 * reg * W
    
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
