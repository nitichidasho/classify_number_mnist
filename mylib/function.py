'''
出典：斎藤康毅, ゼロから作る Deep Learning, オライリージャパン, 2018
'''
import numpy as np

#シグモイド関数#
def sigmoid(x):
  y = 1/(1+np.exp(-x))
  return y  

#恒等関数#
def identify_function(x):
  return x

#ソフトマックス関数
def softmax(a):
  c = np.max(a)#オーバフロー対策#
  exp_a = np.exp(a-c)#"a-c"オーバフロー対策#
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

#2乗和誤差
def mean_squared_error(y,t):
 return 0.5 * np.sum((y-t)**2)

#交差エントロピー誤差(tがデータラベルで与えられるとき)
def cross_entropy_error(y,t):
 if y.ndim == 1:           #1次元の配列を、形式上２次元配列に変える。
   y = y.reshape(1,y.size)
   t = t.reshape(1,t.size)
 batch_size = y.shape[0]
 return - np.sum(np.log(y[np.arange(batch_size),t] + le-7 )) / batch_size #le-7はオーバフロー対策

#交差エントロピー誤差(tがone_hot表現で与えられるとき)
def cross_entropy_error_one_hot(y,t):
 if y.ndim == 1:
   y = y.reshape(1,y.size)
   t = t.reshape(1,t.size)
 batch_size = y.shape[0]
 return -np.sum(t * np.log(y + le-7 )) / batch_size#le-7はオーバフロー対策

#試験用２変数関数
def function_2(x):
  return np.sum(x**2)   #x^2 + y^2

#試験用１変数関数
def function_1(x):
  return 0.01*x**2 + 0.1*x

#数値微分
def numerical_diff(f,x):
  h = le-4
  return (f(x+h) - f(x-h)) / (2*h)

#勾配ベクトル(2変数関数のある点の、偏微分nによるベクトル)
def numerical_gradient(f,x):
  h = le-4
  grad = np.zeros_like(x)  #xと同じ形状の配列を生成(要素は全部０)

  for idx in (0,x.shape):
    temp = x[idx]
    #+h分の計算
    x[idx] = temp + h
    fx1 = f(x)
    #-h分の計算
    x[idx] = temp - h
    fx2 = f(x)
  
    grad[idx] = (fx1-fx2) / (2*h) 
    x[idx] = temp                 #値を元に戻す

  return grad

#勾配降下法
def gradient_descent(f,init_x,lr,step_num):
  x = init_x #初期位置,例(1.5,2.0)

  for i in range(step_num):
    grad = numerical_gradient(f,x)
    x -= lr * grad    #lrは学習率、ハイパーパラメータともいう。
  
  return x

#



    
     


 
 