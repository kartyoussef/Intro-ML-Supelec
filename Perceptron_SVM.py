######################################################################
#    Perceptron et soft-SVM avec descente de gradient stochastique
#               Author :  Youssef Kartit
######################################################################

#### Librairies
import csv
import random as rd
import time
import numpy as np
#import matplotlib.pyplot as plt

# Prepare Data
def csv_2_list(file_x, file_y):
    ''' This function converts csv files of the dataset X and y to Python lists '''
    #Read the csv files file_x and file_y
    with open(file_x) as data_x:
        dataset_x = csv.reader(data_x, delimiter = ',')
        list_x = list(dataset_x)
    with open(file_y) as data_y:
        dataset_y = csv.reader(data_y)
        list_y = list(dataset_y)
    # Convert values
    for i in range(len(list_x)):
        for j in range(len(list_x[0])):
            list_x[i][j] = float(list_x[i][j])
        list_y[i] = int(list_y[i][0])
    return(list_x,list_y)

def build_training_set(x, y, m_prime):
   ''' This function ''' 
   #lenght of the dataset m (number of samples)
   m = len(x)
   x_train, y_train = [], []
   training_index = []
   validation_index = [k for k in range(m)] 
   while len(training_index) != m_prime:
       i_prime = rd.randint(0, m-1)
       if  not(i_prime in training_index):
           training_index.append(i_prime)
           validation_index.remove(i_prime)
           x_train.append(x[i_prime])
           y_train.append(y[i_prime])
   return(x_train, y_train, training_index, validation_index)

#### Mapping Data
#def mapping(x, ):

#### Average Error Rate
def evaluate_algorithm(algorithm, x, y, *params):
    ''' This function calculate the average error rate for algorithm
    params = (m_prime, step, l_rate)
    '''
    avg_err = 0
    m = len(x)
    d = len(x[0])
    m_prime = params[0]
    if algorithm == "perceptron":m_prime_time_list = []
    for loop in range(100):
        xt_, yt_, training_index, validation_index = build_training_set(x, y, m_prime)
        if algorithm == "perceptron":
            w_, dt = perceptron(xt_, yt_)
            m_prime_time_list.append(dt)
        elif algorithm == "soft-SVM":
            w_ = soft_SVM(xt_, yt_, params[1] , m_prime, params[2])
        err = 0
        for v in validation_index:
            S = sum(w_[k]*x[v][k] for k in range(d))
            if y[v]*S < 0:
                err+=1
            else: continue
        err = (1/(m-m_prime))*err
        avg_err += err
    avg_err = avg_err/100
    if algorithm == "perceptron": 
        return(avg_err,m_prime_time_list)
    elif algorithm == "soft-SVM":
        return avg_err   
     
#### Perceptron Algorithm
def perceptron(x, y):
    ''' This function implements the perceptron algorithm , an algorithm due to Rosenblatt which finds
a separating hyperplane '''
    # Initialize variables 
    d = len(x[0])
    m = len(x)
    w = [[0 for _ in range(d)]]
    t = 0
    # Variable used to calculate the perceptron running time
    p_time  = time.time()
    # Perceptron algorithm 
    while True: 
        for l in range(m):
            #Calcul de S = <w(t),xi>yi : 
            S  = sum(w[t][k]*x[l][k] for k in range(d))
            test = y[l]*S
            if test <= 0:
                w.append(list(np.add(w[t], np.multiply(y[l],x[l]))))
                t += 1
                break
            elif (test > 0 and l < m-1) :
                continue
            else : 
                return(w[t],time.time()-p_time)

#### soft-SVM              
def soft_SVM(x, y, step, T, l_rate):
    ''' This function '''
    d = len(x[0])
    m_prime = len(x)
    w = [[0 for _ in range(d)]]
    w_ = [[0 for _ in range(d)]]
    for t in range(T):
        i = round(rd.uniform(0, m_prime-1))
        S = y[t]*sum(w[t][k]*x[i][k] for k in range(d))
        #
        vt_ = np.multiply(2*l_rate,w[t])-np.multiply(y[i],x[i])*int(S < 1)
        #
        w.append(np.subtract(w[t],np.multiply(step,vt_)))
        w_ = np.add(w_,w[t+1])
    w_ = np.multiply(1/T, w_)
    return w_