#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Labb 7; Eigenvalue-algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import qr
from scipy.linalg import inv

"""
 Lab 6, Part 1- The Jacobi Method
"""

# Q.1 given code
def Create2by2Matrix ( lambda_a , lambda_b ):
   P=np.array ([[0.5 , 0.5], [0.2 , 0.8]])
   D=np.array ([[ lambda_a , 0.0], [0.0 , lambda_b ]])
   Pinv =inv( P )
   return np.matmul(np.matmul(P , D ) , Pinv )


A=Create2by2Matrix(4.002,4)
print(A)




# Q.2 - QR-algorithm for calculating eigenvalues

def qralgorithm_eigen(matrixA,n_iter):
    iterration =0
    
    for i in range(n_iter):
        Q, R= qr(matrixA)
        matrixA= np.dot(R, Q)
        iterration+=1
        
    return np.diag(matrixA)
    
print(qralgorithm_eigen(A,10000))



"""
Q.2. Yes ,we will the right matrix after some iteration.
"""




# Q.3  Power iterations

# calcutating the dominant eigen value using power iteration


# Have used some code from wikipedia as given in the question
# but made changes in order to get the eigenvalue not the eigenvector.
def power_iteration(A, num_simulations: int):
    
    v_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        
        v_k1 = np.dot(A, v_k)

        
        v_k1_norm = np.linalg.norm(v_k1)

        
        v_k = v_k1 / v_k1_norm
        
    
        eigen1 = np.matmul(np.transpose(v_k),A)
        eigen2 = np.matmul(eigen1, v_k)
        eigen3= np.matmul(np.transpose(v_k),v_k)
        
        eigen = eigen2/eigen3
    return eigen

print(power_iteration(A,100000))


"""
Q.3: Yes, we will get the right dominant eigenvalue, but if we choose 
the eigenvalue that cannot be resented exactly as binary(base 2),
then you will get some rounding error as well.
"""



"""
Q.4: With letting the two eigenvalues to be very close to each other,
the effects can be that it will take a lot longer to converge than if
the two eigenvalues aren't close to each other.


You can choose the eigenvalues close to each other and try it out 
yourself in the function above.
"""













