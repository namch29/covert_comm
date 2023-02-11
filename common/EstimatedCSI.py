import numpy as np
from numpy.random import randn, rand
from numpy import sqrt

def EstimatedCSI(Para, h_1, h_2):
    M = Para.M
    N = Para.N
    C0 = 0
    C1 = 0
    num_pilots = 10
    total_pilot_training = num_pilots * N
    
    for i in range(total_pilot_training):
        modulated_bit = 0
        s_1n = sqrt(1/2)*(randn()+1j*randn())                   # transmitter's signals - CSCG
        u_n = np.asmatrix(sqrt(1/2)*(randn(M,1)+1j*randn(M,1))) #noise - CSCG Mx1
        y = h_1*s_1n + h_2*s_1n*modulated_bit + u_n             # Mx1 + Mx1 + Mx1
        C0 = C0 + y*y.H                                         #MxM
        
    for i in range(total_pilot_training):
        modulated_bit = 1
        s_1n = sqrt(1/2)*(randn()+1j*randn())                   # transmitter's signals - CSCG
        u_n = np.asmatrix(sqrt(1/2)*(randn(M,1)+1j*randn(M,1))) #noise - CSCG Mx1
        y = h_1*s_1n + h_2*s_1n*modulated_bit + u_n
        C1 = C1 + y*y.H                                         #MxM 
        
    C0 = C0/total_pilot_training
    C1= C1/total_pilot_training
    return C0, C1