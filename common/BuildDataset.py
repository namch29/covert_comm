import numpy as np
from numpy.random import randn, rand
from numpy import sqrt
import torch as T

'''
v1.1: 07-12-2022
'''
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


def GenerateDataset(Para):
    M = Para.M
    N = Para.N
    I = Para.I

    alpha_dt = Para.alpha_dt
    alpha_bt = Para.alpha_bt
    T = Para.num_generate_data_set # number of realizations

    num_col = M*M*2
    num_row = I*T                   # total number of rows in the dataset

    D = np.zeros((num_row, num_col+1), dtype = "complex_")        #array to save dataset
    row = 0

    for t in range(T):
        # channel coefficient (in this case: small scale fading)
        f_rm = np.asmatrix(1/sqrt(2)*(randn(M,1)+1j*randn(M,1))) # Transmitter to receiver Mx1
        f_bm = np.asmatrix(1/sqrt(2)*(randn(M,1)+1j*randn(M,1))) # Tag to receiver  Mx1

        g_r = 1/sqrt(2)*(randn()+1j*randn())            # transmitter to tag
        
        #Calculate C0 and C1
        h_1 = f_rm * sqrt(alpha_dt)                     #Mx1
        h_2 = g_r * f_bm * sqrt(alpha_bt * alpha_dt)    #Mx1
        
        #Generate original bits
        T_bits = np.round(rand(I))
#         print(f'T_bits{T_bits}')
        
        #Transmit each bit
        previous_bit = 1
        C0, C1 = EstimatedCSI(Para, h_1, h_2)           # MxM and MxM
        C0 = C0 + np.eye(M)
        C1 = C1 + np.eye(M)
     
        for k in range(I):
            bit = T_bits[k]
#             print(f'T_bit: {bit}')
            y_received = 0
            modulated_bit = np.mod(previous_bit+bit,2)
            previous_bit = modulated_bit
            for n in range(N):
                # Randomly generate y_0
                s_1n = sqrt(1/2)*(randn()+1j*randn())                   # transmitter's signals - CSCG
                u_n = np.asmatrix(sqrt(1/2)*(randn(M,1)+1j*randn(M,1))) # noise - CSCG      Mx1
                y = h_1*s_1n + h_2*s_1n*modulated_bit + u_n
                y_received = y_received + y*y.H
           
            y_received = y_received/N                       #Covariance of received signal
      
            raw0 = np.linalg.solve(C0.T, y_received.T).T    #similar to '/' in matlab (solve the equation Ax=B)
            raw1 = np.linalg.solve(C1.T, y_received.T).T    #raw_0 = y_received/C0, raw_1 = y_received/C1;          
            raw0 = raw0.flatten()
            raw1 = raw1.flatten()

            D[row, :M*M] = raw0
            D[row, M*M:M*M*2] = raw1
            D[row, M*M*2] = modulated_bit
            row+=1
    # print(type(D))        
    numRow, numCol = D.shape
    dataset = np.asmatrix(D)
    dataset = np.concatenate((np.real(dataset[:,:numCol-1]),np.imag(dataset[:,:numCol-1]),
                              np.absolute(dataset[:,:numCol-1]),np.real(dataset[:,-1])),axis=1) #last column is label
    
    return D, dataset


class SignalDataset(T.utils.data.Dataset):
    def __init__(self, source_data, num_rows=None):
#         all_data = np.loadtxt(src_file, max_rows=num_rows,
#                                  usecols=range(1,6), 
#                                  delimiter="\t", skiprows=0,
#                                  dtype=np.float32)  # strip IDs off
        device = T.device("cpu") 
        no_row, no_collum = source_data.shape        
        self.x_data = T.tensor(source_data[:,0:no_collum-1],
                                dtype=T.float32).to(device)
        self.y_data = T.tensor(source_data[:,no_collum-1],
                                dtype=T.float).to(device)

        # n_vals = len(self.y_data)
        # self.y_data = self.y_data.reshape(n_vals,1)
        self.y_data = self.y_data.reshape(-1,1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx,:]  # idx rows, all 4 cols
        lbl = self.y_data[idx,:]    # idx rows, the 1 col
        sample = { 'predictors' : preds, 'target' : lbl }
        return sample