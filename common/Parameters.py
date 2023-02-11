class Para(object):
    def __init__(self, vary, M=10, I=100, N=50, n_dataset=100, n_test=100):
        self.M = M      # No. of antennas
        self.I = I     # No. of bits per backscatter frame
        self.N = N     # Number of RF source symbol period over one backscatter symbol
                
        self.num_generate_data_set = n_dataset
        self.num_testing = n_test
        self.num_col = self.M*self.M*2
        
        #SNR
        self.alpha_dt = 10**(vary/10)    # SNR of the direct link (7 dB)
        self.alpha_bt = 10**(-10/10)     # SNR of the bacscatter link (-10 dB)

        print(f'Vary alpha_dt:{vary}')