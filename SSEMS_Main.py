"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import keras
from keras.models import model_from_json
from keras import backend as K

import scipy.io
import scipy.stats
import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
import numpy as np
import numpy.matlib
import sys
batch_size=1


def get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    return FileList;
	
def make_spectrum_with_phase(y, Noisy=False):
    
    F = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    phase = np.angle(F)
    Lp = np.log10(np.abs(F)**2)

    if Noisy==True:
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase

def make_spectrum_for_ssems(y):

    F = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    Lp=np.abs(F)

    NLp=np.reshape(Lp.T,(1,Lp.shape[1],257))
    return NLp

def convert_to_ssems_input(x_enh,phase):
    x_enh=recons_spec_phase(np.sqrt(10**(x_enh.T)),phase)
    x_enh=x_enh/np.max(abs(x_enh))

    x_out=make_spectrum_for_ssems(x_enh)
    
    return x_out
	
def recons_spec_phase(mag,phase):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming)
    return result            
	
	
def Selecting_best(Test_Noisy_paths, Test_Noisy_wavename):
    os.system("mkdir LSTM_best_model/")
	
    print 'load SE model'	
    Md1_NamePath='WSJ_LSTM_2D_MaleHSNR'
    Md2_NamePath='WSJ_LSTM_2D_MaleLSNR'
    Md3_NamePath='WSJ_LSTM_2D_FemaleHSNR'
    Md4_NamePath='WSJ_LSTM_2D_FemaleLSNR'
   
    with open(Md1_NamePath+'.json', "r") as f:
        model_1 = model_from_json(f.read());
    model_1.load_weights(Md1_NamePath+'.hdf5');
    
    with open(Md2_NamePath+'.json', "r") as f:
        model_2 = model_from_json(f.read());
    model_2.load_weights(Md2_NamePath+'.hdf5');

    with open(Md3_NamePath+'.json', "r") as f:
        model_3 = model_from_json(f.read());
    model_3.load_weights(Md3_NamePath+'.hdf5');
    
    with open(Md4_NamePath+'.json', "r") as f:
        model_4 = model_from_json(f.read());		
    model_4.load_weights(Md4_NamePath+'.hdf5');

    print ('Quality-Net loading...') 
    MdNamePath='Quality-Net_(Non-intrusive)'
    with open(MdNamePath+'.json', "r") as f:
         Quality = model_from_json(f.read());
    Quality.load_weights(MdNamePath+'.hdf5');
    
    print 'Testing...'
    for path in Test_Noisy_paths:   
        S=path.split('/')
        wave_name=S[-1]
        
        noisy, rate  = librosa.load(path,sr=16000)
        noisy_LP, noisy_phase=make_spectrum_with_phase(noisy, Noisy=True)
        
       
        m1=np.squeeze(model_1.predict(noisy_LP, verbose=0, batch_size=batch_size))
        m2=np.squeeze(model_2.predict(noisy_LP, verbose=0, batch_size=batch_size))
        m3=np.squeeze(model_3.predict(noisy_LP, verbose=0, batch_size=batch_size))
        m4=np.squeeze(model_4.predict(noisy_LP, verbose=0, batch_size=batch_size))

        m1=convert_to_ssems_input(m1, noisy_phase)
        m2=convert_to_ssems_input(m2, noisy_phase)
        m3=convert_to_ssems_input(m3, noisy_phase)
        m4=convert_to_ssems_input(m4, noisy_phase)
        
        PESQ_1=Quality.predict(m1, verbose=0, batch_size=1)
        PESQ_2=Quality.predict(m2, verbose=0, batch_size=1)
        PESQ_3=Quality.predict(m3, verbose=0, batch_size=1)
        PESQ_4=Quality.predict(m4, verbose=0, batch_size=1)

        num_model = 4
        id_dat = 0 
        best = 'PESQ_%d'%(1,)
        best = eval(best) 

        for i in range(num_model):
            input = 'PESQ_%d'%(i+1,)
            curr_pesq = eval(input)
            if curr_pesq >= best:
               best = curr_pesq
               id_data = i + 1     
      
        final_enh = 'm%d'%(id_data,)		
        enhanced_LP = eval(final_enh)
        enhanced_LP = np.log10(np.abs(enhanced_LP)**2)
        enhanced_LP = np.squeeze(enhanced_LP)

        enhanced_wav=recons_spec_phase(np.sqrt(10**(enhanced_LP.T)),noisy_phase)        
        enhanced_wav=enhanced_wav/np.max(abs(enhanced_wav)) 

        librosa.output.write_wav(os.path.join("LSTM_best_model",wave_name), enhanced_wav, 16000)
          
if __name__ == '__main__':	
   
    print 'Testing Best Pesq...'
    Test_Noisy_paths = get_filenames('/List/'+sys.argv[1])
    Test_Noisy_wavename=[]
    for path in Test_Noisy_paths:
       S=path.split('/')[-1]
       Test_Noisy_wavename.append(S)
    Selecting_best(Test_Noisy_paths, Test_Noisy_wavename)
    print 'Complete Testing Stage'
