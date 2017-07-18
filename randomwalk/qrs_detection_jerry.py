#encoding:utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import codecs
import os
import scipy.io as sio
from scipy import signal
import pywt
from pywt import wavedec
from pywt import waverec
def pre_sig(rawsig):
    # fs=250
    N=8
    wt_sig=DTCWT(lead_signal=rawsig,N=8)
    qrs_sig=np.sum(abs(wt_sig[2:7,:]),axis=0)
    return qrs_sig

def DTCWT(lead_signal, N=8):
    HL = [0.03516384, 0, -0.08832942, 0.23389032, 0.76027237, 0.58751830, 0, -0.11430184, 0, 0]
    LD_a, HD_a, LR_a, HR_a, LD_b, HD_b, LR_b, HR_b = Qshift(HL)
    filter_a = [LD_a, HD_a, LR_a, HR_a]
    wavelet_a = pywt.Wavelet(name='wavelet_a', filter_bank=filter_a)
    filter_b = [LD_b, HD_b, LR_b, HR_b]
    wavelet_b = pywt.Wavelet(name='wavelet_b', filter_bank=filter_b)
    C1 = wavedec(lead_signal, wavelet=wavelet_a, level=N, axis=-1)
    # print C1[0]
    C2 = wavedec(lead_signal, wavelet=wavelet_b, level=N, axis=-1)
    Ra_sig = np.zeros((10, len(lead_signal)))
    Rb_sig = np.zeros((10, len(lead_signal)))
    R_sig = np.zeros((10, len(lead_signal)))
    for i in range(len(C1)):
        tmp = [np.zeros(x.shape) for x in C1]
        tmp[i] = C1[i]
        Ra_sig[i, :] = np.squeeze(waverec(tmp, wavelet=wavelet_a, axis=-1))
        tmp[i] = C2[i]
        Rb_sig[i, :] = np.squeeze(waverec(tmp, wavelet=wavelet_b, axis=-1))
        R_sig = (Ra_sig + Rb_sig) / 2.0
    return R_sig

def Qshift(HL):
    L = len(HL)
    H00a, H01a, H10a, H11a, H00b, H01b, H10b, H11b, Coef_HL \
        = np.zeros(L), np.zeros(L), \
          np.zeros(L), np.zeros(L), \
          np.zeros(L), np.zeros(L), \
          np.zeros(L), np.zeros(L), np.zeros(L)
    for i in range(L - 1, -1, -1):
        Coef_HL[L - 1 - i] = math.ceil(L / 2) - i
    for j in range(L):
        H00a[j] = HL[L - 1 - j]
        H01a[j] = ((-1) ** Coef_HL[j]) * HL[j]
        H00b[j] = HL[j]
        H01b[j] = ((-1) ** Coef_HL[L - 1 - j]) * HL[L - 1 - j]
        H10a[j] = HL[j]
        H11a[L - 1 - j] = H01a[j]
        H10b[j] = HL[L - 1 - j]
        H11b[L - 1 - j] = H01b[j]
    return H00a.tolist(), H01a.tolist(), H10a.tolist(), H11a.tolist(), \
           H00b.tolist(), H01b.tolist(), H10b.tolist(), H11b.tolist()


def qrs_detector(re_rawsig,fs=250):
    qrs_sig=pre_sig(re_rawsig)
    Xmax=list()
    Ymax=list()
    l=len(qrs_sig)
    detect_win=int(math.floor(150.0/1000*fs))
    baseline=np.ones(l)
    for i in range(2*fs,l-2*fs):
        baseline[i]=np.mean(qrs_sig[i-2*fs:i+2*fs+1])
    for i in range(0,2*fs):baseline[i]=baseline[2*fs]
    for i in range(l-2*fs,l):baseline[i]=baseline[l-2*fs-1]
    for i in range(detect_win,l-detect_win):
        qrs_win=qrs_sig[i-detect_win:i+detect_win+1]
        xmax=np.argmax(qrs_win,axis=0)
        ymax=np.amax(qrs_win,axis=0)
        if ymax>2*baseline[i] and xmax==detect_win:
            Xmax.append(i)
            Ymax.append(ymax)
    return Xmax


if __name__ == '__main__':
   path='/home/chenbin/hyf/Sourecode/jerry/jerry_py'
   files=os.listdir(path)
   for file in files:
       if file[-5:]=='m.mat':
           Fs=500
           matpath=os.path.join(path,file)
           rawdata=sio.loadmat(matpath)
           rawsig=np.squeeze(rawdata['II'])
           re_rawsig=signal.resample(rawsig,int(len(rawsig)/float(Fs)*250))
           re_loc_qrs=qrs_detector(re_rawsig)
           loc_qrs=[int(x*Fs/250.0) for x in re_loc_qrs]
           amp_qrs=rawsig[np.array(loc_qrs)]
           plt.plot(rawsig,'r')
           plt.plot(loc_qrs,amp_qrs,'bo')
           plt.show()
