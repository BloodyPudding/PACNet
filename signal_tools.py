import numpy as np
import math
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_filter_bank(data, frequency,bandwidth, fs, order=1):
    return butter_bandpass_filter(data, frequency-bandwidth/2, frequency+bandwidth/2, fs, order)

def butter_bandpass_filter_fast(data, lowcut, highcut, fs, order=1):
    assert len(data.shape)==2
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = np.zeros(data.shape)
    for i in range(data.shape[0]):
        y[i] = lfilter(b,a,data[i])
    return y

def gen_wave(f = 2,time = 1,srate = 1000,phase=0):
    time  = np.linspace(-time/2,time/2,int(srate*time))
    cmw = np.exp(1j*2*math.pi*f*(time+(phase/360)/f))
    return cmw

def gen_wavelet(f = 100,cycle = 7,time_window = 0.1,srate = 1000,phase=0):
    time  = np.linspace(-time_window,time_window,int(srate*time_window*2))
    s = cycle/(2*math.pi*f)
    cmw = np.exp(1j*2*math.pi*f*(time+(phase/360)/f))
    return cmw*np.exp( (-time**2) / (2*s**2) )

def get_cpu_threads(disable_ht = False):
    from cpuinfo import get_cpu_info
    cpu_info = get_cpu_info()
    print("CPU:%s" % cpu_info['brand_raw'])
    threads_num = cpu_info['count']
    if 'ht' in cpu_info['flags'] and disable_ht:
        threads_num = int(threads_num/2)
    print("Using %d Threads" % threads_num)
    return threads_num