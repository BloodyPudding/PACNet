from multiprocessing import Pool,Manager
import data_adapter
import numpy as np
from scipy import io as sio
from scipy import signal
import time
import signal_tools
import random

def get_amp_band(pac,ch,pha_band):
    res_f = []
    for k in pha_band.keys():
        tar_sig= pac[k][ch].mean(axis=1)
        amp_f = []
        tar_sig = signal.savgol_filter(tar_sig,9,5)
        peaks = signal.find_peaks(tar_sig)[0]*2+70
        edges = signal.find_peaks(1-tar_sig)[0]*2+70
        for i in range(peaks.shape[0]):
            p = peaks[i]
            if np.where(p > edges)[0].size == 0:
                left= 70
            else:
                left = edges[np.where(p>edges)[0].max()]
            if np.where(p < edges)[0].size == 0:
                right = 200
            else:
                right = edges[np.where(p < edges)[0].min()]
            amp_f.append([left,right])
        res_f.append(amp_f)
    return res_f

def erpac_filter_ch(data,pac_band,sf,order,pha_band,mode='pac',band_limit=5):
    band_signal = np.zeros((4,data.shape[0],data.shape[1]))
    print(f'Pac Mode:{mode} - Sample rate: {sf}')
    for idx in range(data.shape[0]):
        for i,k in enumerate(pha_band.keys()):
            if mode == 'HF':
                limit = 0
                for amp_f in pac_band[i]:
                    band_signal[i][idx] += signal_tools.butter_bandpass_filter(data[idx], amp_f[0],amp_f[1], sf, 1)
                    limit += 1
                    if limit < band_limit:
                        break
            elif mode == 'HF_Full':
                band_signal[i][idx]+=signal_tools.butter_bandpass_filter(data[idx],70,200,sf,1)
            elif mode == 'HF_random':
                for amp_f in range(random.randint(1, 5)):
                    band_signal[i][idx] += signal_tools.butter_bandpass_filter_bank(data[idx], random.randint(70, 200),
                                                                                    20, sf, 1)
            else:
                band_signal[i][idx]=signal_tools.butter_bandpass_filter(data[idx],pha_band[k][0]-0.5,pha_band[k][1]+0.5,sf,order)
                if mode == 'pac':
                    for amp_f in pac_band[i]:
                        band_signal[i][idx] += signal_tools.butter_bandpass_filter(data[idx],amp_f[0],amp_f[1],sf,1)
                if mode == 'full':
                    band_signal[i][idx]+=signal_tools.butter_bandpass_filter(data[idx],70,200,sf,order)
                if mode == 'random':
                    for amp_f in range(random.randint(1,5)):
                        band_signal[i][idx] += signal_tools.butter_bandpass_filter_bank(data[idx],random.randint(70,200),20,sf,1)
    return band_signal

def erpac_filter(x,dataname,subject_id,sf,order=1,pac_mode='pac',threads_num=24,band_limit=5):
    pha_band = {
        "delta": [1, 3, 1],
        "theta": [4, 8, 1],
        "alpha": [8, 12, 2],
        "beta": [12, 20, 2]
    }
    erpac_res = sio.loadmat(f'pac_result/{dataname}_ergcpac_subject_{subject_id}.mat')
    erpac_filter_paras = []
    filter_res = []
    if threads_num>1:
        for ch in range(x.shape[1]):
            amp_band = get_amp_band(erpac_res,ch,pha_band)
            #filter_res.append(erpac_filter_ch(x[:, ch, :], get_amp_band(erpac_res, ch), sf, order,pha_band))&$
            erpac_filter_paras.append([x[:,ch,:],amp_band,sf,order,pha_band,pac_mode])
        with Pool(threads_num) as p:
            filter_res = p.starmap(erpac_filter_ch, erpac_filter_paras)
    else:
        for ch in range(x.shape[1]):
            print(ch)
            amp_band = get_amp_band(erpac_res,ch,pha_band)
            filter_res.append(erpac_filter_ch(x[:, ch, :], amp_band, sf, order,pha_band,pac_mode,band_limit))
    filter_res = np.array(filter_res)
    filter_res = filter_res.swapaxes(0,1)
    filter_res = filter_res.swapaxes(1,2)
    return filter_res

def direct_band_filter(x,band_l,band_h,sf,order=1):
    y = np.zeros(x.shape)
    for idx in range(x.shape[0]):
        for ch in range(x.shape[1]):
            y[idx][ch]= signal_tools.butter_bandpass_filter(x[idx][ch], band_l, band_h, sf, order)
    return y
# if __name__ == '__main__':
#     subject_id = 1
#     kjm_ff = data_adapter.fingerflex('../ecog_kjm_data/fingerflex/')
#     x, y = kjm_ff.load_data(subject_id=subject_id - 1, post_time=500)
#     print(erpac_filter(x,1,1000,threads_num=24).shape)