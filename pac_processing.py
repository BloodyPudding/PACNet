import data_adapter
import numpy as np
from scipy import io as sio
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import signal_tools
import argparse

def erpac(x,ch,sf=1000,f_pha=[1,3], f_amp=[(70,200,4,2)],method='gc'):
    p = EventRelatedPac(f_pha=f_pha, f_amp=f_amp)
    pha = p.filter(sf, x, ftype='phase', n_jobs=1)
    amp = p.filter(sf, x, ftype='amplitude', n_jobs=1)
    res = p.fit(pha, amp, method=method, smooth=100, n_jobs=1).squeeze()
    return ch,res

def pac_process(data,ch,sf=1000,f_pha=[(2,40,1, 0.5)], f_amp=(60, 200, 2, 1)):
    p_obj = Pac(idpac=(6, 0, 0), f_pha=f_pha, f_amp=f_amp)
    # extract all of the phases and amplitudes
    pha_p = p_obj.filter(sf, data, ftype='phase',n_jobs=1)
    amp_p = p_obj.filter(sf, data, ftype='amplitude',n_jobs=1)
    res = p_obj.fit(pha_p,amp_p)
    return ch,res

def pac_pararrel(x,sf,f_pha,f_amp,pac_method):
    start_time = time.time()
    paras = []
    for ch in range(x.shape[1]):
        paras.append([x[:, ch, :],ch,sf,f_pha,f_amp])
    threads_num = signal_tools.get_cpu_threads(disable_ht=True)
    with Pool(threads_num) as p:
        res = p.starmap(pac_method, paras)
    print('used %d second' % (time.time() - start_time))
    pac_res = np.zeros((len(res), res[0][1].shape[0], res[0][1].shape[1]))
    for r in res:
        pac_res[r[0]] = r[1]
    return pac_res

def get_top_ch(subject,pac_mode="HF",top_num=3):
    data = sio.loadmat(f'result/kjm_fh_deeplift_res_subject{subject}.mat')
    data = data[pac_mode].sum(axis=0)
    res = []
    for i in range(top_num):
        top = data.argmax()
        res.append(top)
        data[top]=0
    return res

def rnd_pac(o_x,subject_id,label=0):
    pha_band = {
        "delta": [(1, 3, 0.25, 0.1)],
        "theta": [(4, 8, 0.25, 0.2)],
        "alpha": [(8, 12, 0.25, 0.2)],
        "beta": [(12, 20, 0.5, 0.5)]
    }
    amp_band = {
        "high_gamma": [(70, 200, 4, 2)]
    }
    sf = 1000
    for rid in range(50):
        x = o_x[np.random.randint(0, o_x.shape[0], size=int(o_x.shape[0] * 0.7), dtype=int)]
        print(x.shape)
        s_res = {}
        s_res['channels'] = x.shape[1]
        s_res['sample_rate'] = sf
        for pha_k in pha_band.keys():
            reskey = "{}".format(pha_k)
            print("Processing PAC - Phase Band:{}".format(pha_k))
            band_width = pha_band[pha_k][0][1]
            print(band_width)
            # ecpac
            method = 'gc'
            pf_down = pha_band[pha_k][0][0]
            pf_up = pha_band[pha_k][0][1]
            s_res[reskey] = []
            ch, r = erpac(x, 1, sf, [pf_down, pf_up], [(70 - band_width, 200 + band_width, band_width * 2, 2)])
            s_res[reskey].append(r)
            # gcpac
            # s_res[reskey] = pac_pararrel(x, sf, pha_band[pha_k], [(70-band_width, 200+band_width, band_width * 2, 2)])
        sio.savemat("pac_val/kjm_fh_pac_subject_{}_rnd_{}_max_label_{}_relax.mat".format(subject_id, rid,label), s_res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "set subject"
    parser.add_argument("-s", "--subject", help=help_, default=1, type=int)
    #target_data = data_adapter.kjm_fingerflex('../ecog_kjm_data/fingerflex/')
    #target_data = data_adapter.zju_gesture('../ecog_zju_data/')
    #target_data = data_adapter.zju_mi('0819')
    #target_data = data_adapter.kjm_im('../ecog_kjm_data/imagery_basic/')
    target_data = data_adapter.kjm_fh('../ecog_kjm_data/faces_basic/')
    args = parser.parse_args()

    subject_id =args.subject
    top_chs = get_top_ch(subject_id,pac_mode="HF",top_num=1)
    print(top_chs)
    o_x, y = target_data.load_data(subject_id=subject_id - 1,time_offset=-500)#,pre_time=1500, post_time=-500)
    o_x = o_x[:,top_chs,:].squeeze()
    print(o_x.shape)
    for i in range(2):
        idx = np.where(y==i)[0]
        rnd_pac(o_x, subject_id,label=i)
    #rnd_pac(o_x,subject_id)




