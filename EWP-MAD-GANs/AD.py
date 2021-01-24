import tensorflow as tf
import numpy as np
import pdb
import json
import model
from mod_core_rnn_cell_impl import LSTMCell  # modified to allow initializing bias in lstm

import utils
import eval
import DR_discriminator
import data_utils

# from pyod.utils.utility import *
from sklearn.utils.validation import *
from sklearn.metrics.classification import *
from sklearn.metrics.ranking import *
from time import time

begin = time()

"""
Here, only the discriminator was used to do the anomaly detection
"""

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
# data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
# print('Loading data from', data_path)
settings["eval_single"] = False
settings["eval_an"] = False
samples, labels, index = data_utils.get_data(settings["data"], settings["seq_length"], settings["seq_step"],
                                             settings["num_signals"], settings["sub_id"], settings["eval_single"],
                                             settings["eval_an"])
# --- save settings, data --- #
# no need
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
# json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

class myADclass():
    def __init__(self, epoch, settings=settings, samples=samples, labels=labels, index=index):
        self.epoch = epoch
        self.settings = settings
        self.samples = samples
        self.labels = labels
        self.index = index
    def ADfunc(self):
        num_samples_t = self.samples.shape[0]
        t_size = 1156
        print('sample_shape:', self.samples.shape[0])
        print('num_samples_t', num_samples_t)

        T_index1 = np.asarray(list(range(2562,3140))) # test_normal
        T_index2 = np.asarray(list(range(6872,7450))) # test_anomaly
        T_index = np.concatenate((T_index1, T_index2))

        # -- only discriminate one batch for one time -- #
        D_test = np.empty([t_size, self.settings['seq_length'], 1])
        DL_test = np.empty([t_size, self.settings['seq_length'], 1])
        T_samples = np.empty([t_size, self.settings['seq_length'],self.settings['num_signals']])
        L_mb = np.empty([t_size, self.settings['seq_length'], 1])
        I_mb = np.empty([t_size, self.settings['seq_length'], 1])
        # batch_times = t_size // self.settings['batch_size']
        # for batch_idx in range(0, t_size // self.settings['batch_size']):
        for batch_idx in range(0, t_size):
            # print('batch_idx:{}
            # display batch progress
            model.display_batch_progression(batch_idx, t_size)
            # start_pos = batch_idx * self.settings['batch_size']
            # end_pos = start_pos + self.settings['batch_size']
            T_mb = self.samples[T_index[batch_idx], :, :]
            L_mmb = self.labels[T_index[batch_idx], :, :]
            I_mmb = self.index[T_index[batch_idx], :, :]
            para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(
                self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
            D_t, L_t = DR_discriminator.dis_D_model(self.settings, T_mb, para_path)
            T_samples[batch_idx,:,:] = T_mb
            D_test[batch_idx, :, :] = D_t
            DL_test[batch_idx, :, :] = L_t
            L_mb[batch_idx, :, :] = L_mmb
            I_mb[batch_idx, :, :] = I_mmb
        '''
        # start_pos = (num_samples_t // self.settings['batch_size']) * self.settings['batch_size']
        # end_pos = start_pos + self.settings['batch_size']
        # size = samples[start_pos:end_pos, :, :].shape[0]
        # fill = np.ones([self.settings['batch_size'] - size, samples.shape[1], samples.shape[2]])
        # batch = np.concatenate([samples[start_pos:end_pos, :, :], fill], axis=0)
        para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(
            self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
        D_t, L_t = DR_discriminator.dis_trained_model(self.settings, batch, para_path)
        L_mmb = self.labels[start_pos:end_pos, :, :]
        I_mmb = self.index[start_pos:end_pos, :, :]
        D_test[start_pos:end_pos, :, :] = D_t[:size, :, :]
        DL_test[start_pos:end_pos, :, :] = L_t[:size, :, :]
        L_mb[start_pos:end_pos, :, :] = L_mmb
        I_mb[start_pos:end_pos, :, :] = I_mmb
	'''
        results = np.zeros(5)
        tao = 0.7
        Accu2, Pre2, Rec2, F12, FPR2, D_L2 = DR_discriminator.detection_D_I(D_test, L_mb, I_mb, self.settings['seq_step'],tao)
        print('seq_length:',self.settings['seq_length'])
        print('Comb-statistic-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec:{:.4}; FPR: {:.4}'.format(self.epoch, tao, Accu2, Pre2, Rec2, FPR2))
        results = [Accu2, Pre2, Rec2, F12, FPR2]
	
        '''
        for i in range(2, 8):
            tao = 0.1 * i
            Accu2, Pre2, Rec2, F12 = DR_discriminator.detection_Comb(
                DL_test, L_mb, I_mb, self.settings['seq_step'], tao)
            print('seq_length:', self.settings['seq_length'])
            print('Comb-logits-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}'
                  .format(self.epoch, tao, Accu2, Pre2, Rec2, F12))
            results[i - 2, :] = [Accu2, Pre2, Rec2, F12]

            Accu3, Pre3, Rec3, F13 = DR_discriminator.detection_Comb(
                D_test, L_mb, I_mb, self.settings['seq_step'], tao)
            print('seq_length:', self.settings['seq_length'])
            print('Comb-statistic-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}'
                  .format(self.epoch, tao, Accu3, Pre3, Rec3, F13))
            results[i - 2+6, :] = [Accu3, Pre3, Rec3, F13]

            Accu5, Pre5, Rec5, F15 = DR_discriminator.sample_detection(D_test, L_mb, tao)
            print('seq_length:', self.settings['seq_length'])
            print('sample-wise-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}'
                  .format(self.epoch, tao, Accu5, Pre5, Rec5, F15))
            results[i - 2+12, :] = [Accu5, Pre5, Rec5, F15]
            '''
     
        return results

if __name__ == "__main__":
    print('Main Starting...')

    Results = np.zeros(5)

    epoch = 499 
    # for epoch in range(50, 60):
    ob = myADclass(epoch)
    Results = ob.ADfunc()

    # res_path = './experiments/plots/Results' + '_' + settings['sub_id'] + '_' + str(
    #     settings['seq_length']) + '.npy'
    # np.save(res_path, Results)

    print('Main Terminating...')
    end = time() - begin
    print('Testing terminated | Training time=%d s' % (end))
