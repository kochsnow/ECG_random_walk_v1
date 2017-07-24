#encoding:utf8
import os
import sys
import pickle
import matplotlib.pyplot as plt
import pdb
import time
import json
import random
import numpy as np
import scipy.io as sio
import json
import codecs
import glob
from random_walker import RandomWalker
# from QTdata.loadQTdata import QTloader
from feature_extractor.feature_extractor import ECGfeatures
from qrs_detection_jerry import qrs_detector
import scipy.signal
curfolder = os.path.split(os.path.realpath(__file__))[0]

def Test1(target_label = 'P', num_training = 25):
    '''Test case 1: random walk.'''
    qt = QTloader()
    record_list = qt.getreclist()
    training_list = random.sample(record_list, num_training)
    testing_list = list(set(record_list) - set(training_list))

    random_forest_config = dict(
            max_depth = 10)
    walker = RandomWalker(target_label = target_label,
            random_forest_config = random_forest_config)

    start_time = time.time()
    for record_name in training_list:
        print 'Collecting features from record %s.' % record_name
        sig = qt.load(record_name)
        walker.collect_training_data(sig['sig'], qt.getExpert(record_name))
    print 'random forest start training...'
    walker.training()
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    for record_name in testing_list:
        sig = qt.load(record_name)
        raw_sig = sig['sig']

        seed_position = random.randint(100, len(raw_sig) - 100)
        plt.figure(1)
        plt.clf()
        plt.plot(sig['sig'], label = record_name)
        plt.title(target_label)
        for ti in xrange(0, 20):
            seed_position += random.randint(1,200)
            print 'testing...(position: %d)' % seed_position
            start_time = time.time()
            results = walker.testing_walk(sig['sig'], seed_position, iterations = 100,
                    stepsize = 10)
            print 'testing finished in %.3f seconds.' % (time.time() - start_time)

            pos_list, values = zip(*results)
            predict_pos = np.mean(pos_list[len(pos_list) / 2:])
            
            # amp_list = [raw_sig[int(x)] for x in pos_list]
            amp_list = []
            bias = raw_sig[pos_list[0]]
            for pos in pos_list:
                amp_list.append(bias)
                bias -= 0.01

            plt.plot(predict_pos,
                    raw_sig[int(predict_pos)],
                    'ro',
                    markersize = 14,
                    label = 'predict position')
            plt.plot(pos_list, amp_list, 'r',
                    label = 'walk path',
                    markersize = 3,
                    linewidth = 8,
                    alpha = 0.3)
            plt.xlim(min(pos_list) - 100, max(pos_list) + 100)
            plt.grid(True)
            plt.legend()
            plt.show(block = False)
            pdb.set_trace()

def testing(random_walker, raw_sig, seed_step_size = 200):
    result_list = list()
    start_time = time.time()
    for seed_position in xrange(0, len(raw_sig), seed_step_size):
        sys.stdout.write('\rTesting: %06d samples left.' % (len(raw_sig) - 1 - seed_position))
        sys.stdout.flush()

        results = random_walker.testing_walk(raw_sig,
                seed_position,
                iterations = 100,
                stepsize = 10)
        pos_list, values = zip(*results)
        predict_pos = np.mean(pos_list[len(pos_list) / 2:])
        confidence = 1.0
        result_list.append((predict_pos, random_walker.target_label, confidence, pos_list))
    print 'testing finished in %.3f seconds.' % (time.time() - start_time)
    return result_list

# def TrainingModels(target_label, model_file_name, training_list):
#     '''Randomly select num_training records to train, and test others.'''
#     qt = QTloader()
#     record_list = qt.getreclist()
#     testing_list = list(set(record_list) - set(training_list))
#
#     random_forest_config = dict(
#             max_depth = 10)
#     walker = RandomWalker(target_label = target_label,
#             random_forest_config = random_forest_config,
#             random_pattern_file_name = os.path.join(os.path.dirname(model_file_name), 'random_pattern.json'))
#
#     start_time = time.time()
#     for record_name in training_list:
#         print 'Collecting features from record %s.' % record_name
#         sig = qt.load(record_name)
#         walker.collect_training_data(sig['sig'], qt.getExpert(record_name))
#     print 'random forest start training(%s)...' % target_label
#     walker.training()
#     print 'trianing used %.3f seconds' % (time.time() - start_time)
#
#     import joblib
#     start_time = time.time()
#     walker.save_model(model_file_name)
#     print 'Serializing model time cost %f' % (time.time() - start_time)

#train model for short_PR
def TrainingModels():
    # files=os.listdir(os.path.join(curfolder,'data'))
    # SIG = np.zeros(len(tranning_list))
    # KEYP = np.zeros(len(tranning_list))
    # for i in range(len(tranning_list)):
    #     x = tranning_list[i]
    #     datapath = os.path.join(curfolder, 'data', x, '.mat')
    #     keypointpath = os.path.join(curfolder, 'data', x, '.json')
    #     rawdata = sio.loadmat(datapath)
    #     rawsig = np.squeeze(rawdata['II'])
    #     with codecs.open(keypointpath, 'rb', encoding='utf-8') as fout:
    #         keypoint = json.load(fout)
    #     SIG[i] = rawsig
    #     KEYP[i] = keypoint
    for target_label in ['P', 'Ponset', 'Poffset', 'T', 'Tonset', 'Toffset', 'Ronset', 'Roffset']:
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder,'data','random_pattern.json'),fs=500.0)
        labelfolder=os.path.join(curfolder,'data','labels',target_label)
        sigfolder=os.path.join(curfolder,'data','sig')
        jsonfile=glob.glob(os.path.join(labelfolder,'*.json'))
        for file in jsonfile:
            with codecs.open(file,'rb',encoding='utf-8') as fin:
                 data=json.load(fin)
            annot_list=data['poslist']
            expannot_list=zip(annot_list,len(annot_list)*[target_label])
            ID=os.path.splitext(os.path.split(file)[-1])[0]
            matpath=os.path.join(sigfolder,str(ID)+'.mat')
            rawdata=sio.loadmat(matpath)
            rawsig=np.squeeze(rawdata['II'])
            walker.collect_training_data(rawsig, expannot_list)
        walker.training()
        walker.save_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
#test model
def test():
    Fs = 500
    allIDpath=glob.glob(os.path.join(curfolder,'data','sig','*.mat'))
    allID=[os.path.splitext(os.path.split(file)[-1])[0] for file in allIDpath]
    trainIDpath=glob.glob(os.path.join(curfolder,'data','labels','*.json'))
    trainID = [os.path.splitext(os.path.split(file)[-1])[0] for file in trainIDpath]
    test_list=list(set(allID)-set(trainID))
    for j in range(min(5,len(test_list))):
        x = test_list[j]
        datapath = os.path.join(curfolder, 'data','sig', str(x)+'.mat')
        rawdata = sio.loadmat(datapath)
        rawsig = np.squeeze(rawdata['II'])
        re_rawsig = scipy.signal.resample_poly(rawsig, 1, int(Fs / 250.0))
        r_list = [int(x * Fs / 250.0) for x in qrs_detector(re_rawsig, fs=250)]

        results=list()
        target_label = 'Ronset'
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(1,len(r_list)):
            seed_position=r_list[i]
            confined_range=[r_list[i-1],r_list[i]]
            temp=walker.testing_walk_extractor(feature_extractor=feature_extractor,seed_position=seed_position,iterations=100,stepsize=4,
                                      confined_range=confined_range)
            pos,value=zip(*temp)
            results.append([int(np.mean(pos[len(pos)/2:])),target_label])

        target_label = 'Roffset'
        Rofflist=list()
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(0, len(r_list)-1):
            seed_position = r_list[i]
            confined_range = [r_list[i], r_list[i+1]]
            temp = walker.testing_walk_extractor(feature_extractor=feature_extractor, seed_position=seed_position,
                                                 iterations=100, stepsize=4,
                                                 confined_range=confined_range)
            pos, value = zip(*temp)
            Rofflist.append([int(np.mean(pos[len(pos) / 2:])), target_label])
            results.append([int(np.mean(pos[len(pos) / 2:])),target_label])

        target_label = 'P'
        Plist=list()
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(1, len(r_list)):
            seed_position = r_list[i]
            confined_range = [r_list[i - 1], r_list[i]]
            temp = walker.testing_walk_extractor(feature_extractor=feature_extractor, seed_position=seed_position,
                                                 iterations=200, stepsize=4,
                                                 confined_range=confined_range)
            pos, value = zip(*temp)
            Plist.append([int(np.mean(pos[len(pos) / 2:])),target_label])
            results.append([int(np.mean(pos[len(pos) / 2:])),target_label])

        target_label = 'Ponset'
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(1, len(r_list)):
            seed_position = Plist[i-1]
            confined_range = [r_list[i - 1], Plist[i-1][0]]
            temp = walker.testing_walk_extractor(feature_extractor=feature_extractor, seed_position=seed_position,
                                                 iterations=200, stepsize=4,
                                                 confined_range=confined_range)
            pos, value = zip(*temp)
            results.append([int(np.mean(pos[len(pos) / 2:])), target_label])

        target_label = 'Poffset'
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(1, len(r_list)):
            seed_position = r_list[i]
            confined_range = [Plist[i-1][0],r_list[i]]
            temp = walker.testing_walk_extractor(feature_extractor=feature_extractor, seed_position=seed_position,
                                                 iterations=200, stepsize=4,
                                                 confined_range=confined_range)
            pos, value = zip(*temp)
            results.append([int(np.mean(pos[len(pos) / 2:])), target_label])

        target_label = 'T'
        Tlist = list()
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(0, len(r_list)-1):
            seed_position = Rofflist[i][0]
            confined_range = [Rofflist[i][0], r_list[i+1]]
            temp = walker.testing_walk_extractor(feature_extractor=feature_extractor, seed_position=seed_position,
                                                 iterations=200, stepsize=4,
                                                 confined_range=confined_range)
            pos, value = zip(*temp)
            Tlist.append([int(np.mean(pos[len(pos) / 2:])), target_label])
            results.append([int(np.mean(pos[len(pos) / 2:])), target_label])

        target_label = 'Tonset'
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(0, len(r_list)-1):
            seed_position = Tlist[i]
            confined_range = [r_list[i], Tlist[i][0]]
            temp = walker.testing_walk_extractor(feature_extractor=feature_extractor, seed_position=seed_position,
                                                 iterations=100, stepsize=4,
                                                 confined_range=confined_range)
            pos, value = zip(*temp)
            results.append([int(np.mean(pos[len(pos) / 2:])), target_label])

        target_label = 'Toffset'
        walker = RandomWalker(target_label=target_label, random_forest_config=dict(),
                              random_pattern_file_name=os.path.join(curfolder, 'data', 'random_pattern.json'), fs=500.0)
        walker.load_model(model_file_name=os.path.join(curfolder, 'data', 'models', target_label+'.mdl'))
        configuration_info = walker.get_configuration()
        feature_extractor = ECGfeatures(rawsig=rawsig, configuration_info=configuration_info)
        for i in range(0, len(r_list) - 1):
            seed_position = Tlist[i][0]
            confined_range = [Tlist[i][0],r_list[i+1]]
            temp = walker.testing_walk_extractor(feature_extractor=feature_extractor, seed_position=seed_position,
                                                 iterations=100, stepsize=4,
                                                 confined_range=confined_range)
            pos, value = zip(*temp)
            results.append([int(np.mean(pos[len(pos) / 2:])), target_label])

        results.extend(zip(r_list,len(r_list)*['R']))
        plt.figure(1)
        plt.plot(rawsig, label='ECG')
        pos_list, label_list = zip(*results)
        labels = set(label_list)
        labels=list(labels)
        convert = {'R': 'ro', 'Ronset': 'r<', 'Roffset': 'r>',
                   'P': 'bo', 'Ponset': 'b<', 'Poffset': 'b>',
                   'T': 'go', 'Tonset': 'g<', 'Toffset': 'g>'}
        for label in list(labels):
            pos_list = [int(x[0]) for x in results if x[1] == label]
            amp_list = [rawsig[x] for x in pos_list]
            plt.plot(pos_list, amp_list, convert[label],
                     markersize=6,
                     label=label)
        print j
        plt.grid(True)
        plt.legend()
        plt.show()

        # configuration_info=walker.get_configuration()
        # feature_extractor=ECGfeatures(rawsig,configuration_info=configuration_info,wavelet='db2')



if __name__=='__main__':
    # TrainingModels()
    test()
#
# if __name__ == '__main__':
#     label_list = ['P', 'Ponset', 'Poffset',
#             'T', 'Toffset',
#             'Ronset', 'R', 'Roffset']
#     root_folder = 'data/db2WT'
#     # Refresh training list
#     num_training = 105
#     trianing_list = list()
#     qt = QTloader()
#     record_list = qt.getreclist()
#     must_train_list = [
#         "sel35",
#         "sel36",
#         "sel31",
#         "sel38",
#         "sel39",
#         "sel820",
#         "sel51",
#         "sele0104",
#         "sele0107",
#         "sel223",
#         "sele0607",
#         "sel102",
#         "sele0409",
#         "sel41",
#         "sel40",
#         "sel43",
#         "sel42",
#         "sel45",
#         "sel48",
#         "sele0133",
#         "sele0116",
#         "sel14172",
#         "sele0111",
#         "sel213",
#         "sel14157",
#         "sel301"
#             ]
#     num_training -= len(must_train_list)
#     record_list = list(set(record_list) - set(must_train_list))
#     training_list = must_train_list
#     if num_training > 0:
#         training_list.extend(random.sample(record_list, num_training))
#     # Save training list
#     with open(os.path.join(root_folder, 'training_list.json'), 'w') as fout:
#         json.dump(training_list, fout, indent = 4)
#     for target_label in label_list:
#         model_file_name = os.path.join(root_folder, '%s.mdl' % target_label)
#         TrainingModels(target_label, model_file_name, training_list)
