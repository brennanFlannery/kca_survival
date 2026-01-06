from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn
import os
import time
import logging
import numpy as np
import configparser
import h5py
import torch
from torch.utils.data import Dataset
import torchio as tio
from lifelines.utils import concordance_index
import random
import torch.nn.functional as F
#from pytorch_grad_cam.base_cam import BaseCAM
from typing import Tuple, List
from scipy.ndimage import zoom
import pandas as pd
import glob
from SimpleITK import ReadImage
import SimpleITK as sitk

def read_config(ini_file):
    ''' Performs read config file and parses it.
    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index
    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)
    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def create_logger(logs_dir):
    ''' Performs creating logger
    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    # logs settings
    log_file = os.path.join(logs_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # initialize console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger

class SurvivalDatasetImg(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train, transform=None, is_vol = False):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.e, self.y = \
            self._read_h5_file(h5_file, is_train)
        self.transform = transform
        self.is_vol = is_vol
        self.num_events = np.sum(self.e)
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        return X, e, y

    def _normalize(self):
        ''' Performs normalizing X data. '''
        # self.X = (self.X-self.X.min(axis=0)) / \
        #     (self.X.max(axis=0)-self.X.min(axis=0))
        self.X = (self.X-self.X.min()) / \
            (self.X.max()-self.X.min())

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item].astype(np.float32) # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        if self.is_vol:
            if len(X_item.shape) > 3:
                X_item = X_item[:,(128-25):(128+25), :,:] # -25,+25
                X_item = X_item[:,:, np.linspace(0, 99,50, dtype=np.uint16),:]
                X_item = X_item[:,:, :,np.linspace(0, 99,50, dtype=np.uint16)]
            else:
                # X_item = X_item[np.linspace(0,255, 10, dtype=np.uint16), :,:] # was 128
                X_item = X_item[(50-25):(50+25), :,:] # -25,+25
                X_item = X_item[:, np.linspace(0, 99,50, dtype=np.uint16),:]
                X_item = X_item[:, :,np.linspace(0, 99,50, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            if len(X_item.shape) > 3:
                X_tensor = torch.Tensor(self.transform(X_item))
            else:
                X_tensor = torch.Tensor(self.transform(X_item[None, ::]))
            # if self.is_vol:
            #     X_tensor = X_tensor.permute(1,0,2)
        else:
            X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)


        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]

class SurvivalDatasetImgFromFolder(Dataset):
    def __init__(self, clinical_path, phase, transform, surv_analysis = False, crop=True):
        self.crop = crop
        # loads data
        self.clinical = self._read_clinical_file(clinical_path)
        self.clinical = self.clinical[self.clinical["institution"]=="Uminn"]
        self.surv_analysis = surv_analysis
        if phase == "train":
            self.clinical = self.clinical[self.clinical.split == 0]
            self.clinical = self.clinical.sample(frac=1, random_state=42).reset_index(drop=True)
            self.clinical = self.clinical.head(int(len(self.clinical)*0.7))
        elif phase == "test":
            self.clinical = self.clinical[self.clinical.split == 0]
            self.clinical = self.clinical.sample(frac=1, random_state=42).reset_index(drop=True)
            self.clinical = self.clinical.tail(int(len(self.clinical)*0.3))
        elif phase == "exval":
            self.clinical = self.clinical[self.clinical.split == 1]

        self.transform = transform
        self.num_events = np.sum(self.clinical["vital_status"])
        self.e = [(x,x) for x in self.clinical.vital_status.to_numpy()]
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.clinical.shape[0]))

    def _read_clinical_file(self, csv_path):
        clinical_df = pd.read_csv(csv_path)
        clinical_df = clinical_df.set_index("patient_id")
        return clinical_df

    def _normalize(self):
        ''' Performs normalizing X data. '''
        # self.X = (self.X-self.X.min(axis=0)) / \
        #     (self.X.max(axis=0)-self.X.min(axis=0))
        self.X = (self.X-self.X.min()) / \
            (self.X.max()-self.X.min())

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        patient = self.clinical.iloc[item]
        x_item = sitk.GetArrayFromImage(ReadImage(patient["file"]))
        if self.crop:
            cen = np.array(np.shape(x_item))//2
            patch_size = 100
            cen_new = (np.random.randint(cen[0]-(patch_size//2), cen[0]+(patch_size//2)),
                   np.random.randint(cen[1]-patch_size, cen[1]+patch_size),
                   np.random.randint(cen[2]-patch_size, cen[2]+patch_size))
            x_item = x_item[(cen_new[0] - (patch_size//2)):(cen_new[0] + (patch_size//2)),
                     (cen_new[1] - (patch_size//2)):(cen_new[1] + (patch_size//2)),
                     (cen_new[2] - (patch_size//2)):(cen_new[2] + (patch_size//2))]
        # x_item = x_item[:, (cen - 164):(cen + 164), (cen - 194):(cen + 194)]
        # random_side = np.random.randint(low=0, high=2)
        # random_topbot = np.random.randint(low=0, high=2)
        # x_item = x_item[:, (random_topbot * np.shape(x_item)[1]//2):((random_topbot * np.shape(x_item)[1]//2) + np.shape(x_item)[1]//2),
        #          (random_side * np.shape(x_item)[2]//2):((random_side * np.shape(x_item)[2]//2) + np.shape(x_item)[2]//2)]

        # x_item = x_item[np.linspace(0, 255, 128, dtype=np.uint16), ::]
        # x_item = x_item[:,np.linspace(0, 447, 224, dtype=np.uint16),:]
        # x_item = x_item[:,:,np.linspace(0, 447, 224, dtype=np.uint16)]
        x_item = x_item[None, ::]
        # constructs torch.Tensor object
        if self.transform:
            X_tensor = torch.Tensor(self.transform(x_item))
        else:
            X_tensor = torch.from_numpy(x_item)
        e_tensor = torch.as_tensor(patient["vital_status"])
        y_tensor = torch.as_tensor(patient["vital_days_after_surgery"])

        if patient.institution == "Uminn":
            pname = patient.file.split("/")[-1].split(".")[0]
        else:
            pname = patient["file"].split("\\")[-1].split("_")[0]
        if self.surv_analysis:
            return X_tensor, y_tensor, e_tensor, pname
        else:
            return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.clinical.shape[0]

class SurvivalDatasetImgClinicalSSL(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train=True, transform=False, is_vol = False, *extra):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.e, self.y, self.names = \
            self._read_h5_file(h5_file, is_train)
        self.transform = transform
        self.num_events = np.sum(self.e)
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test' #CHANGE THIS BACK ONCE YOU FIX THE DATA!!!!!!!!!!!
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
            names = f[split]['name'][()]

        return X, e, y, names

    def _normalize(self):
        ''' Performs normalizing X data. '''
        # self.X = (self.X-self.X.min(axis=0)) / \
        #     (self.X.max(axis=0)-self.X.min(axis=0))
        self.X = (self.X-self.X.min()) / \
            (self.X.max()-self.X.min())

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        pname = self.names[item]

        if self.transform:
            X_item = X_item + (X_item * .1 * (2*np.random.rand(len(X_item)) - 1))

        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)


        return X_tensor, y_tensor, e_tensor, pname

    def __len__(self):
        return self.X.shape[0]

class ClassificationSSLDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, config, phase = "train", transform=None, *extra):
        # loads data
        self.outcome = config["outcome"]
        self.lefts, self.rights, self.ssl, self.outcomes, self.names = \
            self._read_h5_file(h5_file, phase)
        self.transform = transform
        self.modeltype = config["modeltype"]
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.lefts.shape[0]))

    def _read_h5_file(self, h5_file, split):
        with h5py.File(h5_file, 'r') as f:
            lefts = f[split]['left'][()]
            rights = f[split]['right'][()]
            ssl = f[split]['ssl'][()]
            outcomes = f[split][self.outcome][()].reshape(-1, 1)
            names = f[split]['name'][()]

        # Remove NAs from dataset
        remove_loc = [x[0] for x in np.argwhere(np.isnan(outcomes))]
        lefts = np.delete(lefts, remove_loc, axis=0)
        rights = np.delete(rights, remove_loc, axis=0)
        ssl = np.delete(ssl, remove_loc, axis=0)
        outcomes = np.delete(outcomes, remove_loc, axis=0)
        names = np.delete(names, remove_loc, axis=0)

        return lefts, rights, ssl, outcomes, names

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        if self.modeltype == "SSL_Pre":
            X_item = self.ssl[item]
        else:
            X_item = np.concatenate((self.lefts[item], self.rights[item]), axis=2)
            X_item = X_item[None, ::]
        if self.transform:
            X_item = self.transform(X_item)
        if self.modeltype in ["SSL_Full", "SSL_Retrain"]:
            X_item = X_item[:,np.random.randint(0, np.shape(X_item)[1]), ::]
            X_item = np.stack((X_item, X_item, X_item)).transpose(1,0, 2, 3)
            X_item = X_item[0,::]

        y_item = self.outcomes[item]
        pname = self.names[item]

        X_tensor = torch.from_numpy(X_item)
        y_tensor = torch.from_numpy(y_item)

        return X_tensor, y_tensor, pname

    def __len__(self):
        return self.lefts.shape[0]

class SurvivalDatasetImgClinical(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, split, transform=None, is_vol = False, random_split=False):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.e, self.y, self.names = \
            self._read_h5_file(h5_file, split)
        self.transform = transform
        self.is_vol = is_vol
        self.num_events = np.sum(self.e)
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, split):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        # split = 'train' if is_train else 'test' #CHANGE THIS BACK ONCE YOU FIX THE DATA!!!!!!!!!!!
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
            names = f[split]['name'][()]

        return X, e, y, names

    def _normalize(self):
        ''' Performs normalizing X data. '''
        # self.X = (self.X-self.X.min(axis=0)) / \
        #     (self.X.max(axis=0)-self.X.min(axis=0))
        self.X = (self.X-self.X.min()) / \
            (self.X.max()-self.X.min())

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item].astype(np.float32)
        if np.mean(X_item) < -800:
            item = item + 1
            X_item = self.X[item].astype(np.float32)
            # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        pname = self.names[item]
        if self.is_vol:
            if len(X_item.shape) > 3:
                X_item = X_item[:,(50-25):(50+25), :,:] # -25,+25
                X_item = X_item[:,:, np.linspace(0, 99,50, dtype=np.uint16),:]
                X_item = X_item[:,:, :,np.linspace(0, 99,50, dtype=np.uint16)]
            else:
                # X_item = X_item[np.linspace(0,255, 10, dtype=np.uint16), :,:] # was 128
                X_item = X_item[(50-25):(50+25), :,:] # -25,+25
                X_item = X_item[:, np.linspace(0, 99,50, dtype=np.uint16),:]
                X_item = X_item[:, :,np.linspace(0, 99,50, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            X_tensor = self.transform(X_item)[None, ::]
        else:
            X_tensor = torch.from_numpy(X_item)[None, ::]
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)

        return X_tensor, y_tensor, e_tensor, pname

    def __len__(self):
        return self.X.shape[0]
        

class SurvivalDatasetImgClinicalRadPre(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, split, transform=None, is_vol=False, random_split=False, few_shot=None):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        :param few_shot: (float or None) fraction of data to use for few-shot learning, should be between 0 and 1.
        '''
        if few_shot is not None and (few_shot <= 0 or few_shot > 1):
            raise ValueError("few_shot must be a decimal between 0 (exclusive) and 1 (inclusive).")
        
        # loads data
        self.X, self.e, self.y, self.names = \
            self._read_h5_file(h5_file, split)
        
        if few_shot is not None:
            event_indices = np.where(self.e.flatten() == 1)[0]
            non_event_indices = np.where(self.e.flatten() == 0)[0]
            
            few_shot_event_samples = int(len(event_indices) * few_shot)
            few_shot_non_event_samples = int(len(non_event_indices) * few_shot)
            
            selected_event_indices = np.random.choice(event_indices, few_shot_event_samples, replace=False)
            selected_non_event_indices = np.random.choice(non_event_indices, few_shot_non_event_samples, replace=False)
            
            selected_indices = np.concatenate([selected_event_indices, selected_non_event_indices])
            
            self.X = self.X[selected_indices]
            self.e = self.e[selected_indices]
            self.y = self.y[selected_indices]
            self.names = self.names[selected_indices]
        
        self.transform = transform
        self.is_vol = is_vol
        self.num_events = np.sum(self.e)
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, split):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
            names = f[split]['name'][()]

        return X, e, y, names

    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        X_item = self.X[item].astype(np.float32)
        orig = item
        if np.mean(X_item) < -800:
            try:
                item = orig + 1
                X_item = self.X[item].astype(np.float32)
            except:
                item = orig - 1
                X_item = self.X[item].astype(np.float32)
        e_item = self.e[item]  # (1)
        y_item = self.y[item]  # (1)
        pname = self.names[item]
        # constructs torch.Tensor object
        if self.transform:
            X_tensor = self.transform(X_item)[None, ::]
        else:
            X_tensor = torch.from_numpy(X_item)[None, ::]
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)

        return X_tensor, y_tensor, e_tensor, pname

    def __len__(self):
        return self.X.shape[0]

class SurvivalDatasetImgClinicalGradCam(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, shap=False):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.e, self.y, self.segs, self.names = \
            self._read_h5_file(h5_file)
        self.num_events = np.sum(self.e)
        self.shap = shap
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
            seg = f[split]['seg'][()]
            names = f[split]['name'][()]

        return X, e, y, seg, names

    def _normalize(self):
        ''' Performs normalizing X data. '''
        # self.X = (self.X-self.X.min(axis=0)) / \
        #     (self.X.max(axis=0)-self.X.min(axis=0))
        self.X = (self.X-self.X.min()) / \
            (self.X.max()-self.X.min())

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item].astype(np.float32) # (m)
        seg_item = self.segs[item].astype(np.float32)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        pname = self.names[item]

        X_item = X_item[(128-35):(128+35), :,:] # -25,+25
        seg_item = seg_item[:, (128-35):(128+35), :,:]
        # seg_item = np.vstack((seg_item[None, (128-25):(128+25), :,:], seg_item[None, (128-25):(128+25), :,:])) # -25,+25
        X_orig = X_item.copy()
        X_item = X_item[:, np.linspace(0, 99,50, dtype=np.uint16),:]
        X_item = X_item[:, :,np.linspace(0, 99,50, dtype=np.uint16)]
        # constructs torch.Tensor object

        X_tensor = torch.from_numpy(X_item)
        seg_tensor = torch.from_numpy(seg_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)

        if self.shap:
            # X_tensor = X_tensor[(128-25):(128+25), :,:] # -25,+25
            return X_tensor
        else:
            return X_orig, X_tensor, y_tensor, e_tensor, seg_tensor, pname

    def __len__(self):
        return self.X.shape[0]

class SurvivalDatasetImg2(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train, img_transform=None, is_vol = False):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        # self.X, self.e, self.y = \
        #     self._read_h5_file(h5_file, is_train)
        self.train = is_train
        self.file = h5_file
        self.transform = img_transform
        self.is_vol = is_vol
        split = 'train' if is_train else 'test' #CHANGE THIS BACK ONCE YOU FIX THE DATA!!!!!!!!!!!
        print(split)
        with h5py.File(h5_file, 'r') as f:
            self.length = f[split]['x'].shape[0]
        phase = "Train" if is_train else "Validation"
        print(f"Loaded {phase} Dataset")

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test' #CHANGE THIS BACK ONCE YOU FIX THE DATA!!!!!!!!!!!
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        return X, e, y

    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        split = 'train' if self.train else 'test' #CHANGE THIS BACK ONCE YOU FIX THE DATA!!!!!!!!!!!
        with h5py.File(self.file, 'r') as f:
            X_item = f[split]['x'][item]
            e_item = [f[split]['e'][item]]
            y_item = [f[split]['t'][item]]
        # gets data with index of item
        # X_item = self.X[item].astype(np.float32) # (m)
        # e_item = self.e[item] # (1)
        # y_item = self.y[item] # (1)
        if self.is_vol:
            X_item = X_item[np.linspace(0,255, 10, dtype=np.uint16), :,:]
            X_item = X_item[:, np.linspace(0, 99,50, dtype=np.uint16),:]
            X_item = X_item[:, :,np.linspace(0, 99,50, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            X_tensor = self.transform(image=X_item)['image']
            if self.is_vol:
                X_tensor = X_tensor.permute(1,0,2)
        else:
            X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.Tensor(e_item)
        y_tensor = torch.Tensor(y_item)


        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.length


class SurvivalDatasetImg2DualBranches(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''

    def __init__(self, int_h5_file, ext_h5_file, is_train, img_transform=None, is_vol=False):
        # loads data
        # self.X, self.e, self.y = \
        #     self._read_h5_file(h5_file, is_train)
        self.train = is_train
        self.intfile = int_h5_file
        self.extfile = ext_h5_file
        self.transform = img_transform
        self.is_vol = is_vol
        split = 'train' if is_train else 'test'
        print(split)
        with h5py.File(ext_h5_file, 'r') as f:
            self.length = f[split]['x'].shape[0]
        phase = "Train" if is_train else "Validation"
        print(f"Loaded {phase} Dataset")

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        split = 'train' if self.train else 'test'
        with h5py.File(self.intfile, 'r') as f:
            X1_item = f[split]['x'][item]
            e_item = [f[split]['e'][item]]
            y_item = [f[split]['t'][item]]
        with h5py.File(self.extfile, 'r') as f:
            X2_item = f[split]['x'][item]

        if self.is_vol:
            X1_item = X1_item[np.linspace(0, 255, 10, dtype=np.uint16), :, :]
            X1_item = X1_item[:, np.linspace(0, 99, 50, dtype=np.uint16), :]
            X1_item = X1_item[:, :, np.linspace(0, 99, 50, dtype=np.uint16)]
            X2_item = X2_item[np.linspace(0, 255, 10, dtype=np.uint16), :, :]
            X2_item = X2_item[:, np.linspace(0, 99, 50, dtype=np.uint16), :]
            X2_item = X2_item[:, :, np.linspace(0, 99, 50, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            X1_tensor = self.transform(image=X1_item)['image']
            X2_tensor = self.transform(image=X2_item)['image']
            if self.is_vol:
                X1_tensor = X1_tensor.permute(1, 0, 2)
                X2_tensor = X2_tensor.permute(1, 0, 2)
        else:
            X1_tensor = torch.from_numpy(X1_item)
            X2_tensor = torch.from_numpy(X1_item)

        e_tensor = torch.Tensor(e_item)
        y_tensor = torch.Tensor(y_item)

        return (X1_tensor, X2_tensor), y_tensor, e_tensor

class SurvivalDatasetImgDualBranches(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, int_h5_file, ext_h5_file, is_train, img_transform=None, is_vol = False):
        # loads data
        # self.X, self.e, self.y = \
        #     self._read_h5_file(h5_file, is_train)
        self.train = is_train
        self.intfile = int_h5_file
        self.extfile = ext_h5_file
        self.transform = img_transform
        self.is_vol = is_vol
        split = 'train' if is_train else 'test'
        # loads data
        self.X, self.e, self.y = \
            self._read_h5_file(self.intfile, self.extfile, is_train)
        self.transform = img_transform
        self.is_vol = is_vol
        # normalizes data
        # self._normalize()

        phase = "Train" if is_train else "Validation"
        print(f"Loaded {phase} Dataset")

    def _read_h5_file(self, int_file, ext_file, is_train):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test' #CHANGE THIS BACK ONCE YOU FIX THE DATA!!!!!!!!!!!
        with h5py.File(int_file, 'r') as f:
            X1 = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        with h5py.File(ext_file, 'r') as f:
            X2 = f[split]['x'][()]
            e2 = f[split]['e'][()].reshape(-1, 1)
            y2 = f[split]['t'][()].reshape(-1, 1)
        return (X1,X2), e, y

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        split = 'train' if self.train else 'test'
        # gets data with index of item
        X1_item = self.X[0][item].astype(np.float32)
        X2_item = self.X[1][item].astype(np.float32)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        if self.is_vol:
            X1_item = X1_item[np.linspace(0,255, 10, dtype=np.uint16), :,:]
            X1_item = X1_item[:, np.linspace(0, 99,50, dtype=np.uint16),:]
            X1_item = X1_item[:, :,np.linspace(0, 99,50, dtype=np.uint16)]
            X2_item = X2_item[np.linspace(0,255, 10, dtype=np.uint16), :,:]
            X2_item = X2_item[:, np.linspace(0, 99,50, dtype=np.uint16),:]
            X2_item = X2_item[:, :,np.linspace(0, 99,50, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            X1_tensor = self.transform(image=X1_item)['image']
            X2_tensor = self.transform(image=X2_item)['image']
            if self.is_vol:
                X1_tensor = X1_tensor.permute(1,0,2)
                X2_tensor = X2_tensor.permute(1,0,2)
        else:
            X1_tensor = torch.from_numpy(X1_item)
            X2_tensor = torch.from_numpy(X1_item)

        e_tensor = torch.Tensor(e_item)
        y_tensor = torch.Tensor(y_item)


        return (X1_tensor, X2_tensor), y_tensor, e_tensor

    def __len__(self):
        return self.X[0].shape[0]



class SegmentationDataset(object):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train, img_transform=None, reduce=False, crop = False, tumor = False):
        self.train = is_train
        self.file = h5_file
        self.transform = img_transform
        self.reduce = reduce
        self.crop = crop
        split = 'train' if is_train else 'test'
        print(split)
        with h5py.File(h5_file, 'r') as f:
            self.length = f['v'].shape[0]
        phase = "Train" if is_train else "Test"
        print(f"Loaded {phase} Dataset")
        self.tumor = tumor
        if tumor and tumor != "Both":
            checked = []
            with h5py.File(self.file, 'r') as f:
                for i in range(self.length):
                    if np.any(f['t'][i][:,:,:]):
                        checked.append(i)
            self.checked = checked
            print(f"Checked {self.length} items, {len(self.checked)} items with tumor")
    def __getitem__(self, item):
        if self.tumor and self.tumor != "Both":
            if item not in self.checked:
                item = random.choice(self.checked)
        split = 'train' if self.train else 'test'
        with h5py.File(self.file, 'r') as f:
            v_item = f['v'][item][None,:,:,:]
            if self.tumor and self.tumor != "Both":
                label_item = f['t'][item][None,:,:,:]
            elif self.tumor == "Both":
                label_item = f['t'][item][None,:,:,:] + f['k'][item][None, :, :, :]
            else:
                label_item = f['k'][item][None, :, :, :]

        if self.reduce:
            v_item = v_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            v_item = v_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            v_item = v_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

            label_item = label_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            label_item = label_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            label_item = label_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

        v_tensor = torch.Tensor(v_item)
        label_tensor = torch.Tensor(label_item)


        if self.crop:
            thickness = 10
            hthickness = thickness // 2
            temp = torch.where(torch.squeeze(torch.sum(label_tensor, axis=(2,3))))[0].numpy()

            temp = np.array([x for x in temp if x > hthickness])
            temp = np.array([x for x in temp if x < (np.shape(label_tensor)[1] - hthickness)])
            if np.any(temp) == 0:
                rng = np.shape(label_tensor)[1] // 2
            else:
                rng = random.choice(temp)
            # rng = random.randint(hthickness, 16 - hthickness)
            # hsize = 384//2
            # rr = random.randint(hsize, 448 - hsize)
            # rc = random.randint(hsize, 448 - hsize)
            # v_tensor = v_tensor[:,(rng-hthickness):(rng+hthickness), (rr-hsize):(rr+hsize), (rc-hsize):(rc+hsize)]
            # k_tensor = k_tensor[:,(rng-hthickness):(rng+hthickness), (rr-hsize):(rr+hsize), (rc-hsize):(rc+hsize)]
            # t_tensor = t_tensor[:,(rng-hthickness):(rng+hthickness), (rr-hsize):(rr+hsize), (rc-hsize):(rc+hsize)]
            v_tensor = v_tensor[:,(rng-hthickness):(rng+hthickness), ::]
            label_tensor = label_tensor[:,(rng-hthickness):(rng+hthickness), ::]


        # constructs torch.Tensor object
        if self.transform:
            subject = tio.Subject(
                image = tio.ScalarImage(tensor=v_tensor),
                # label = tio.LabelMap(tensor=t_item>0)
            )
            v_trans = self.transform(subject)
            v_tensor = v_trans["image"].data
            yes_trans = ['Flip', 'Affine', "Crop"]
            trans = tio.Compose([x for x in v_trans.get_composed_history() if x.name in yes_trans])
            label_tensor = torch.Tensor((trans(label_tensor) > .5) * 1)
            v_tensor = v_trans["image"].tensor
            # t_tensor = v_trans["label"].tensor

        return v_tensor, label_tensor

    def __len__(self):
        return self.length


class SegmentationDatasetMulticlass(object):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train, img_transform=None, reduce=False, crop = False):
        self.train = is_train
        self.file = h5_file
        self.transform = img_transform
        self.reduce = reduce
        self.crop = crop
        split = 'train' if is_train else 'test'
        print(split)
        with h5py.File(h5_file, 'r') as f:
            self.length = f['v'].shape[0]
        phase = "Train" if is_train else "Test"
        print(f"Loaded {phase} Dataset")

    def __getitem__(self, item):
        split = 'train' if self.train else 'test'
        with h5py.File(self.file, 'r') as f:
            v_item = f['v'][item][None,:,:,:]
            t_item = f['t'][item][None,:,:,:]
            k_item = f['k'][item][None,:,:,:]


        if self.reduce:
            v_item = v_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            v_item = v_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            v_item = v_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

            t_item = t_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            t_item = t_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            t_item = t_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

            k_item = k_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            k_item = k_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            k_item = k_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

        v_tensor = torch.Tensor(v_item)
        k_tensor = torch.Tensor(k_item)
        t_tensor = torch.Tensor(t_item)

        if self.crop:
            thickness = 10
            hthickness = thickness // 2
            temp = torch.where(torch.squeeze(torch.sum(t_tensor, axis=(2,3))))[0].numpy()

            temp = np.array([x for x in temp if x > hthickness])
            temp = np.array([x for x in temp if x < (np.shape(t_tensor)[1] - hthickness)])
            if np.any(temp) == 0:
                rng = np.shape(t_tensor)[1] // 2
            else:
                rng = random.choice(temp)
            # rng = random.randint(hthickness, 16 - hthickness)
            # hsize = 384//2
            # rr = random.randint(hsize, 448 - hsize)
            # rc = random.randint(hsize, 448 - hsize)
            # v_tensor = v_tensor[:,(rng-hthickness):(rng+hthickness), (rr-hsize):(rr+hsize), (rc-hsize):(rc+hsize)]
            # k_tensor = k_tensor[:,(rng-hthickness):(rng+hthickness), (rr-hsize):(rr+hsize), (rc-hsize):(rc+hsize)]
            # t_tensor = t_tensor[:,(rng-hthickness):(rng+hthickness), (rr-hsize):(rr+hsize), (rc-hsize):(rc+hsize)]
            v_tensor = v_tensor[:,(rng-hthickness):(rng+hthickness), ::]
            t_tensor = t_tensor[:,(rng-hthickness):(rng+hthickness), ::]
            k_tensor = k_tensor[:,(rng-hthickness):(rng+hthickness), ::]


        # constructs torch.Tensor object
        if self.transform:
            subject = tio.Subject(
                image = tio.ScalarImage(tensor=v_tensor),
                # label = tio.LabelMap(tensor=t_item>0)
            )
            v_trans = self.transform(subject)
            v_tensor = v_trans["image"].data
            yes_trans = ['Flip', 'Affine', "Crop"]
            trans = tio.Compose([x for x in v_trans.get_composed_history() if x.name in yes_trans])
            t_tensor = torch.Tensor((trans(t_tensor) > .5) * 1)
            k_tensor = torch.Tensor((trans(k_tensor) > .5) * 1)
            v_tensor = v_trans["image"].tensor
            # t_tensor = v_trans["label"].tensor

        return v_tensor, torch.cat((k_tensor, t_tensor), dim=0)

    def __len__(self):
        return self.length

class SegmentationDatasetMulticlass_softmax(SegmentationDatasetMulticlass):
    def __init__(self, h5_file, is_train, img_transform=None, reduce=False, crop = False):
        super().__init__(h5_file, is_train, img_transform, reduce, crop)
    def __getitem__(self, item):
        split = 'train' if self.train else 'test'
        with h5py.File(self.file, 'r') as f:
            v_item = f['v'][item][None,:,:,:]
            t_item = f['t'][item][None,:,:,:]
            k_item = f['k'][item][None,:,:,:]


        if self.reduce:
            v_item = v_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            v_item = v_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            v_item = v_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

            t_item = t_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            t_item = t_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            t_item = t_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

            k_item = k_item[:,np.linspace(0,255, 50, dtype=np.uint16), :,:]
            k_item = k_item[:,:, np.linspace(0, 447,50, dtype=np.uint16),:]
            k_item = k_item[:,:, :,np.linspace(0, 447,50, dtype=np.uint16)]

        v_tensor = torch.Tensor(v_item)
        k_tensor = torch.Tensor(k_item)
        t_tensor = torch.Tensor(t_item)

        if self.crop:
            thickness = 10
            hthickness = thickness // 2
            temp = torch.where(torch.squeeze(torch.sum(t_tensor, axis=(2,3))))[0].numpy()

            temp = np.array([x for x in temp if x > hthickness])
            temp = np.array([x for x in temp if x < (np.shape(t_tensor)[1] - hthickness)])
            if np.any(temp) == 0:
                rng = np.shape(t_tensor)[1] // 2
            else:
                rng = random.choice(temp)
            v_tensor = v_tensor[:,(rng-hthickness):(rng+hthickness), ::]
            t_tensor = t_tensor[:,(rng-hthickness):(rng+hthickness), ::]
            k_tensor = k_tensor[:,(rng-hthickness):(rng+hthickness), ::]


        # constructs torch.Tensor object
        if self.transform:
            subject = tio.Subject(
                image = tio.ScalarImage(tensor=v_tensor),
                # label = tio.LabelMap(tensor=t_item>0)
            )
            v_trans = self.transform(subject)
            v_tensor = v_trans["image"].data
            yes_trans = ['Flip', 'Affine', "Crop"]
            trans = tio.Compose([x for x in v_trans.get_composed_history() if x.name in yes_trans])
            t_tensor = torch.Tensor((trans(t_tensor) > .5) * 1)
            k_tensor = torch.Tensor((trans(k_tensor) > .5) * 1)
            v_tensor = v_trans["image"].tensor
            # t_tensor = v_trans["label"].tensor

        none_tensor = torch.logical_not(k_tensor + t_tensor) * 1
        out = torch.cat((k_tensor, t_tensor, none_tensor), dim=0)
        return v_tensor, out

class SegmentationDataset2D(object):
    def __init__(self, h5_file, is_train, img_transform=None, reduce=False, crop = False):
        self.train = is_train
        self.file = h5_file
        self.transform = img_transform
        self.reduce = reduce
        self.crop = crop
        split = 'train' if is_train else 'test'
        print(split)
        with h5py.File(h5_file, 'r') as f:
            self.length = f['v'].shape[0]
        phase = "Train" if is_train else "Test"
        print(f"Loaded {phase} Dataset")

    def __getitem__(self, item):
        split = 'train' if self.train else 'test'
        with h5py.File(self.file, 'r') as f:
            v_item = f['v'][item][:,:]
            t_item = f['t'][item][:,:]
            k_item = f['k'][item][:,:]

        img_new = v_item.astype(np.float32)
        img_new = img_new[None, :,:, None]
        t_item = t_item[None, :,:, None]
        k_item = k_item[None, :,:, None]

        if self.transform:
            subject = tio.Subject(
                image = tio.ScalarImage(tensor=img_new),
                # label = tio.LabelMap(tensor=t_item>0)
            )
            v_trans = self.transform(subject)
            yes_trans = ['Affine', "ElasticDeformation", "Flip"]
            trans = tio.Compose([x for x in v_trans.get_composed_history() if x.name in yes_trans])
            t_item = torch.Tensor((trans(torch.Tensor(t_item)) > .5) * 1)
            k_item = torch.Tensor((trans(torch.Tensor(k_item)) > .5) * 1)
            img_new = v_trans["image"].tensor

        t_item = torch.Tensor(t_item)
        k_item = torch.Tensor(k_item)
        none_item = torch.logical_not(t_item + k_item) * 1
        mask_new = torch.stack((k_item, t_item, none_item))

        return img_new[:,:,:,0], mask_new[:,0,:,:,0]
    def __len__(self):
        return self.length

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class nll_loss(nn.Module):
    def __init__(self, device, alpha, eps):
        self.device = device
        self.alpha = alpha
        self.eps = eps
        super(nll_loss, self).__init__()
    def forward(self, hazards, S, Y, c):
        batch_size = len(Y)
        Y = Y.view(batch_size, 1).type(torch.LongTensor).to(self.device) # ground truth bin, 1,2,...,k
        c = c.view(batch_size, 1).float().to(self.device) # censorship status, 0 or 1
        if S is None:
            S = torch.cumprod(1 - hazards, dim=1).to(self.device)  # surival is cumulative product of 1 - hazards
        # without padding, S(0) = S[0], h(0) = h[0]
        S_padded = torch.cat([torch.ones_like(c).to(self.device), S],
                             1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
        # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
        # h[y] = h(1)
        # S[1] = S(1)
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=self.eps)) + torch.log(
            torch.gather(hazards, 1, Y).clamp(min=self.eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=self.eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss
        loss = loss.mean()
        return loss


ALPHA = 0.5
BETA = 0.5
GAMMA = 1
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky
def compute_per_channel_dice3D(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = torch.permute(input, (1,0,2,3,4))
    target = torch.permute(target, (1,0,2,3,4))

    input = torch.flatten(input, start_dim=1, end_dim=4)
    target = torch.flatten(target, start_dim=1, end_dim=4)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

def compute_per_channel_dice2D(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = torch.permute(input, (1,0,2,3))
    target = torch.permute(target, (1,0,2,3))

    input = torch.flatten(input, start_dim=1, end_dim=3)
    target = torch.flatten(target, start_dim=1, end_dim=3)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

class ClassificationDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train, img_transform=None, is_vol=False, target="label"):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.target = target
        self.X, self.label = \
            self._read_h5_file(h5_file, is_train)
        self.transform = img_transform
        self.is_vol = is_vol
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            label = f[split][self.target][()].reshape(-1, 1)

        return X, label

    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X - self.X.min(axis=0)) / \
                 (self.X.max(axis=0) - self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item].astype(np.float32)  # (m)
        label = self.label[item]  # (1)
        if self.is_vol:
            # center = np.random.randint(64, 192)
            center = 128
            # X_item = X_item[np.linspace(0,255, 32, dtype=np.uint16), :,:] # was 128
            X_item = X_item[(center - 25):(center + 25), :, :]  # was 128
            X_item = X_item[:, np.linspace(0, 99, 100, dtype=np.uint16), :]
            X_item = X_item[:, :, np.linspace(0, 99, 100, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            X_tensor = torch.Tensor(self.transform(X_item[None, :, :, :]))
        else:
            X_tensor = torch.from_numpy(X_item)
        label_tensor = torch.from_numpy(label)

        return X_tensor, label_tensor

    def __len__(self):
        return self.X.shape[0]

class SurvivalClassificationDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train, img_transform=None, is_vol = False, target = "label"):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.target = target
        self.X, self.label, self.e, self.t = \
            self._read_h5_file(h5_file, is_train)
        self.transform = img_transform
        self.is_vol = is_vol
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            if self.target == "tumor": # duplicate e,t but create labels for tumor volume and non-tumor volume
                label = np.concatenate((np.ones((len(X), 1)), np.zeros((len(X), 1))))
                e = np.concatenate((f[split]['e'][()].reshape(-1, 1), f[split]['e'][()].reshape(-1, 1)))
                t = np.concatenate((f[split]['t'][()].reshape(-1, 1), f[split]['t'][()].reshape(-1, 1)))
                X = np.concatenate((X, f[split]['x2'][()]))
                bla = 1
            elif self.target == "kidney":
                label = np.concatenate((np.ones((len(X), 1)), np.zeros((len(X), 1))))
                e = np.concatenate((f[split]['e'][()].reshape(-1, 1), f[split]['e'][()].reshape(-1, 1)))
                t = np.concatenate((f[split]['t'][()].reshape(-1, 1), f[split]['t'][()].reshape(-1, 1)))
                X = np.concatenate((X[:,(50-12):(50+12), (50-12):(50+12), (50-12):(50+12)], X[:,0:24, 0:24, 0:24]))
                # X = np.concatenate((X, np.random.normal(loc=np.mean(X), scale=np.std(X), size=X.shape)))
            else:
                label = f[split][self.target][()].reshape(-1, 1)
                e = f[split]['e'][()].reshape(-1, 1)
                t = f[split]['t'][()].reshape(-1, 1)

        # Remove nans
        nan_indices = np.where(np.isnan(label))[0]
        if len(nan_indices) > 0:
            X = np.delete(X, nan_indices, axis=0)
            label = np.delete(label, nan_indices, axis=0)
            e = np.delete(e, nan_indices, axis=0)
            t = np.delete(t, nan_indices, axis=0)

        return X, label, e, t

    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item].astype(np.float32) # (m)
        label = self.label[item] # (1)
        e = self.e[item] # (1)
        t = self.t[item] # (1)

        if self.is_vol:
            # center = np.random.randint(64, 192)
            center = 50
            # X_item = X_item[np.linspace(0,99, 50, dtype=np.uint16), :,:] # was 128
            # # X_item = X_item[(center-25):(center+25), :,:] # was 128
            # X_item = X_item[:, np.linspace(0, 99,50, dtype=np.uint16),:]
            # X_item = X_item[:, :,np.linspace(0, 99,50, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            X_tensor = torch.Tensor(self.transform(X_item[None, :,:,:]))
        else:
            X_tensor = torch.from_numpy(X_item)[None, ::]
        label_tensor = torch.from_numpy(label)

        e_tensor = torch.from_numpy(e)
        t_tensor = torch.from_numpy(t)

        return X_tensor, label_tensor, e_tensor,t_tensor

    def __len__(self):
        return self.X.shape[0]

class SurvivalMultiClassificationDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train, img_transform=None):
        ''' Loading data from .h5 file based on (h5_file, is_train).
        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.label, self.e, self.t = \
            self._read_h5_file(h5_file, is_train)
        self.transform = img_transform
        # normalizes data
        # self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.
        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            label = f[split]['label'][()].reshape(-1, 1)
            e = f[split]['e'][()].reshape(-1, 1)
            t = f[split]['t'][()].reshape(-1, 1)

        return X, label, e, t

    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item].astype(np.float32) # (m)
        label = self.label[item] # (1)
        e = self.e[item]# (1)
        t = self.t[item]
        center = 50
        X_item = X_item[(center-12):(center+12), :,:] # was 128
        X_item = X_item[:, np.linspace(0, 99,25, dtype=np.uint16),:]
        X_item = X_item[:, :,np.linspace(0, 99,25, dtype=np.uint16)]
        # constructs torch.Tensor object
        if self.transform:
            X_tensor = torch.Tensor(self.transform(X_item[None, :,:,:]))
        else:
            X_tensor = torch.from_numpy(X_item)[None, ::]
        label_tensor = torch.from_numpy(label)
        e_tensor = torch.from_numpy(e)
        t_tensor = torch.from_numpy(t)

        return X_tensor, label_tensor, e_tensor, t_tensor

    def __len__(self):
        return self.X.shape[0]

# class GradCAM3D(BaseCAM):
    # def __init__(self, model, target_layers, use_cuda=False,
                 # reshape_transform=None):
        # super(
            # GradCAM3D,
            # self).__init__(
            # model,
            # target_layers,
            # use_cuda,
            # reshape_transform)

    # def get_target_width_height(self,input_tensor: torch.Tensor) -> Tuple[int, int, int]:
        # depth, height, width = input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)
        # return depth, height, width
    # def get_cam_weights(self,
                        # input_tensor,
                        # target_layer,
                        # target_category,
                        # activations,
                        # grads):
        # return np.mean(grads, axis=(2, 3))
    # def compute_cam_per_layer(
            # self,
            # input_tensor: torch.Tensor,
            # targets: List[torch.nn.Module],
            # eigen_smooth: bool) -> np.ndarray:
        # activations_list = [a.cpu().data.numpy()
                            # for a in self.activations_and_grads.activations]
        # grads_list = [g.cpu().data.numpy()
                      # for g in self.activations_and_grads.gradients]
        # target_size = self.get_target_width_height(input_tensor)

        # cam_per_target_layer = []
        # # Loop over the saliency image from every layer
        # for i in range(len(self.target_layers)):
            # target_layer = self.target_layers[i]
            # layer_activations = None
            # layer_grads = None
            # if i < len(activations_list):
                # layer_activations = activations_list[i]
            # if i < len(grads_list):
                # layer_grads = grads_list[i]

            # cam = self.get_cam_image(input_tensor,
                                     # target_layer,
                                     # targets,
                                     # layer_activations,
                                     # layer_grads,
                                     # eigen_smooth)
            # cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])

        # return cam_per_target_layer

# def scale_cam_image(cam, target_size=None):
    # result = []
    # for img in cam:
        # img = img - np.min(img)
        # img = img / (1e-7 + np.max(img))
        # if target_size is not None:
            # delta = tuple([x/y for x,y in zip(target_size, np.shape(img))])
            # img = zoom(img, delta)
            # # img = cv2.resize(img, target_size)
        # result.append(img)
    # result = np.float32(result)

    # return result

class element_weighted_bce_loss(nn.Module):
    def __init__(self, weights, factor = 0.7):
        super(element_weighted_bce_loss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
        self.factor = factor
    def forward(self, inputs, targets, event):
        loss = self.loss(inputs, targets)
        if self.factor is None:
            return torch.sum(loss)
        else:
            class0_loss = torch.sum((loss * ((1-event)) * (1-self.factor)))
            class1_loss = torch.sum(loss * event * self.factor)
            total_loss = class0_loss + class1_loss
            return total_loss

