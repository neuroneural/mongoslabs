from typing import Sized
import pickle

import numpy as np
from pymongo import MongoClient
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from mongoslabs.gencoords import CoordsGenerator


class MongoDataset(Dataset):
    """
    A dataset for fetching batches of records from a MongoDB
    """
    def __init__(self, indices, transform, collection, fields=None, id="id"):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param transform: a function to be applied to each extracted record
        :param collection: pymongo collection to be used
        :param fields: a list of fields to be fetched for each record
        :param id: the field to be used as an index. The `indices` are values of this field
        :returns: an object of MongoDataset class

        """
        self.indices = indices
        self.transform = transform
        self.collection = collection
        self.fields = fields
        self.id = id

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, batch):
        if self.fields is None:
            field_list = {}
        else:
            field_list = {_: 1 for _ in self.fields}
        samples = self.collection.find(
            {self.id: {"$in": [self.indices[_] for _ in batch]}}, field_list
        )
        return [self.transform(_) for _ in samples]


class MBatchSampler(Sampler):
    """
    A batch sampler from a random permutation. Used for generating indices for MongoDataset
    """
    data_source: Sized

    def __init__(self, data_source, batch_size=1):
        """TODO describe function

        :param data_source: a dataset of Dataset class
        :param batch_size: number of samples in the batch (sample is an MRI split to 8 records)
        :returns: an object of mBatchSampler class

        """
        self.batch_size = batch_size
        self.data_source = data_source
        self.data_size = len(self.data_source)

    def __chunks__(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def __iter__(self):
        return self.__chunks__(np.random.permutation(self.data_size), self.batch_size)

    def __len__(self):
        return len(self.data_source)


def create_client(worker_id,
                  dbname,
                  colname,
                  mongohost):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    collection = colname
    client = MongoClient("mongodb://" + mongohost + ":27017")
    dataset.collection = client[dbname][collection]


def preprocess_image(img):
    """Unit interval preprocessing"""
    img = (img - img.min()) / (img.max() - img.min())
    return img


def list2dict(mlist):
    mdict = {}
    for element in mlist:
        if element["subject"] in mdict:
            mdict[element["subject"]].append(element)
        else:
            mdict[element["subject"]] = [element]
    return mdict


def mtransform(subrecord, data="subdata", label="sublabel"):
    subrecord[data] = pickle.loads(subrecord[data])
    subrecord[label] = pickle.loads(subrecord[label])
    return subrecord


def mcollate(mlist, labelname="sublabel", cubesize=256):
    mdict = list2dict(mlist[0])
    data = []
    labels = []
    data = torch.empty(
        len(mdict), cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.float
    )
    labels = torch.empty(
        len(mdict), cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.long
    )
    cube = np.empty(shape=(cubesize, cubesize, cubesize))
    label = np.empty(shape=(cubesize, cubesize, cubesize))
    for i, subj in enumerate(mdict):
        for sub in mdict[subj]:
            x, y, z = sub["coords"]
            sz = sub["subdata"].shape[0]
            cube[x : x + sz, y : y + sz, z : z + sz] = sub["subdata"]
            label[x : x + sz, y : y + sz, z : z + sz] = sub[labelname]
        cube1 = preprocess_image(torch.from_numpy(cube).float())
        label1 = torch.from_numpy(label).long()
        data[i, :, :, :] = cube1
        labels[i, :, :, :] = label1
    del cube
    del label
    return data.unsqueeze(1), labels


def mcollate(mlist, labelname="sublabel", cubesize=256):
    mdict = list2dict(mlist[0])
    num_subjs = len(mdict)
    if num_subjs > 1:
        data = torch.empty(
            num_subjs, cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.float
            )
        labels = torch.empty(
            num_subjs, cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.long
            )
        for i, subj in enumerate(mdict):
            mdict[subj].sort(key=lambda _: _["id"])
            cube = np.vstack([sub["subdata"] for sub in mdict[subj]])
            label = np.vstack([sub[labelname] for sub in mdict[subj]])
            data[i, :, :, :] = preprocess_image(torch.from_numpy(cube).float())
            labels[i, :, :, :] = torch.from_numpy(label).long()
    else:
        subj = next(iter(mdict))
        mdict[subj].sort(key=lambda _: _["id"])
        cube = torch.vstack([torch.from_numpy(sub["subdata"]).float() for sub in mdict[subj]])
        label = torch.vstack([torch.from_numpy(sub[labelname]).long() for sub in mdict[subj]])

        data = preprocess_image(cube).unsqueeze(0)
        labels = label.unsqueeze(0)

    return data.unsqueeze(1), labels.long()

def collate_subcubes(mlist, coord_generator, samples=4,
                     labelname="sublabel", cubesize=256):
    data, labels = mcollate(mlist, labelname=labelname, cubesize=cubesize)
    num_subjs = labels.shape[0]
    data = data.squeeze(1)
    coords = coord_generator.get_coordinates()
    size = coords[0][1] - coords[0][0]

    batch_data = torch.empty(
        num_subjs * samples, size, size, size, dtype=torch.float
    )
    batch_labels = torch.empty(
        num_subjs * samples, size, size, size, dtype=torch.long
    )

    for i in range(num_subjs):
        batch_data, batch_labels = subcube_list(
            data[i,:,:,:],
            labels[i,:,:,:],
            samples,
            batch_data,
            batch_labels,
            i,
            coord_generator,
        )
    return batch_data.unsqueeze(1), batch_labels


def subcube_list(cube, labels, num, subcubes, sublabels, subj, coords_generator):
    for i in range(num):
        coords = coords_generator.get_coordinates()
        subcubes[subj * num + i, :, :, :] = cube[
            coords[0][0] : coords[0][1],
            coords[1][0] : coords[1][1],
            coords[2][0] : coords[2][1],
        ]
        sublabels[subj * num + i, :, :, :] = labels[
            coords[0][0] : coords[0][1],
            coords[1][0] : coords[1][1],
            coords[2][0] : coords[2][1],
        ]
    return subcubes, sublabels
