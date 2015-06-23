# Copyright 2013    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
import gzip
import os
import sys, re
import glob

import numpy
import theano
import theano.tensor as T
from utils.utils import string_2_bool
from model_io import log
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label


def load(f):
    # File format:
    #
    # Big file -
    # num_row num_col
    # sample_1 0.1 0.2 0.4 ... 20
    # sample_2 0.1 0.2 0.3 ... 13
    #
    # Small file -
    # sample_1 0.1 0.2 0.4 ... 20
    # sample_2 0.1 0.2 0.3 ... 13

    l = f.readline()
    l = l.strip().split(' ')
    is_big = True
    label_arr = []

    if l[0].isdigit():
	    # big file
	    c = 0
	    num_row, num_col = int(l[0]), int(l[1])
	    array = numpy.zeros((num_row, num_col), dtype=numpy.float32)
    else:
	    # small file
	    is_big = False
	    array = []
	    label_arr.append(int(l[-1]))
	    array.append(numpy.array([float(e) for e in l[1:-1]], dtype=numpy.float32))
	    
    while True:
	    l = f.readline()
	    if not l:
		    break
	    l=l.split(' ')
	    label_arr.append(int(l[-1]))
	    if is_big:
		    array[c] = numpy.array([float(e) for e in l[1:-1]], dtype=numpy.float32)
		    c += 1
	    else:
		    array.append(numpy.array([float(e) for e in l[1:-1]], dtype=numpy.float32))

    label_arr = numpy.asarray(label_arr, dtype=numpy.float32)
    array = array if is_big else numpy.asarray(array, dtype=numpy.float32)

    return [array, label_arr]

class ArkDataRead(object):

    def __init__(self, pfile_path_list, read_opts):

        self.pfile_path_list = pfile_path_list
        self.cur_pfile_index = 0
        self.pfile_path = pfile_path_list[0]
        self.read_opts = read_opts

        self.feat_mat = None
        self.label_vec = None

        # other variables to be consistent with PfileDataReadStream
        self.cur_frame_num = 0
        self.end_reading = False

    def load_next_partition(self, shared_xy):
        pfile_path = self.pfile_path_list[self.cur_pfile_index]
        if self.feat_mat is None or len(self.pfile_path_list) > 1:
            fopen = smart_open(pfile_path, 'rb')
            self.feat_mat, self.label_vec = load(fopen)
            fopen.close()
            shared_x, shared_y = shared_xy

            self.feat_mat, self.label_vec = \
                preprocess_feature_and_label(self.feat_mat, self.label_vec, self.read_opts)
            if self.read_opts['random']:
                shuffle_feature_and_label(self.feat_mat, self.label_vec)

            shared_x.set_value(self.feat_mat, borrow=True)
            shared_y.set_value(self.label_vec.astype(theano.config.floatX), borrow=True)

        self.cur_frame_num = len(self.feat_mat)
        self.cur_pfile_index += 1

        if self.cur_pfile_index >= len(self.pfile_path_list):   # the end of one epoch
            self.end_reading = True
            self.cur_pfile_index = 0

    def is_finish(self):
        return self.end_reading

    def initialize_read(self, first_time_reading = False):
        self.end_reading = False

    def make_shared(self):
        # define shared variables
        feat = numpy.zeros((10,10), dtype=theano.config.floatX)
        label = numpy.zeros((10,), dtype=theano.config.floatX)

        shared_x = theano.shared(feat, name = 'x', borrow = True)
        shared_y = theano.shared(label, name = 'y', borrow = True)
        return shared_x, shared_y

