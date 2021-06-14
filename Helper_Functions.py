import numpy as np
import random 
import os
import sys
import pickle
import matplotlib
matplotlib.use('agg') 
# the automatic one on my comps (TkAgg) causes segfaults in certain cases, 
# see: https://github.com/matplotlib/matplotlib/issues/5795
import matplotlib.pyplot as plt
from time import time, process_time
import gzip
import torch
from datetime import datetime


def timestamp():
	return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


def mean(v): # unlike numpy's mean, doesnt get upset with tensors
	return sum(v)/len(v) 

def pad(s,l=5):
	total = l-len(str(s))
	lohalf = int(np.floor(total/2))
	hihalf = total - lohalf
	return " "*lohalf + str(s)+" "*hihalf

def class_to_dict(class_instance,ignoring_list=None):
	if None is ignoring_list:
		ignoring_list = []
	if hasattr(class_instance,"ignore_in_dict"):
		ignoring_list += class_instance.ignore_in_dict
	values = [p for p in dir(class_instance) if (not "__" in p) and\
	 (not callable(getattr(class_instance,p))) and (not p in ignoring_list)]
	return {p:getattr(class_instance,p) for p in values}

def chronological_scatter(vec,alpha=1,title="",vec2=None,vec1label=None,vec2label=None,filename=None,s=2):
	plt.scatter(range(len(vec)),vec,alpha=alpha,label=vec1label,s=s)
	if not None is vec2:
		plt.scatter(range(len(vec2)),vec2,alpha=alpha,label=vec2label,s=s)
	plt.title(title)
	if not (None is vec1label and None is vec2label):
		plt.legend()
	if not None is filename:
		prepare_directory(filename)
		plt.savefig(filename)
		plt.close()
	else:
		plt.show()

def softmax(x): # taken from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
	"""Compute softmax values for each sets of scores in x."""
	x = np.array(x)
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()
  
def pad_to_length(seq,length):
	return str(seq)+" "*(length-len(str(seq)))

def clean_val(num,digits=3):
	if digits == np.inf:
		return num
	if num in [np.inf,np.nan]:
		return num
	res = round(num,digits)
	if digits == 0:
		res = int(res)
	return res

def prepare_directory(path,includes_filename=True):
	if includes_filename:
		path = "/".join(path.split("/")[:-1])
	if len(path)==0:
		return
	if not os.path.exists(path):
		# print("making path:",path)
		os.makedirs(path)

def prepare_txtfile_if_not_ready(path,clear=False):
	prepare_directory(path,includes_filename=True)
	if clear or not os.path.exists(path):
		with open(path,"w") as f:
			print("",end="",file=f)

def overwrite_file(contents,filename,dont_zip=False):
	if not dont_zip:
		filename += "" if filename.endswith(".gz") else ".gz"
	prepare_directory(filename)
	open_fun = open if dont_zip else gzip.open
	with open_fun(filename,'wb') as f:
		pickle.dump(contents,f)

def load_from_file(filename,quiet=False):
	if not quiet:
		print("trying to load data from file:",filename)
	if not os.path.exists(filename):
		if not filename.endswith(".gz"):
			return load_from_file(filename+".gz",quiet=quiet) # maybe have it zipped
		if not quiet:
			print("no such file: ",filename)
		return None
	open_fun =  gzip.open if filename.endswith(".gz") else open 
	with open_fun(filename,'rb') as f:
		res = pickle.load(f)
	return res

def things_in_path(path,ignoring_list=None,only_folders=False,only_files=False):
	if not os.path.exists(path):
		return []
	ignoring_list = ignoring_list if not None is ignoring_list else []
	ignoring_list.append(".DS_Store")
	ignoring_list.append(".ipynb_checkpoints")
	res = sorted([name for name in os.listdir(path) if not name in ignoring_list])
	if only_folders:
		res = [r for r in res if os.path.isdir(path+"/"+r)]
	if only_files:
		res = [r for r in res if os.path.isfile(path+"/"+r)]
	return res

pytorch_ignore_when_saving = ['_backward_hooks', '_buffers', '_forward_hooks', '_forward_pre_hooks',\
 	'_load_state_dict_pre_hooks', '_modules', '_parameters', '_state_dict_hooks', '_version',
 	'T_destination','_non_persistent_buffers_set','dump_patches']


def save_pytorch_model(model,folder,name="model"):
	if hasattr(model,"parameters"):
		was_cuda = next(model.parameters()).is_cuda
		model.cpu() # seems like a better idea if want to open on another computer later
	prepare_directory(folder,includes_filename=False)
	if hasattr(model,"build_from_dict"):
		overwrite_file(
			class_to_dict(model,ignoring_list=pytorch_ignore_when_saving),
			folder+"/"+name+"_dict")
	torch.save(model.state_dict(),folder+"/"+name)
	if hasattr(model,"parameters") and was_cuda: # put it back!!
		model.cuda()


def load_pytorch_model(empty_model,folder,name="model",quiet=False):	
	model_path = folder+"/"+name
	if not os.path.exists(model_path):
		return None
	if hasattr(empty_model,"build_from_dict"):
		source_dict = load_from_file(model_path+"_dict",quiet=quiet)
		if None is source_dict:
			return None
		empty_model.build_from_dict(source_dict)

	empty_model.load_state_dict(torch.load(model_path))
	res = empty_model
	if hasattr(res,"parameters"):
		res.eval() # always load for eval, whenever training can set explicitly to train and then return to eval
		res.cuda() if torch.cuda.is_available() else res.cpu()
	return res
