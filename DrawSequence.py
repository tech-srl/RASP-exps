# import torch # probably need for indexing

import numpy as np
from Helper_Functions import prepare_directory, clean_val
import matplotlib.pyplot as plt
import math


def process_and_draw(model,seq,target_folder="",into_subfolder=None,\
	with_numbers=-1,hyperparams_str="",size=5,gold=None,
	reset_main_txt=False,with_colorbar=True):
	is_cuda = model.is_cuda()
	model.cpu()
	full_attn_internals , full_outs, model_output = model.get_distrs_and_outs(seq) # layers X heads X tgt len X src len
	# print("model ouput was:",model_output,"\t(",subfolder,", test acc:",last_test_acc(model),")")
	
	full_seq = (model.non_token + seq) if model.add_BOS_to_input else (seq) 
	if not target_folder:
		target_folder = "maps/" + model.lang_name +"/" +full_seq+"/"
	if not None is into_subfolder:
		target_folder += into_subfolder + "/"
	top_target_folder = target_folder
	if hyperparams_str:
		target_folder += hyperparams_str + "/"

	draw_distrs(full_seq,full_attn_internals,target_folder,size,with_numbers,with_colorbar=with_colorbar)
	draw_scores(full_seq,full_attn_internals,target_folder,size,with_numbers)

	print_extra_info(target_folder,hyperparams_str,seq,model_output,gold,model,filename="in_out.txt",copy_to_terminal=True)
	if not (target_folder == top_target_folder):
		print_extra_info(top_target_folder,hyperparams_str,seq,model_output,gold,model,
							append=not reset_main_txt,filename="ins_outs.txt",copy_to_terminal=False)
	if is_cuda:
		model.cuda()

def as_str(seq,non_token):
	with_bos = seq[0] == non_token
	if with_bos:
		seq = seq[1:]
	tostr = (lambda b:"T" if b else "F") if isinstance(seq[0],bool) else str
	as_strs = list(map(tostr,seq))
	if with_bos:
		as_strs = [str(non_token)] + as_strs
	joiner = "" if set(len(s) for s in as_strs)==set([1]) else ", "
	return joiner.join(as_strs)

def choose_dims(num_ims,force_even_ncols=False):
	nrows = int(math.sqrt(num_ims))
	ncols = math.ceil(num_ims/nrows)
	if (ncols%2==1) and force_even_ncols:
		ncols += 1
	nrows = math.ceil(num_ims/ncols)
	assert nrows*ncols >= num_ims
	return nrows, ncols

def last_test_acc(model):
	l = model.metrics["test_accs"]
	return l[-1] if l else -1

def doubleprint(*a,file=None,**kw):
	print(*a,**kw)
	if not None is file:
		print(*a,**kw,file=file)

def draw_distrs(full_seq,full_attn_internals,target_folder,size,with_numbers,with_colorbar=True):
	full_distrs = {i:full_attn_internals[i].distributions for i in full_attn_internals}
	nrows,ncols = choose_dims(len(full_distrs[0]))
	draw_distrs_or_scores(full_seq,full_distrs,target_folder+"distributions/",
		with_numbers=with_numbers,width=size*ncols,height=size*nrows,probabilities=True,with_colorbar=with_colorbar)

def draw_scores(full_seq,full_attn_internals,target_folder,size,with_numbers):
	full_scores = {i:full_attn_internals[i].scores for i in full_attn_internals}
	nrows,ncols = choose_dims(len(full_scores[0]))
	draw_distrs_or_scores(full_seq,full_scores,target_folder+"scores/",
		with_numbers=with_numbers,width=size*ncols,height=size*nrows,probabilities=False)

def print_extra_info(target_folder,hyperparams_str,seq,model_output,gold,model,
					append=False,copy_to_terminal=True,filename="in_out.txt"):
	def _print_extra_info(f,model_output):
		pp = doubleprint if copy_to_terminal else print
		pp("========================",file=f)
		pp("model with hyperparams:",hyperparams_str,"(num epochs:",len(model.metrics["train_accs"]),")",file=f)
		pp("model has test acc:",last_test_acc(model),file=f)
		pp("",file=f)
		pp("input length:",len(seq),file=f)
		pp("input:  ",seq,file=f)
		if isinstance(seq,str):
			model_output = as_str(model_output,model.non_token)
		pp("output: ",model_output,file=f)
		if not None is gold:
			target = gold(seq)
			if len(target) == 2 and isinstance(target[1],dict):
				# target is actually output seq X attentions
				target = target[0]
			if isinstance(seq,str):
				target = as_str(target,model.non_token)
			pp("target: ",target,file=f)
		pp("========================",file=f)

	if append:
		with open(target_folder+"/"+filename,"a") as f:
			_print_extra_info(f,model_output)
	else:
		with open(target_folder+"/"+filename,"w") as f:
			_print_extra_info(f,model_output)


def fill_img(ax,data,ylabels=None,xlabels=None,name="",with_numbers=-1,
					add_cbar=False,probabilities=False):
	# plot drawing copied almost verbatim from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
	im = ax.imshow(data,aspect='auto')
	if not None is xlabels:
		# ... and label them with the respective list entries
		ax.set_xticks(np.arange(len(xlabels)))
		ax.set_xticklabels(list(xlabels))
	if not None is ylabels:
		ax.set_yticks(np.arange(len(ylabels)))
		ax.set_yticklabels(list(ylabels))
	# Loop over data dimensions and create text annotations.
	if with_numbers>0:
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				text = ax.text(j, i, clean_val(data[i][j].tolist(),digits=with_numbers),
							   ha="center", va="center", color="w")
	ax.set_title(name)
	# if probabilities:
		# im.set_clim(0,1)
	if add_cbar:
		cbar = ax.figure.colorbar(im, ax=ax)
	return im

def draw_heatmaps(matrices,with_numbers,subplot_name_fn,
					figsize=None,xlabels=None,ylabels=None,
					always_cbar=False,probabilities=False,
					with_colorbar=True):
	if not isinstance(matrices,dict):
		matrices = {i:matrices[i] for i in range(len(matrices))}
	nrows,ncols = choose_dims(len(matrices))
	if not None is figsize:
		fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
	else:
		fig,axes = plt.subplots(nrows=nrows,ncols=ncols)
	for i,mat_key in enumerate(sorted(list(matrices.keys()))):
		mat = matrices[mat_key]
		j = int(i/ncols) # row
		k = i - (j*ncols) # column: remainder after div by num columns
		if nrows==1 and ncols==1:
			a=axes
		elif nrows==1:
			a = axes[k]
		else:
			a = axes[j,k]
		img = fill_img(a,mat,xlabels=xlabels,ylabels=ylabels,
			name=subplot_name_fn(i),with_numbers=with_numbers,
			add_cbar=with_colorbar and (always_cbar or (k==(ncols-1))),
			probabilities=probabilities)
	return fig


def draw_distrs_or_scores(full_seq,full_distrs,target_folder,
	with_numbers=-1,width=5,height=5,probabilities=False,with_colorbar=True):
# hs = range(len(full_distrs[0]))
	for l in range(len(full_distrs)):
		fig = draw_heatmaps(full_distrs[l],with_numbers,lambda i:"head "+str(i),
							xlabels=full_seq,ylabels=full_seq,
							figsize=(width,height),probabilities=probabilities,
							with_colorbar=with_colorbar)
		what = "probabilities" if probabilities else "scores"
		fig.suptitle(what+" of each token (y axis) over full sequence (x axis)")
		fig.tight_layout()
		target_file = target_folder+"/layer "+str(l)+".png"
		prepare_directory(target_file,includes_filename=True)
		plt.savefig(target_file)
		plt.close()



