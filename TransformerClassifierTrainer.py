import torch
import torch.nn as nn
from copy import deepcopy
from random import shuffle
from Helper_Functions import mean, save_pytorch_model, load_pytorch_model,\
		 prepare_txtfile_if_not_ready, chronological_scatter, prepare_directory
import random
import numpy as np
import sys
from time import process_time
import matplotlib.pyplot as plt


STOP, CONTINUE = 0,1


class TransformerEncoderClassifierTrainer:
	def __init__(self,model,lr,store_folder,lr_factor=0.5,lr_patience=2,
		reset_patience=5,init_from_checkpoint=False,training_out=sys.stdout,\
		optim="adam",gamma=0.9,loud=False,perfect_acc_cut=3,attn_loss_c=1,plot_every=50):
		self.store_folder = store_folder
		self.plots_path = self.store_folder+"/training_plots"
		prepare_directory(self.plots_path,includes_filename=False)
		self.epochs_without_improvement = 0
		self.reset_patience = reset_patience
		self.model = model # even if its going to be initiated from the checkpoint, need a dummy model to dump everything into
		self.pos_probs = {}
		self.neg_probs = {}
		self.perfect_acc_cut = perfect_acc_cut if perfect_acc_cut>0 else np.inf
		self.perfect_acc_count = 0
		self.attn_loss_c = attn_loss_c
		self.optim_type = optim
		self.gamma = gamma
		self.initial_lr = lr
		self.plot_every = plot_every
		self.plot_counter = 0
		
		if init_from_checkpoint:
			self.load_model(keep_current_metrics=False) # load_model automatically make optims for the model
			if None is self.model:
				return # failed to load
		else:
			self.make_optims() # need optimisers to match given model

		self.best_val_loss = getattr(self.model,"best_val_loss",np.inf)
		print("loaded model with best val loss:",tonum(self.best_val_loss),file=training_out)			
		# these two are for bookkeeping somewhere, look into what exactly later
		self.model.last_gamma = gamma 
		self.model.last_initial_lr = lr

		self.training_out = sys.stdout # the training_out file may be temporary, always use stdout unless given other option
		
	def set_loss_func(self):
		# makes self.loss_func, which gets input: Y, probs, mask, lengths.
		# in case that training attention as well, Y is actually (seq_Y,attn_Y)

		if self.model.finite_classes:
			main_loss_func = self.classifier_loss_finite_classes if self.model.is_classifier else\
			 lambda y,p,m,lengths:self.fullseq_loss_finite_classes(y,p,m,lengths,self.model.add_BOS_to_input)
		else:
			main_loss_func = self.classifier_loss_infinite_classes if self.model.is_classifier else\
			 lambda y,p,m,lengths:self.fullseq_loss_infinite_classes(y,p,m,lengths)
				
		if self.model.training_attn:
			def combined_loss(Y,probs,mask,lengths):
				# Y was tuple of (out1,out2,...,outN)
				# s.t. outi = (actual_out_i,target_attns_i)
				seq_Y,attn_Y = tuple(zip(*Y))
				seq_loss, seq_acc = main_loss_func(seq_Y,probs,mask,lengths) 
				attn_loss = self.attn_loss(attn_Y,probs,mask,lengths)
				total_loss = seq_loss + (self.attn_loss_c * attn_loss)
				self.model.debprint("seq loss:",tonum(seq_loss.data),
						"c * attn loss:",tonum(attn_loss.data)*self.attn_loss_c)
				return (total_loss,seq_loss,attn_loss), seq_acc # for now
			self.loss_func = combined_loss
		else:
			self.loss_func = main_loss_func

	def attn_loss(self,Y,probs,mask,lengths):
		# Y is: target_attns_1, tgt_attns2, ..., tgt_attnsN where N=batch_size
		MSELoss = torch.nn.MSELoss(reduction="mean")
		def simple_normalise(a):
			# normalises across dim 1, i.e. sums and divides per-row
			# also add epsilon in case a row has sum 0 once in a while (some selectors will)
			a = a+1e-8
			return a/(torch.sum(a,1).view(a.shape[0],1).expand(a.shape))

		def single_head_loss(target,actual):
			# check: is actual's shape actually batch_max_len X batch_max_len? and, is it zeros everywhere the padding is?
			# if so, *might* be convenient to do whole batch's loss together
			# however, MSE loss gets smaller if there is padding with more zeros, at least if using reduction='mean', so maybe dont?
			target = self.model.make_tensor(target).float() # .float() is voodoo for loss.backward() again, otherwise it gets upset, even if do things like dtype=torch.Double or whatever
			# target shape is: length(output) X length (input)
			target = simple_normalise(target) # normalise along input dimension (have given selectors)
			# print("actual shape:",actual.shape,"target shape:",target.shape)
			# self.model.debprint("target is:",target)
			actual = actual[:target.shape[0],:target.shape[1]]
			# self.model.debprint("actual is:",actual)
			res = MSELoss(actual,target)
			# self.model.debprint("loss is:",res)
			return res
		num_heads_lossed = 0
		loss = self.model.make_tensor(0.0)
		# self.model.debprint("getting attn loss")
		for i,tgt_attn in enumerate(Y):
			# i is index within batch
			for l in tgt_attn:
				for h in tgt_attn[l]:
					# self.model.debprint("getting loss for layer",l,"head",h,end="")
					head_target = tgt_attn[l][h] # batch size X tgt len X src len # (still have to make)
					head_actual = self.model.get_differentiable_last_attn(l,h)[i] # output len X input len
					loss += single_head_loss(head_target,head_actual) # make sure this thing 
					# normalises for actual number of positions being 'tested', eg length X length X batch size (if all sequences in batch are same length)
					num_heads_lossed += 1
		return loss/num_heads_lossed
		# still TODO: 1. datasets have to add the per-head-goal into their Y
		# 2. ideally, transformer automatically converts to train on attention as well if sees this in dataset (instead of manually telling it to do so)
		# 3. have training fail if too few layers/heads for request?



	def reset_best_loss(self,new_val=np.inf):
		self.best_val_loss = new_val # might want to do this when coming back to a model but with a different dataset

	def make_optims(self,keep_lr=False):
		lr = self.initial_lr if not keep_lr else self.lr_scheduler.state_dict()["_last_lr"][0]
		self.model.debprint("making optims for model")
		if self.optim_type=="sgd":
			self.optim = torch.optim.SGD(self.model.parameters(),lr=lr,momentum=0)
		elif self.optim_type=="adam":
			self.optim = torch.optim.Adam(self.model.parameters(),lr=lr) 
		else:
			raise Exception("optim type must be sgd or adam, got:",self.optim_type)
		self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
			self.optim,
			gamma=self.gamma
			)
		# alternative: 
		# self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		# 	self.optim, 
		# 	mode='min', # lr is decreased when tracked value stops decreasing (as opposed to increasing) 
		# 	factor=lr_factor, patience=lr_patience, threshold=0.0001, 
		# 	threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

	def dump_checkpoint(self):
		self.model.best_val_loss = tonum(self.best_val_loss) # 'tonum' because otherwise we'll end up with gpu stuff in the
		# model (outside of its actual modules) and i wont be able to load it on my computer
		save_pytorch_model(self.model,self.store_folder,"model")

	def load_model(self,keep_current_metrics=True,keep_lr=False):
		metrics = self.model.metrics
		self.model = load_pytorch_model(self.model,self.store_folder,"model")
		if None is self.model:
			return
		if keep_current_metrics:
			self.model.metrics = metrics
		self.best_val_loss = self.model.best_val_loss
		self.make_optims(keep_lr=keep_lr)

	def load_checkpoint(self,keep_lr=False,keep_current_metrics=True): # might want optimisers to keep decreasing if failing to improve between checkpoints
		self.load_model(keep_current_metrics=keep_current_metrics,keep_lr=keep_lr)

	def check_and_save(self,val_loss):
		if isinstance(val_loss,tuple):
			val_loss, _, _ = val_loss # split to total, seq, attn
		if val_loss < self.best_val_loss:
			self.best_val_loss = val_loss
			self.dump_checkpoint()
			self.epochs_without_improvement = 0			
			print("\t\t !! validation improved - saved checkpoint. best val now:,",loss_str(self.best_val_loss),file=self.training_out)
		else:
			self.epochs_without_improvement += 1
			if self.epochs_without_improvement > self.reset_patience:
				self.load_checkpoint(keep_lr=True) # continue from a lower lr, maybe that's what was going wrong
				if len(self.model.metrics["reloads"])>0:
					self.model.metrics["reloads"][-1] = 1
				print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
				    "\nvalidation not improved for too long (",
					  self.epochs_without_improvement,"epochs), restored from last save",
			        "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",file=self.training_out)
				self.epochs_without_improvement = 0

	def append_vals(self,losses,accs,overwrite=False):
		def add(l,v):
			if len(l)==0 or not overwrite:
				l.append(v)
			else:
				l[-1]=v

		# fix losses representation before doing anything else
		with_attn_losses = isinstance(losses["train"],tuple)
		if with_attn_losses:
			assert isinstance(losses["val"],tuple)
			assert isinstance(losses["test"],tuple)
			for n in ["train","test","val"]:
				losses[n], losses[n+"_seq"], losses[n+"_attn"] = losses[n]

		losses = {d:tonum(losses[d]) for d in losses}
		accs = {d:tonum(accs[d]) for d in accs}

		if with_attn_losses:
			for tv in ["train","val","test"]:
				for sa in ["seq","attn"]:
					add(self.model.metrics[tv+"_"+sa+"_losses"],losses[tv+"_"+sa])

		print("epoch",len(self.model.metrics["train_losses"])+1,
				", new train loss:",losses["train"],"new val loss:",losses["val"])

		add(self.model.metrics["max_param_vals"],max([t.abs().max().item() for t in self.model.parameters()]))
		add(self.model.metrics["avg_param_vals"],mean([t.abs().mean().item() for t in self.model.parameters()]))
		add(self.model.metrics["reloads"],0)
		for tv in ["train","val","test"]:
			add(self.model.metrics[tv+"_losses"],losses[tv])
			add(self.model.metrics[tv+"_accs"],accs[tv])

		add(self.model.metrics["train_minus_val_losses"],losses["train"]-losses["val"])

		# attn training

		val_acc = accs["val"]
		if val_acc == 1:
			self.perfect_acc_count += 1
			if self.perfect_acc_count >= self.perfect_acc_cut:
				return STOP
		else:
			self.perfect_acc_count = 0
		return CONTINUE



	def plot_vals(self,force=False):
		self.plot_counter += 1
		if not (force or (self.plot_counter%self.plot_every==1)):
			return
		filename = lambda x:self.plots_path+"/others/"+x+".png"
		main_filename = lambda x:self.plots_path+"/"+x+".png"
		for n in self.model.metrics:
			chronological_scatter(self.model.metrics[n],title=n,filename=filename(n),alpha=0.3) 
		chronological_scatter(self.model.metrics["train_losses"],vec2=self.model.metrics["val_losses"],
			title="train vs validation losses",vec1label="train",vec2label="val",
			filename=main_filename("train_vs_val_losses"),alpha=1)
		chronological_scatter(self.model.metrics["train_accs"],vec2=self.model.metrics["val_accs"],
			title="train vs validation accuracies",vec1label="train",vec2label="val",
			filename=main_filename("train_vs_val_accs"),alpha=1)

		diffs_filename = self.plots_path+"/others/train_minus_val_losses.txt"
		with open(diffs_filename,"w") as f:
			print("\n".join([str(v) for v in self.model.metrics["train_minus_val_losses"]]),file=f)

	def classifier_loss_infinite_classes(self,Y,preds,mask,lengths=None):
		# lengths is throwaway for uniformity with other funcs
		loss_fn = torch.nn.MSELoss(reduction='mean') # mean - will average the loss over the batch. 
		# MSELoss does does L2 (ie (x-y)^2), can also have nn.L1Loss (i.e. |x-y|).

		# in infinite class, classifier (not fullseq) setting, preds is simply a 1-dim vector of length batch_len. 
		# each pred is already the model's guess for the output

		Y_for_MSE = self.model.make_tensor(Y)
		Y_for_MSE = Y_for_MSE.float() # ???? voodoo for backprop with MSELossBackward. without it loss.backward() gets upset, despite fact that MSE seems happy as a clam
		loss = loss_fn(preds,Y_for_MSE)
		# loss = loss.float() # 
		num_correct = sum([self.model.tensor_double_to_out(p)==y for p,y in zip(preds,Y)])
		return loss, num_correct*1.0/len(Y) # need to multiply by 1.0 to get actual fraction and not literally 0 for every fraction under 1

	def fullseq_loss_infinite_classes(self,Y,preds,mask):
		raise NotImplementedError

	def classifier_loss_finite_classes(self,Y,probs,mask,lengths=None,suppress_zerobatch_prints=True):
		# lengths is throwaway but helpful
		# Y_for_CEL = self.model.make_tensor([[self.model.char2int[c] for c in s] for s in Y])
		Y_for_CEL = self.model.make_tensor(Y)
		# following line is in init but copied here for reference
		n_classes = probs.shape[1]
		counts = {c:Y.count(c) for c in range(n_classes)}
		weights = [(1.0*probs.shape[0]/counts[c] if counts[c]>0 else 0) for c in range(n_classes)]
		# print("training batch with weights:",weights)
		if not suppress_zerobatch_prints:
			if 0 in weights:
				print("!! have batch with no samples from some classes! counts are:",counts)
				print("\n\n!!!!!!!\n!! have batch with no samples from some classes! counts are:",
					counts,"\n!!!!!!!!\n\n",file=self.training_out)
		loss_fn = nn.CrossEntropyLoss(self.model.make_tensor(weights)) # expects x: N (input length) X C (num classes), y: N (with ints in 0 to C-1)        
		loss = loss_fn(probs,Y_for_CEL) # nn.crossentropyloss already normalises for number of samples
		num_correct = sum(torch.argmax(probs,1)==Y_for_CEL)
		return loss, num_correct*1.0/len(Y) # need to multiply by 1.0 to get actual fraction and not literally 0 for every fraction under 1
		# compute acc as num correct preds over total preds, i.e. total length of samples

	def fullseq_loss_finite_classes(self,Y,probs,mask,lengths,input_has_extra_BOS):
		if input_has_extra_BOS:
			# print("losing the BOS output")
			probs = probs[1:] # lose the BOS-location output, we're not training on that one
			lengths = [l-1 for l in lengths] # these lengths were computed from X
			# which already had BOS added to it, so fix
		# probs shape: max_len x len(X) x num output classes
		loss_fn = nn.CrossEntropyLoss() # not going to use custom made weights for seq2seq models, expect relatively fair number of each token overall?
		num_correct = 0
		loss = self.model.make_tensor(0.0)
		for i,l in enumerate(lengths):
			y_for_cel = self.model.make_tensor(Y[i]) # Y[i,:l] should be single array with length seq len, of ints (the int of the output for each token for this x)
			p = probs[:l,i,:]
			loss += loss_fn(p,y_for_cel)  # already normalises for length, so only have to divide total of all losses by number of samples at end
			num_correct += sum(torch.argmax(p,1)==y_for_cel)
		total_preds = sum(lengths)
		return loss/len(Y),  num_correct*1.0/total_preds# divide loss by num seqs and correct predictions by all predictions!!

	def get_batch_loss(self,prepped_batch):		
		# pad all to same length, but be wary not to add dummies to loss later
		X,Y,lengths,max_len,mask = prepped_batch
		# if classifier: Y should be single vector with length = len(X) (i.e. num samples), of ints (the int of the output for each X). 
		# else: Y is list of outputs for each X. each sample in Y one shorter than X counterpart because ignores BOS, which is an internal model thing

		# print("example x,y:\n",''.join(X[0]),"\n",''.join(Y[0]))
		if self.model.is_cuda():
			mask = mask.cuda()

		probs = self.model.pred(X,mask=mask,just_scores=True,
			already_longtensored=True,real_lengths=lengths) 
		# probs shape: if model.is_classifier: len(X) (i.e. num samples) X num output classes
		# 			                     else: max_len X len(X) X num output classes

		return self.loss_func(Y,probs,mask,lengths)
		
	def train_loss(self,loss):
		self.optim.zero_grad()
		if isinstance(loss,tuple): # got total loss, seq loss, attn loss:
			loss[0].backward()
		else:
			loss.backward()
		self.optim.step()
		return loss
			
	def train_seq(self,x): # gets a single seq before conversion to ints, and adds ends and starts, this one
		if isinstance(x,str):
			x = list(x)
		assert not self.model.non_token in x
		return self.train_single_x_y(self.add_start(x),self.add_end(x))
	

	def log_final_losses(self,train_batches,validation_batches,test_batches):
		losses, accs = {}, {}
		for n,batches in zip(["train","val","test"],[train_batches,validation_batches,test_batches]):
			# TODO figure out losses, accs
			b_losses, b_accs = [], []
			for b in batches:
				with torch.no_grad():
					l,a = self.get_batch_loss(b)
				b_losses.append(l) # l might be a single loss, or 3 losses: total_loss, seq_loss, attn_loss
				b_accs.append(a)
			if isinstance(b_losses[0],tuple):
				# b_losses is tuple of 3 losses - tot,seq,attn - from each batch
				# zip(*b_losses) is 3 lists - tot,seq,attn - of losses per batch
				losses[n] = tuple(map(mean,list(zip(*b_losses))))
			else:
				losses[n] = mean(b_losses)
			accs[n] = mean(b_accs)
			print(n,"loss:",careful_loss_str(losses[n]),end="",file=self.training_out)
			print(",\t\t acc:",loss_str(accs[n],with_e=False),file=self.training_out)
		self.append_vals(losses,accs,overwrite=True)

	def train_for_epochs(self,train_batches,validation_batches,test_batches,n_epochs,
		# add_starts=False,add_ends=False,
		newline_rep="\n",training_out=sys.stdout):
		prev_training_out = self.training_out
		self.training_out = training_out
		self.set_loss_func() # have to do this exactly when starting to train, 
		# so have time to mark if model is getting attentions to learn too
		# TODO: fix spaghetti code. dataset should be in charge of telling everyone whether 
		# also learning attention, not weirdly through model. and trainer should learn that (from dataset) on creation.
		print("beginning training",file=self.training_out)
		self.model.debprint("beginning training")
		self.model.debprint("am storing attention:",self.model.training_attn)
		t_loss,v_loss = 0, 0 # just for edge case prints

		overall_start = process_time()
		try:
			for i in range(n_epochs):
				start = process_time()
				t_loss, v_loss, t_acc, v_acc = self.train_epoch(train_batches,validation_batches)
				epoch_time = process_time() - start
				print("finished epoch",i,", that took:",epoch_time,"seconds.\
				 \nreached losses: train",careful_loss_str(t_loss),", val",careful_loss_str(v_loss),\
				  "\n reached accs: train",loss_str(t_acc,with_e=False),", val",loss_str(v_acc,with_e=False)\
				  ,file=self.training_out,flush=True)
				zero_loss = (0,0,0) if isinstance(t_loss,tuple) else 0
				losses = {"train":t_loss,"val":v_loss,"test":zero_loss}
				accs = {"train":t_acc,"val":v_acc,"test":0}
				decision = self.append_vals(losses,accs)
				self.plot_vals()
				if decision == STOP:
					print("!! stopped training - ",self.perfect_acc_count,"iters with validation accuracy == 1",file=self.training_out,flush=True)
					print("!! stopped training - ",self.perfect_acc_count,"iters with validation accuracy == 1",flush=True)
					break
				self.check_and_save(v_loss) 
		except KeyboardInterrupt:
			print("!! stopped training - interrupted by user",file=self.training_out,flush=True)
			print("!! stopped training - interrupted by user",flush=True) # flush so i know if that even went in
		overall_time = process_time() - overall_start
		print("\n\nfinished training, that took overall",overall_time,"s (",int(100*overall_time/3600)/100,"hrs)\nreached losses:\n",file=self.training_out)
		self.load_checkpoint() # get back to best saved model
		self.log_final_losses(train_batches,validation_batches,test_batches)
		self.dump_checkpoint() # dump with new info
		self.plot_vals(force=True)
		print("\n"*4,file=self.training_out) # leave some whitespace in case this gets trained more, so its easy to look at
		self.training_out = prev_training_out


	def train_epoch(self,train_batches,validation_batches):
		# shuffle(train_batches) # unlike RNNs, there's no state to be losing here
		self.model.debprint("training epoch with lr-scheduler:",self.lr_scheduler.state_dict())
		def loop_and_append(batches,pref,training=False):
			losses_l = self.model.metrics[pref+"_batch_losses"]
			losses_seq_l = self.model.metrics[pref+"_seq_batch_losses"]
			losses_attn_l = self.model.metrics[pref+"_attn_batch_losses"]
			accs_l = self.model.metrics[pref+"_batch_accs"]

			losses, accs = [], []
			for b in batches:
				loss,acc = self.get_batch_loss(b)
				multiloss = isinstance(loss,tuple)
				if training:
					self.train_loss(loss)
				losses.append(loss)
				accs.append(acc)
			if multiloss:
				all_losses = list(zip(*losses))
				assert len(all_losses)==3 # total, sequence, attention losses

				store_losses = [map(tonum,ls) for ls in all_losses]				
				tot,seq,attn = store_losses
				losses_l += tot
				losses_seq_l += seq
				losses_attn_l += attn
				mean_losses = tuple(mean(ls) for ls in all_losses)
			else:
				losses_l += [tonum(l) for l in losses]
				mean_losses = mean(losses)
			accs_l += [tonum(a) for a in accs]
			return mean_losses, mean(accs)

		def get_means_validation(batches):
			with torch.no_grad():
				return loop_and_append(batches,"val")
		def get_means_train(batches):
			self.model.train()
			res = loop_and_append(batches,"train",training=True)
			self.model.eval()
			return res
		# self.model.debprint("=======\nprocessing train\n==========")
		train_loss, train_acc = get_means_train(train_batches)
		# self.model.debprint("=======\nprocessing val\n==========")
		validation_loss, validation_acc = get_means_validation(validation_batches)
		# print("done processing\n============\n=========")
		self.lr_scheduler.step()# (validation_loss) exponentialLR scheduler does not track loss
		return train_loss, validation_loss, train_acc, validation_acc



def careful_loss_str(l):
	if isinstance(l,tuple):
		tot,seq,attn = l
		return "total: "+loss_str(tot,with_e=False)+", seq: "+loss_str(seq)+", attn: "+loss_str(attn,with_e=False)
	return loss_str(l)


def loss_str(loss,with_e=True):
	if not torch.is_tensor(loss):
		loss = torch.tensor(loss*1.0)
	return str(loss.item())+(" (e^loss = "+str(torch.exp(loss*1.0).item())+")" if with_e else "")

def tonum(v):
	if not torch.is_tensor(v): 
		return v
	return v.item()

def prep_batch(batch,non_token_index): # gets called by the dataset generators outside, 
	# as this is all batch 'preprocessing' stuff
	# that can be done in parallel in the cpus instead of wasting gpu time

	# print("received batch is:",batch)
	def _longtensor_seqs(seqs):
		seqs = torch.LongTensor(seqs) # this comes batch dim first
		return seqs.transpose(1,0) # transformer encoder/decoders expect: seq len X batch size X hidden dim

	# copied straight from TransformerClassifier, just can't pass the model in here 
	def generate_lengths_mask(lengths):
		mask = torch.zeros(len(lengths),max(lengths))
		for i,l in enumerate(lengths):
			mask[i,l:]=1
		mask = mask.bool()
		return mask

	X,Y = list(zip(*batch)) # X's might have BOS, Y's (if in seq2seq style) don't (and if not, are just a single int per sample anyway)
	lengths = [len(s) for s in X] # already includes non token BOS. hence len(s) = number of predictions for s, including empty prefix
	max_len = max(lengths)
	X = [x+[non_token_index]*(max_len-len(x)) for x in X] # padding

	assert len(set(len(s) for s in X))==1 # all should be same length. 
	mask = generate_lengths_mask(lengths) # hide paddings
	X = _longtensor_seqs(X) # same as: model._longtensor_seqs(X,already_ints=True), but without using the model cause we don't have it here (and trying to have it here would mean storing it in the dataset and upsetting pytorch)
	return X,Y,lengths,max_len,mask
	# X: max len X batch size (tensor)
	# Y: (if seq2seq:) tuple of batch_size outputs. outputs are either a tuple of ints of length len(input), or a single int.
	# if also training attention, outputs get nested into another tuple, containing also the target attention patterns, eg: Y= (out1,out2,...,outN) and outi = (actual_out_i,attn_patterns_i)
	# lengths: true length of each input sequence (tuple of length batch_size)
	# mask: tensor of shape max len X batch size. mask for the inputs i think
