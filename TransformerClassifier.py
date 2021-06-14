import torch
import torch.nn as nn
from Embeddings import FullEmbedding
from PytorchParts import TransformerEncoderLayer # lets me mess about in it a bit
from DatasetFunctions import OUT_INTS

class Scale(nn.Module):
	def __init__(self,temp=1):
		super(Scale,self).__init__()
		self.temp = temp
		self._main_temp = temp
	def set_temp(self,temp):
		self.temp = temp
	def reset_temp(self):
		self.temp = self._main_temp
	def forward(self,x):
		return x/self.temp        

def dummy_telm():
	return TransformerEncoderClassifier("a",[True],1,1,1,1)

class TransformerEncoderClassifier(nn.Module):
	def __init__(self,alphabet,out_classes,d_model,nheads,nlayers,ff,
		dropout=0.1,non_token='§',train_temp=1,positional_dropout=0,
		max_len=5000,positional_encoding_type='embed',loud=False,
		result_index=0,is_classifier=True,
		lang_name="",add_BOS_to_input=True,with_causal_mask=False,
		train_attn=False):
		super(TransformerEncoderClassifier,self).__init__()
		assert non_token not in alphabet
		self.add_BOS_to_input = add_BOS_to_input
		self.ignore_in_dict = [] # attributes to be ignored by the class_to_dict function used 
		self.lang_name = lang_name
		self.langs_history = []
		# whenever saving this module
		self.is_classifier = is_classifier # does it give single prob per seq or full set of probs per seq
		self.positional_encoding_type = positional_encoding_type
		self.with_causal_mask = with_causal_mask
		if self.with_causal_mask:
			raise NotImplementedError("havent implemented making the classifier force a causal mask for all inputs yet")
		self.d_model = d_model
		self.nheads = nheads
		self.ff = ff
		self.dropout = dropout
		self.nlayers = nlayers
		self.positional_dropout = positional_dropout
		self.internal_alphabet = list(alphabet)+[non_token]
		self.non_token = non_token
		self.max_len = max_len
		self.int2char = self.internal_alphabet
		self.char2int = {c:i for i,c in enumerate(self.int2char)}
		self.set_out_classes(out_classes)
		self.non_token_index = self.char2int[self.non_token]
		self.train_temp = train_temp
		assert result_index in [0,-1] # only real reasonable choices, anything else assumes sequence length >= 2
		self.result_index = result_index
#             nn.Softmax(dim=-1)) # don't want to do this immediately bc cross entropy loss
		self._loud = loud
		self.training_attn = False # until told otherwise
		self.start_metrics()
		self.finish_build()
		self.set_training_attn(train_attn) # has to happen after the transformer is made,
		# to go and set all of its layers appropriately
		self.debug = False
		
	def debprint(self,*a,**kw):
		if self.debug:
			print("DEBUG:",*a,**kw)

	def set_out_classes(self,out_classes):
		self.out_classes = out_classes
		if self.out_classes == OUT_INTS:
			self.tensor_double_to_out = lambda x:round(x.item())
			self.finite_classes = False
			self.ignore_in_dict.append("tensor_double_to_out")
		else:
			self.int2out = out_classes
			self.finite_classes = True	
			self.out2int = {c:i for i,c in enumerate(self.int2out)}			

	def set_training_attn(self,b):
		assert b in [True,False]
		for l in self.te.layers:
			l.training_attn(b) 
		self.training_attn = b

	def get_differentiable_last_attn(self,layer,head):
		l = self.te.layers[layer]
		distrs = l.attn_internals.differentiable_distributions
		# distrs shape: (bsz,num_heads,tgt_len,src_len)
		return distrs[:,head,:,:]
		# returns: batch size X seq_len (output) X seq_len (input)

	def start_metrics(self):
		self.metrics = {
			"reloads":[],
			"max_param_vals":[],
			"avg_param_vals":[],
			"train_batch_losses":[], "val_batch_losses":[],
			"train_batch_accs":[], "val_batch_accs":[],
			"train_minus_val_losses":[],
			"train_accs":[],"test_accs":[],"val_accs":[]}
		# store attention and seq training info
		for n in ["train","val","test"]:
			for sa in ["","_seq","_attn"]:
				for b in ["","_batch"]:
					self.metrics[n+sa+b+"_losses"] = []

	def change_task(self,lang_name,is_classifier,out_classes):
		if not hasattr(self,"old_metrics"):
			self.old_metrics = []
		is_cuda = self.is_cuda()           
		self.old_metrics.append(self.metrics)
		self.start_metrics()
		self.set_out_classes(out_classes) # set new number of out classes
		self.make_h2tag() # only then make new h2tag for them
		if is_cuda:
			self.cuda() # new h2tag might not have been made as cuda            
		self.is_classifier = is_classifier
		if not hasattr(self,"old_langs"):
			self.old_langs = []
		self.old_langs.append(self.lang_name)
		self.lang_name = lang_name


	def finish_build(self):
		self.make_modules()
		self.eval() # assume always eval unless set to train
		self.set_training_attn(self.training_attn) # make layers store attention, if its an attention-storing kind
		if torch.cuda.is_available():
			if self._loud:
				print("starting in cuda")
			self.cuda()
		else:
			if self._loud:
				print("starting in cpu")
			self.cpu()

	def is_cuda(self):
		return next(self.parameters()).is_cuda

	def make_tensor(self,args):
		res = torch.tensor(args)
		if self.is_cuda():
			res = res.cuda()
		return res

	def make_h2tag(self):
		h2tag_width = len(self.int2out) if self.finite_classes else 1

		self.h2tag = nn.Sequential(
			nn.Linear(self.d_model,h2tag_width),
			self.tempr_module)



	def make_modules(self):
		l = TransformerEncoderLayer(self.d_model,self.nheads,
			dim_feedforward=self.ff,
			dropout=self.dropout,activation='relu')
		self.te = nn.TransformerEncoder(l,self.nlayers,norm=None) 
		# uses layernorm inside by default, i think
		
		self.embedding = FullEmbedding(self.d_model,
							len(self.internal_alphabet),
							self.max_len,
							self.positional_encoding_type,
							self.positional_dropout)
		self.tempr_module = Scale(self.train_temp)
		self.make_h2tag()
		self.softmax = nn.Softmax(dim=-1)
		self.ignore_in_dict += ["te","embedding","tempr_module","h2tag","softmax"]

	def build_from_dict(self,source_dict):
		[setattr(self,p,source_dict[p]) for p in source_dict]
		self.set_out_classes(self.out_classes)
		self.finish_build()
		
	def loud(self):
		self._loud = True
	
	def quiet(self):
		self._loud = False

	def _longtensor_seqs(self,seqs,already_ints=False):
		if not already_ints:
			seqs = [[self.char2int[c] for c in s] for s in seqs] 
		seqs = torch.LongTensor(seqs) # this comes batch dim first
		return seqs.transpose(1,0) # transformer encoder/decoders expect: seq len X batch size X hidden dim


	def _embed_seqs(self,seqs,already_ints=False,already_longtensored=False,real_lengths=None):
		# seqs shape now: seq len X batch size
		if not already_longtensored:
			seqs = self._longtensor_seqs(seqs,already_ints=already_ints)
		if self.is_cuda():
			seqs = seqs.cuda()
		return self.embedding(seqs,real_lengths=real_lengths)

	def select_classifier_res(self,res,real_lengths):
		if self.result_index == -1:
			if None is real_lengths:
				res = res[-1,:,:] # assumes all are same length, and taking last
			else:
				out_dim = res.shape[-1]
				ll = [[l-1]*out_dim for l in real_lengths]
				res = res.gather(0,self.make_tensor(ll).long().view(1,-1,out_dim))[0]
				# ^ pytorch version of: res = [res[l-1,i,:] for i,l in enumerate(real_lengths)], which is what we want 
				# but then it would be a list not a tensor and sticking it back together might be hard?
		else: # self.result_index == 0:
			res = res[0,:,:] # we only care about result in first/last (parameter) location because we're doing classification            
		return res

	def forward(self,seqs,mask=None,already_ints=False,real_lengths=None,already_longtensored=False):
		# mask is for src_key_padding_mask: notes which parts of the input sequence are padding and should be ignored
		seqs = self._embed_seqs(seqs,already_ints=already_ints,
					real_lengths=real_lengths,already_longtensored=already_longtensored)
		res = self.te(seqs,src_key_padding_mask=mask) # gives (encoder seq len) X batch size X hidden_dim
		# assert self.result_index in [0,1] # done earlier but be aware its for here
		if self.is_classifier:
			res = self.select_classifier_res(res,real_lengths)
		if self._loud:
			print("te forward res: (shape:",res.shape,") (should be batch size X hidden dim if classifier, else longest seq X batch size X hidden)")
			print(res)
		return res #  batch size X hidden_dim if classifier else seq len x batch size x hidden dim
	

	def pred(self,seqs,mask=None,just_scores=False,already_ints=False,
								real_lengths=None,already_longtensored=False): 
	# (generally, pass this seqs that already have non-token at start)
		res = self.h2tag(self(seqs,mask=mask,already_ints=already_ints,
					real_lengths=real_lengths,already_longtensored=already_longtensored)) 
		#  batch_size X num output tokens if classifier, else seq len X batch size X num output tokens?
		
		if (not just_scores) and self.finite_classes:
			res = self.softmax(res) 
			# if classifier: batch_size X num output tokens
			# else (ie fullseq): batch_size X seq_len X num output tokens
		
		if not self.finite_classes: # lose num output tokens dim, there's only one
			assert res.shape[-1] == 1
			if self.is_classifier:
				res = res[:,0] # batch_size
			else:
				res = res[:,:,0] # batch_size X seq len

		if self._loud:
			print("in pred, res shape:",res.shape,\
				"\nshould be (seq len,batch size,num classes), but \
				drop num classes if not finite_classes, and drop seq len if classifier and not fullseq")
			print(res)
		return res

	def get_distrs_and_outs(self,s):
		for l in self.te.layers:
			l.set_keeps(True)
		c = self.classify(s)
		all_attn_internals = {i:l.attn_internals for i,l in enumerate(self.te.layers)} # might need a tolist if this breaks
		# shape of scores in each layer is batch_size X num_heads X tgt_len X src_len

		all_vals = {i:l.stored_out for i,l in enumerate(self.te.layers)}
		all_vals[-1] = self.te.layers[0].stored_in
		# shape of vals in each layer is seq len X batch size X embed dim
		
		for l in self.te.layers: # get rid of these now
			l.set_keeps(False)
			l.clear_keeps()

		# lose pointless batch dim cause only classifying one sequence
		for i in all_attn_internals:
			all_attn_internals[i].distributions = all_attn_internals[i].distributions[0]
			all_attn_internals[i].scores = all_attn_internals[i].scores[0]
		# now shape of scores in each layer is num_heads X tgt_len X src_len 
		all_vals = {i:all_vals[i][:,0,:] for i in all_vals}
		# and shape of vals in each layer is seq len X embed dim
		return all_attn_internals, all_vals, c


	def classify(self,s,temp=1): # honestly no reason to be messing with temp anymore but doesn't immediately hurt to have it around (in case want to go back to LM model)
		self.tempr_module.set_temp(temp)
		if isinstance(s,str):
			s = list(s)
		if self.add_BOS_to_input:
			s = [self.non_token] + s # don't forget this, it's added to all the samples too
		probs = self.pred([s]) 
		# if finite out tokens: 1 X num_out_tokens if classifier else len(s) X num_out_tokens. 
		# if infinite out tokens: just 1 if classifier and else len(s)
		to_out = (lambda x:self.int2out[torch.argmax(x).item()]) if self.finite_classes else self.tensor_double_to_out

		if self.is_classifier:
			res = to_out(probs)
		else:
			res = [to_out(p) for p in probs]
			if self.add_BOS_to_input:
				res = res[1:]
		self.tempr_module.reset_temp()
		return res

	def generate_lengths_mask(self,lengths):
		mask = torch.zeros(len(lengths),max(lengths))
		for i,l in enumerate(lengths):
			mask[i,l:]=1
		mask = mask.bool()
		return mask