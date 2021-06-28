from torch.utils.data import Dataset
import string
import random
from DatasetFunctions import makesets_generator, GetLangs, alternating_generator, \
							completely_random, SampleGenerator, DeterminisedFunc
from DatasetFunctions import randoms_generator as _randoms_generator
from copy import deepcopy
from collections import Counter
from inspect import signature

##### all kinds of would-be local functions implemented as classes so they can be passed around in multiprocessing workers for pytorch's dataset loaders ===

class ClassifierWithAttn:
	def __init__(self,classifier,tgt_attns):
		self.classifier = classifier
		self.tgt_attns = tgt_attns
	def __call__(self,s):
		attns_dict = self.tgt_attns.make_heads_for_s(s)
		return (self.classifier(s), attns_dict)

class AsStr(SampleGenerator):
	def __init__(self,f):
		super(AsStr,self).__init__()
		self.f = f
		self.determinised = isinstance(f,SampleGenerator) and f.determinised
	def __call__(self,*a,**kw):
		return "".join(self.f(*a,**kw))

class AddPref(SampleGenerator):
	def __init__(self,f,pref):
		super(AddPref,self).__init__()
		self.f = f
		self.pref = pref
		self.determinised = isinstance(f,SampleGenerator) and f.determinised
	def __call__(self,*a,**kw):
		return self.pref + self.f(*a,**kw)

class WrappedOutput:
	def __init__(self,output_fun,add_bos,using_tgt_attns,output_as_string,pref):
		self.output_fun = output_fun
		self.add_bos = add_bos
		self.using_tgt_attns = using_tgt_attns
		self.output_as_string = output_as_string
		self.pref = pref
	def __call__(self,s):
		if self.add_bos:
			s = s[1:] # remove the bos from input
		res = self.output_fun(s)
		if self.using_tgt_attns:
			res_with_attn = res # output_seq, expected attentions
			res = res_with_attn[0] # just the seq
		if self.output_as_string:
			res = "".join(res)
		else:
			res = list(res)
		res = self.pref + res
		if self.using_tgt_attns:
			res = (res,res_with_attn[1]) # bring it back in
		return res
 
###################

class Minilang_Maker:
	def __init__(self,name="lang",tr=5e4,va=1e3,te=1e3,non_token="§",
					shortlen=1,longlen=100,add_bos=False,
					use_tgt_attns=False):
		self.tr = int(tr)
		self.va = int(va)
		self.te = int(te)
		self.name = name # just for helpful prints in error messages
		self.non_token = non_token
		self.shortlen = shortlen
		self.longlen = longlen
		self.add_bos = add_bos
		self.output_as_string = False
		self.input_as_string = False
		self.called_randoms_with = None
		self.tgt_attns = AttnTargetMakers()
		self.use_tgt_attns = use_tgt_attns

	def add_non_token(self,alpha):
		if self.non_token in alpha:
			return alpha
		if isinstance(alpha,str) and isinstance(self.non_token,str) and len(self.non_token)==1:
			return alpha+self.non_token
		return list(alpha) + [self.non_token]

	def make_pref(self,as_str):
		bos = self.non_token if as_str else [self.non_token]
		return bos if self.add_bos else (bos*0)

	def wrap_output(self,output_fun):
		pref = self.make_pref(self.output_as_string)
		return WrappedOutput(output_fun,self.add_bos,
			self.using_tgt_attns,self.output_as_string,pref)

	def wrap_input(self,input_fun):
		asstr_res = AsStr(input_fun) if self.input_as_string else input_fun
		pref = self.make_pref(self.input_as_string)
		withpref_res = AddPref(asstr_res,pref)
		return DeterminisedFunc(withpref_res)

	def set_lang_specifics(self,alpha,out_classes,generator,classifier,
							is_classification_task):
		self.alpha = alpha
		self._out_classes = out_classes 
		self.generator = generator
		self.classifier = classifier
		self.is_classification_task = is_classification_task
		self.tgt_attns.set_bos(self.add_bos)
		classifier_with_attn = ClassifierWithAttn(classifier,self.tgt_attns)

		self.using_tgt_attns = False
		if self.use_tgt_attns: 
			if self.tgt_attns.empty():
				raise Exception("No attentions given for minilang "+self.name)
			self.classifier = classifier_with_attn
			self.using_tgt_attns = True

	def randoms_hps_not_changed(self):
		if not None is self.called_randoms_with:
			a = deepcopy(self.called_randoms_with)
			_ = self.randoms_generator(self.alpha)
			b = self.called_randoms_with
			return a==b
		return True

	def make_minilang(self,dfa=None):
		assert self.non_token not in self.alpha
		assert self.randoms_hps_not_changed()
		self.out_classes = self.add_non_token(self._out_classes) if self.add_bos else self._out_classes
		self.input_as_string = isinstance(self.alpha,str)
		self.output_as_string = isinstance(self.out_classes,str)
		datasets = makesets_generator(self.wrap_input(self.generator),self.wrap_output(self.classifier),
												self.non_token,self.tr,self.va,self.te)
		return datasets, MinilangMetas(self,dfa)

	def randoms_generator(self,alpha):
		self.called_randoms_with = deepcopy([alpha,self.non_token,self.shortlen,self.longlen])
		# don't add the BOS here: that happens uniformly to generator in wrap_input.
		# otherwise, will get double BOS. also, if have multiple types of generators through alternating_generator,
		# then the random one will have double BOS and all the others won't, and the transformer will pick up
		# on that
		return _randoms_generator(alpha=alpha,non_token=self.non_token,
								shortlen=self.shortlen,longlen=self.longlen)


class MinilangMetas:
	def __init__(self,ml,source_dfa):
		self.alpha = ml.alpha
		self.non_token = ml.non_token
		self.out_classes = ml.out_classes
		self.is_classification_task = ml.is_classification_task
		self.using_tgt_attns = ml.using_tgt_attns
		self.dfa = source_dfa

##### head targets, including some defaults ####

class PairwiseAsFull:
	def __init__(self,pairwise):
		self.pairwise = pairwise
		self.pass_olders = num_params(pairwise)==4 # if 3, just takes q,k,s. otherwise, expects older heads too
	def __call__(self,s,older_heads):
		if self.pass_olders:
			return [ [ self.pairwise(q,k,s,older_heads) for k in range(len(s))] 
							for q in range(len(s))]
		else:
			return [ [ self.pairwise(q,k,s) for k in range(len(s))] 
							for q in range(len(s))]

def num_params(f):
	sig = signature(f)
	return len(list(sig.parameters))

class AttnTargetMakers:
	def __init__(self):
		self.head_funs = {}
		self.look_at_bos_too = {}
		self.ref_names = {}
	def set_bos(self,b):
		self.with_bos = b
	def add_head(self,l,h,fun=None,pairwise_fun=None,look_at_bos_too=True,ref_name=None):
		assert (len(set([fun,pairwise_fun]))>1) and (None in [fun,pairwise_fun]) # exactly one is None
		if not None is pairwise_fun:
			fun = PairwiseAsFull(pairwise_fun)
		if not l in self.head_funs:
			self.head_funs[l] = {}
		self.head_funs[l][h] = fun
		self.look_at_bos_too[(l,h)] = look_at_bos_too
		if not None is ref_name:
			self.ref_names[(l,h)] = ref_name
	def empty(self):
		return len(self.head_funs)==0
	def bos_wrap(self,head,l,h):
		if self.with_bos:
			# add bos row:
			bos_too = self.look_at_bos_too[(l,h)]
			bos_row = [True] + [False]*len(head[0]) # bos looks exactly at itself
			head = [[bos_too]+r for r in head] # everyone else has uniform decision about BOS
			head = [bos_row] + head
		return head
	def make_heads_for_s(self,s):
		stored_heads = {} # allows later-layer heads to use older heads, 
		# just to save sample generation time mostly (reduce repetition)
		
		def make_head(l,h):
			f = self.head_funs[l][h]
			if num_params(f)==2: # expects not just s, but also older heads
				res = f(s,stored_heads)
			else:
				res = f(s)
			return self.bos_wrap(res,l,h)

		def remove_bos(target):
			if self.with_bos:
				target = [t[1:] for t in target[1:]] # remove first column and first row: these are BOS values
			return target

		res = {}
		for l in self.head_funs:
			if l not in res:
				res[l] = {}
			for h in self.head_funs[l]:
				target_head = make_head(l,h)
				res[l][h] = target_head
				if (l,h) in self.ref_names:
					stored_heads[self.ref_names[(l,h)]] = remove_bos(target_head) # own functions always assume no bos interference

		return res

		# pytorch dataloader has leak on cpu with num_workers>0 if it passes lists,
		# but is okay with passing tensors? ---.---  https://github.com/pytorch/pytorch/issues/13246   
		### -- didn't fix problem :(. must be something else


####### prep langs with variations ########
def vary_alpha_and_bos(minilangs,lang_name,lang_make,alpha_lengths=[10,26,52,100]):
	def add_minilang(minilangs,bos,attns,alpha):
		bos_suff = "_bos" if bos else ""
		attns_suff = "_with_attns" if attns else ""
		name = lang_name+"_"+str(l)+bos_suff+attns_suff
		minilangs[name] = lambda : lang_make(Minilang_Maker(name=name,add_bos=bos,use_tgt_attns=attns),
											alpha=alpha)
		# add debug minilang
		return minilangs
		# have to do this in a function to keep all the params to Minilang_Maker in the lambda 
		# constant once set (scoping stuff)
	for l in alpha_lengths:
		assert len(all_tokens) >= l
		for bos in [True,False]:
			for attns in [True,False]:
				minilangs = add_minilang(minilangs,bos,attns,all_tokens[:l])
	return minilangs


all_tokens = string.ascii_letters + string.digits + string.punctuation + string.whitespace