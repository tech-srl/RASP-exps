from torch.utils.data import Dataset
import string
import random
from MinilangFunctions import Minilang_Maker, vary_alpha_and_bos, PairwiseAsFull
from DatasetFunctions import makesets_generator, GetLangs, alternating_generator, \
							completely_random, SampleGenerator, DeterminisedFunc
from DatasetFunctions import randoms_generator as _randoms_generator
from copy import deepcopy
from collections import Counter
from inspect import signature

# some general attn patterns
def prev_positions(s):
	return [ [ j<i for j in range(len(s))] for i in range(len(s)) ]

def smaller_tokens(s):
	return [ [ cj<ci for cj in s ] for ci in s ]

minilangs = GetLangs({})

############### langs officially start ################

##
# some common functions shared by langs


def mark_firsts(s):
	res = []
	seen = set()
	for c in s:
		res.append(c not in seen)
		seen.add(c)
	return res

##

# sort functions - have to be outside else it complains about local functions


def sort_classifier(s):
	return sorted(list(s))

def sort_is_before(qi,ki,s):
	return (s[ki]<s[qi]) or ((s[ki]==s[qi]) and (ki<qi))

def sort_grabber_attn(s,older_heads):
	n = len(s)
	is_before_attn = older_heads["is_before"]
	target_locs = [ befores.count(True) for befores in is_before_attn ]
	res = [ [qi==target_locs[ki] for ki in range(n)] for qi in range(n) ]
	return res

def sort_main(hp,alpha=string.ascii_lowercase): 
	# will have tokens for count = 1, 2, ..., max_count. anything greater 
	# than max_count just reports as that
	gen = hp.randoms_generator(alpha)
	out_classes = alpha

	hp.tgt_attns.add_head(0,0,pairwise_fun=sort_is_before,ref_name="is_before") # first layer, first head
	hp.tgt_attns.add_head(1,0,fun=sort_grabber_attn,look_at_bos_too=False) # using pairwise fun for this ends up n^3 cause have to recompute whole row of is_before attention for each pair!

	is_classification_task = False
	hp.set_lang_specifics(alpha,out_classes,gen,sort_classifier,is_classification_task)
	return hp.make_minilang()

minilangs = vary_alpha_and_bos(minilangs,"sort",sort_main)

# reverse functions

def reverse(s):
	return s[-1:0:-1]+s[0:1]

def reverse_main(hp,alpha=string.ascii_lowercase):
	gen = hp.randoms_generator(alpha)
	out_classes = alpha
	classifier = reverse
	is_classification_task = False
	hp.set_lang_specifics(alpha,out_classes,gen,classifier,is_classification_task)
	return hp.make_minilang()

minilangs = vary_alpha_and_bos(minilangs,"reverse",reverse_main)

# histogram functions

class HistClassifier:
	def __init__(self,max_count):
		self.max_count = max_count
	def __call__(self,s):
		c = Counter(s)
		c = {v:min(c[v],self.max_count) for v in c}
		res = [c[v] for v in s]
		return res
	

def histograms_main(hp,alpha=string.ascii_lowercase): 
	# will have tokens for count = 1, 2, ..., max_count. anything greater 
	# than max_count just reports as that
	gen = hp.randoms_generator(alpha)
	max_count = 10
	out_classes = list(range(max_count+1))

	classifier = HistClassifier(max_count)
	is_classification_task = False
	hp.set_lang_specifics(alpha,out_classes,gen,classifier,is_classification_task)
	return hp.make_minilang()

minilangs = vary_alpha_and_bos(minilangs,"hist",histograms_main)

# double histogram functions

class Hist2Classifier:
	def __init__(self,max_count):
		self.max_count = max_count
	def __call__(self,s):
		hist = Counter(s)
		uniques = list(set(s))
		count_per_unique = [hist[c] for c in uniques]
		hist2 = Counter(count_per_unique) # how many tokens there are for each token-count 
		hist2 = {c:min(hist2[c],self.max_count) for c in hist2}
		res = [hist2[hist[c]] for c in s]
		return res

def hist2_is_equal(qi,ki,s):
	return s[qi]==s[ki]

def hist2_is_prev_equal(qi,ki,s):
	return s[qi]==s[ki] and ki<qi

def hist2_representative_with_same_count(s):
	counts = Counter(s)
	is_first = mark_firsts(s)
	def qiki_res(qi,ki):
		return (counts[s[ki]]==counts[s[qi]]) and is_first[ki]
	n = len(s)
	return [[qiki_res(qi,ki) for ki in range(n)] for qi in range(n)]

def hist2_main(hp,alpha=string.ascii_lowercase): 
	# will have tokens for count = 1, 2, ..., max_count. anything greater 
	# than max_count just reports as that
	gen = hp.randoms_generator(alpha)
	max_count = 10
	out_classes = list(range(max_count+1))

	hp.tgt_attns.add_head(0,0,pairwise_fun=hist2_is_equal,look_at_bos_too=True) # to allow computing width in one head, need to look at BOS too
	hp.tgt_attns.add_head(0,1,pairwise_fun=hist2_is_prev_equal)
	hp.tgt_attns.add_head(1,0,fun=hist2_representative_with_same_count)

	classifier = Hist2Classifier(max_count)
	is_classification_task = False
	hp.set_lang_specifics(alpha,out_classes,gen,classifier,is_classification_task)
	return hp.make_minilang()

minilangs = vary_alpha_and_bos(minilangs,"hist2",hist2_main)

# sort by freq functions

class SortByFreqClassifier:
	def __init__(self,non_token):
		self.non_token = non_token
	def __call__(self,s):
		a = Counter(s)
		tokens = list(a.keys())
		firstloc = {c:s.find(c) for c in tokens} # maintain order between tokens with same frequency,
		# for clarity. (instead of trying to learn order of list(a.keys()) and then also recognise 
		# that in the attn. distribution somehow)
		pad = len(s)-len(tokens)
		tokens = sorted(tokens,key=lambda c:(a[c],-firstloc[c]),reverse=True)
		return tokens + [self.non_token]*pad 

def sort_by_freq__is_equal(qi,ki,s):
	return s[qi]==s[ki]

def sort_by_freq__is_prev_equal(qi,ki,s):
	return s[qi]==s[ki] and ki<qi

def sort_by_freq__comes_earlier(s):
	# tokens before self are those with greater count than self, 
	# or equal count to self but earlier repr. only take the representative of those tokens though!
	is_first = mark_firsts(s) # representative of each unique token is its first appearance
	counts = Counter(s)
	max_input_len = 201 # account for BOS too
	score = [counts[c]-((max_input_len+1)*int(not is_first[i])) for i,c in enumerate(s)] # penalty should be higher than any possible count
	# score: higher scores according to frequency, but:
	#  make sure to knock out all those that are not representatives
	def qiki_res(qi,ki):
		if score[ki] > score[qi]:
			return True
		elif score[ki] == score[qi]:
			return ki < qi # tiebreaker: order in sequence
		return False
	n = len(s)
	return [[qiki_res(qi,ki) for ki in range(n)] for qi in range(n)]

def sort_by_freq__is_focus(s,older_heads):
	comes_before = older_heads["comes_before"]
	target_position = [sum(l) for l in comes_before]
	n = len(s)
	return [[qi == target_position[ki] for ki in range(n)] for qi in range(n)]

def sort_by_freq_main(hp,alpha=string.ascii_lowercase): 
	out_classes = hp.add_non_token(alpha)	
	gen = hp.randoms_generator(alpha)

	# Now we define the target attention patterns.
	# A note to the possibly confused visitor:
	# In the very first layer, all attention patterns we can define in RASP and in transformers are truly pairwise relations -
	# the only information we have for each token is itself and its position.
	# From the second layer and up however, we have mixed information between positions, and the next attention pattern
	# describes a binary relation that is no longer `pairwise' with respect to the input sequence alone (though it is of
	# course pairwise with respect to the information that has been gathered at the input sequence)
	# Anyway for this reason the heads in layer 0 are convenient to describe with a pairwise function,
	# Whereas the others sometimes can only be described with a function that looks at the whole sequence.
	# Ultimately, note that we are here describing a program that is legal wrt the information flow of a transformer

	hp.tgt_attns.add_head(0,0,pairwise_fun=sort_by_freq__is_equal,look_at_bos_too=True) # need width of this one for histogram
	hp.tgt_attns.add_head(0,1,pairwise_fun=sort_by_freq__is_prev_equal)


	hp.tgt_attns.add_head(1,0,fun=sort_by_freq__comes_earlier,
									look_at_bos_too=True,ref_name="comes_before") # need to compute the width of this one, so only play with BOS and have it look at BOS too
	hp.tgt_attns.add_head(2,0,fun=sort_by_freq__is_focus)

	classifier = SortByFreqClassifier(hp.non_token)
	is_classification_task = False
	hp.set_lang_specifics(alpha,out_classes,gen,classifier,is_classification_task)
	return hp.make_minilang()

minilangs = vary_alpha_and_bos(minilangs,"sort_by_freq",sort_by_freq_main)

# dycki languages functions

class DyckiGen(SampleGenerator):
	def __init__(self,shortlen,longlen,pairs,neutrals):
		super(DyckiGen,self).__init__()
		self.shortlen, self.longlen = shortlen, longlen
		self.pairs, self.neutrals = pairs, neutrals
	def __call__(self):
		n = random.randint(self.shortlen,self.longlen)
		res = ""
		while len(res)<n:
			i = random.choice(list(range(len(res)+1)))
			start, end = res[:i], res[i:]
			if random.random()<0.3:
				insert = random.choice(self.neutrals)
			else:
				insert = random.choice(self.pairs)
			res = start + insert + end
		return res

class DyckiStartGen(SampleGenerator):
	def __init__(self,balanced_gen,alpha):
		super(DyckiStartGen,self).__init__()
		self.balanced_gen = balanced_gen
		self.alpha = alpha
	def __call__(self):		
		b = self.balanced_gen()
		n = random.choice(list(range(len(b)+1)))
		start = b[:n]
		end = completely_random(self.alpha,len(b)-n)
		return list(start) + list(end)

class DyckiPTFClassifier:
	def __init__(self,PTF,pair_tuples):
		self.PTF = PTF
		self.pair_tuples = pair_tuples
		self.openers, self.closers = list(zip(*pair_tuples))
		self.parens = self.openers + self.closers
	def __call__(self,s):
		stack = []
		res = []
		P,T,F = self.PTF
		for c in s:
			# once broken, stay broken
			if (res) and (res[-1]==F):
				res.append(F)
				continue
			if not c in self.parens:
				res.append(P if stack else T)
			elif c in self.openers:
				stack.append(c)
				res.append(P)
			elif c in self.closers:
				if (stack) and ((stack[-1],c) in self.pair_tuples):
					stack = stack[:-1]
					res.append(P if stack else T)
				else:
					res.append(F)
		return res

all_pairs = ["()","{}","[]","<>","\\/","`'","ab","xy"]
def dycki_ptf(hp,i,neutrals="a"):
	assert i<=len(all_pairs)
	pairs = all_pairs[:i]
	
	parens = "".join(pairs)
	assert True not in [(c in parens) for c in neutrals]

	alpha = parens + neutrals

	balanced_gen = DyckiGen(hp.shortlen,hp.longlen,pairs,neutrals)
	random_gen = hp.randoms_generator(alpha)
	balanced_start = DyckiStartGen(balanced_gen,alpha)

	generators = [balanced_gen,balanced_start,random_gen]
	gen = alternating_generator(generators) 
	# get some unbalanced ones in there as well, to recognise when a sequence can no longer be balanced ever again

	PTF = ["P","T","F"]
	pair_tuples = list(map(tuple,pairs)) 
	classifier = DyckiPTFClassifier(PTF,pair_tuples)
	is_classification_task = False
	hp.set_lang_specifics(alpha,PTF,gen,classifier,is_classification_task)
	return hp.make_minilang()

def add_dycki(minilangs,i):
	minilangs["dyck"+str(i)+"_ptf"] = lambda : dycki_ptf(Minilang_Maker(add_bos=False),i)
	minilangs["dyck"+str(i)+"_ptf_bos"] = lambda : dycki_ptf(Minilang_Maker(add_bos=True),i)
	return minilangs

for i in range(1,len(all_pairs)+1):
	minilangs = add_dycki(minilangs,i)