from TransformerClassifier import TransformerEncoderClassifier, dummy_telm
from TransformerClassifierTrainer import TransformerEncoderClassifierTrainer
from LoadTransformer import load_any_transformer_for_lang
from DatasetFunctions import IntsDataset
from Minilangs import minilangs 
import torch
import argparse
import ast
from Helper_Functions import prepare_txtfile_if_not_ready, mean, load_pytorch_model, timestamp
import numpy as np
from time import process_time
import sys
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite',action='store_true',help=
	'overwrite existing transformer with these parameters')


parser.add_argument('--lang',type=str,default="reverse_26") 

parser.add_argument('--batch-size',type=int,default=50)
parser.add_argument('--max-len',default=5e3)

parser.add_argument('--d-model',type=int,default=128)
parser.add_argument('--nheads',type=int,default=2)
parser.add_argument('--nlayers',type=int,default=2)
parser.add_argument('--ff',type=int,default=256)
parser.add_argument('--dropout',type=float,default=0.01)
parser.add_argument('--with-causal-mask',action='store_true')
parser.add_argument('--initial-lr',type=float,default=0.000001) 
parser.add_argument('--add-BOS',action='store_true')
parser.add_argument('--optim',default="adam",choices=['adam','sgd'])
parser.add_argument('--positional-encoding-type',default='sin',choices=['embed','sin'])
parser.add_argument('--result-index',type=int,default=0,help='which index of output vectors to use for result',choices=[0,-1])
parser.add_argument('--reset-patience',type=int,default=10,help=
	"number of epochs without improvement before trainer resets weights of model")
parser.add_argument('--n-epochs',type=float,default=250)
parser.add_argument('--gamma',type=float,default=0.9)
parser.add_argument('--attn-loss-c',type=float,default=10)

parser.add_argument('--subfolder',type=str,default="")
parser.add_argument('--index',type=int,default=1,help=
	"index lms so can make several transformers with same hyperparams without overwriting each other")

parser.add_argument('--overfit',action='store_true')
parser.add_argument('--tiny',type=int,default=0)
parser.add_argument('--crop-train',type=int,default=0)
parser.add_argument('--crop-all',type=int,default=0)
parser.add_argument('--no-timestamp',action='store_true')
parser.add_argument('--perfect-acc-cut',default=3)


def get_args():
	args = parser.parse_args()
	assert not (args.overfit and (args.tiny>0) and (args.crop_train>0 or args.crop_all>0))
	args.max_len  = int(args.max_len)
	args.n_epochs = int(args.n_epochs)
	return args

def nicelanglist():
	l = list(minilangs.keys())
	starts = sorted(list(set(n.split("_")[0] for n in l)))
	ends = {s:sorted([n for n in l if (n.startswith(s+"_") or (n==s))]) for s in starts}
	res = ""
	for s in starts:
		res+="\n=======\n"+s+":\n\t"+str(ends[s])
	return res

def get_lang(args):
	if args.lang in minilangs:
		return minilangs[args.lang]
	else:
		print("unrecognised langauge. languages are:",nicelanglist(),"\n,asked for:",args.lang)    
		exit()

def apply_overfit_or_tiny(args,datasets):
	if args.overfit:
		print("using sgd (instead of",args.optim,") for overfit run")
		args.optim = "sgd"
	if args.overfit or (args.tiny>0):
		l = 1 if args.overfit else args.tiny
		d = datasets["train"]
		d.crop(l) # for classification (as opposed to lm), 10 seems like a reasonable number of samples for overfit?
		if args.overfit:
			for n in datasets:
				datasets[n] = d
	elif args.crop_train > 0:
		datasets["train"].crop(args.crop_train)
	if args.crop_all > 0:
		for n in datasets:
			datasets[n].crop(args.crop_all)
	return args, datasets

def print_lengths(datasets,name,f=sys.stdout):
	print("\n\nlengths info for dataset:",name,file=f)
	for n in datasets:
		print("checking average length of dataset:",n,flush=True)
		start = process_time()
		d = datasets[n]
		print(n,"set has",len(d),"samples, of average length:",sum(len(x) for x,y in d)/len(d),file=f) 
		print("first sample is:",d[0],file=f,flush=True)
		print("finished, that took:",process_time()-start,flush=True)
	print("\n\n",file=f)

def print_overlaps(datasets,name,f=sys.stdout):
	print("\n\noverlap info for dataset:",name,file=f)
	counts = {n:Counter(str(s) for s in datasets[n]) for n in ["train","valid","test"]}
	def contained(n1,n2):
		print("checking how many of",n1,"contained in",n2,flush=True)
		start = process_time()
		c = sum([counts[n1][s] for s in counts[n1] if s in counts[n2]])
		print(c,"of",n1,"set is in",n2,"set",file=f)
		print("finished, that took:",process_time()-start)
	contained("valid","train")
	contained("test","train")
	contained("valid","test")
	print("\n\n",file=f)

def print_state_visits(datasets,dfa,prints_file):
	if None is dfa:
		return
	def dict_sorted_str(d,truekeys):
		return ", ".join(str(k)+": "+str(d[k]) for k in sorted(truekeys))
	print("samples taken from dfa. dfa state visitation distributions:",file=prints_file)
	for n in datasets:
		total_len = sum(len(x) for x,y in datasets[n])
		counts = dfa.state_counts((x for x,y in datasets[n]))
		print(n,"state visits: {",dict_sorted_str(counts,dfa.Q),"},\n\t as distribution:",
				dict_sorted_str({q:(counts[q]/total_len) for q in dfa.Q},dfa.Q),file=prints_file)


def print_pos_and_neg(datasets,name,f=sys.stdout):
	print("\n\npos/neg info for dataset:",name,file=f)
	for n in datasets:
		d = datasets[n]
		print(n,"set has",sum((1 if y else 0) for x,y in d),"positive samples and",
			sum((0 if y else 1) for x,y in d),"negative samples",file=f)
	print("\n\n",file=f)


def get_model_folder(args):
	hyperparams_str = "_".join(str(a) for a in (["L"+str(args.nlayers),
												 "H"+str(args.nheads),
												 "D"+str(args.d_model),
												 "FF"+str(args.ff),
												 "drop"+str(args.dropout),
												 "epochs"+str(args.n_epochs),
												 "pos-"+str(args.positional_encoding_type),
												 "lr-start"+str(args.initial_lr),
												 "gamma"+str(args.gamma),
												 "max-len"+str(args.max_len),
												 "res-index"+str(args.result_index)]+\
											['withCausalMask']*args.with_causal_mask+\
											['withBOS']*args.add_BOS  ))
	res = "lms/"+((args.subfolder+"/") if args.subfolder else "")+args.lang+"/"+\
							hyperparams_str + "_" + str(args.index)
	if not args.no_timestamp:
		res += "_"+timestamp()
	return res

def get_prints_files(model_folder,args,datasets):
	training_prints_file = model_folder + "/training_prints.txt"
	dataset_prints_file = model_folder + "/dataset_prints.txt"
	prepare_txtfile_if_not_ready(training_prints_file,clear=args.overwrite)
	prepare_txtfile_if_not_ready(dataset_prints_file,clear=args.overwrite)
	return training_prints_file, dataset_prints_file


def print_train_info(training_prints_file,dataset_prints_file,model_folder,args,ds_meta,datasets,batches):
	with open(dataset_prints_file,"a") as f:
		print("\noverall dataset info:\n",file=f)
		print_lengths(datasets,args.lang,f)
		print_overlaps(datasets,args.lang,f)
		if ds_meta.is_classification_task:
			print_pos_and_neg(datasets,args.lang,f)
		if not None is ds_meta.dfa:
			print_state_visits(datasets,ds_meta.dfa,f)
			ds_meta.dfa.draw_nicely(filename=model_folder+"/source_dfa",orig_state_names=True)
		print("\n\nalphabet size is:",len(ds_meta.alpha),"   alphabet:\n",ds_meta.alpha,file=f)
		print("output alphabet size is:",len(ds_meta.out_classes),file=f)		
	with open(training_prints_file,"a") as f:
		print("\n\nmodel info/training args\n",file=f)
		print("model folder:",model_folder,file=f)
		print("training model, args from script (though remember: if model already existed,",
				"then it's using the existing scheduler):\n",file=f)
		d = vars(args)
		print("\t||\t".join(n+":"+str(d[n]) for n in d),file=f)
		print("\n\nbatch info:\n",file=f)
		for n in batches:#,s in zip(["train","val","test"],[train_batches,val_batches,test_batches]):
			print(n,"set:",file=f,end="")
			s = batches[n]
			s.dummy_load = True            
			batch_sizes = [len(b[2]) for b in s] # each b in s is X,Y,lengths,max_len
			print("num batches:",len(s),", batch sizes:",{i:batch_sizes.count(i) for \
				i in sorted(list(set(batch_sizes)))},file=f)
			s.dummy_load = False                        
			# print("avg seq len:",mean([len(x) for b in s for x,y in b]),
				# ", total size:",sum(len(x) for b in s for x,y in b),file=f)
		print("========\n\n",file=f)


def make_batch_loader(args,ds,model):
	return torch.utils.data.DataLoader(ds,args.batch_size,
		collate_fn = ds.prep_batch, 
		num_workers=0, # having multiple workers is messing up my speed like mad and i have no idea why
		shuffle=(not "palindrome" in args.lang)) # palindromes are paired (palindrome, warped version) to hopefully help learn faster

def get_prepped_batches(args,datasets,model):
	# convert once in advance all the sequences to the model's ints for convenience
	def to_int_dataset(n):
		d = datasets[n]
		res = IntsDataset(datasets[n],model)
		return res
	int_datasets = {n:to_int_dataset(n) for n in datasets}
	return {n:make_batch_loader(args,int_datasets[n],model) for n in int_datasets}

def _get_trainer(args,model,model_folder,from_checkpoint,train_file):
	return TransformerEncoderClassifierTrainer(model,args.initial_lr,model_folder,
                                     reset_patience=args.reset_patience,
                                     init_from_checkpoint=from_checkpoint,training_out=train_file,
                                     optim=args.optim,gamma=args.gamma,
                                     perfect_acc_cut=args.perfect_acc_cut,
                                     attn_loss_c=args.attn_loss_c)

def print_base_info(model,prints_file,base_folder):
	print("training language [",model.lang_name,"] on base model taken from folder:",base_folder,file=prints_file)
	print("base model has previous language history (in order):\n\t","\t, ".join(model.old_langs),file=prints_file)

	def print_dataset_metrics(m,n):
		print("best",n,"loss:",m[n+"_losses"][-1],file=prints_file)
		print("best",n,"acc:",m[n+"_accs"][-1],file=prints_file)

	for i,l in enumerate(model.old_langs):
		print("\nmetrics for lang #",i," - ",l,":",file=prints_file)
		m = model.old_metrics[i]
		print("num epochs:",len(m["train_accs"]),file=prints_file)
		[print_dataset_metrics(m,n) for n in ["train","val","test"]]
	
	print("\n\n",file=prints_file)

def _get_model(args,alpha,out_classes,non_token,is_classification_task):
	return TransformerEncoderClassifier(alpha,out_classes,args.d_model,args.nheads,args.nlayers,args.ff,
	                             dropout=args.dropout,non_token=non_token,max_len=args.max_len,
	                             positional_encoding_type=args.positional_encoding_type,result_index=args.result_index,
	                             is_classifier=is_classification_task,
	                             lang_name=args.lang,add_BOS_to_input=args.add_BOS,
	                             with_causal_mask=args.with_causal_mask)


def get_trainer(model_folder,args,prints_file,alpha,out_classes,non_token,
				is_classification_task,prepped_model,train_attn):
	def finish_f(t):
		t.model.set_training_attn(train_attn)
		return t

	if not None is prepped_model:
		return finish_f(_get_trainer(args,prepped_model,model_folder,False,prints_file))
	print("~~~~~~~~~ model loading ~~~~~~~~~~",file=prints_file)
	# attempt to get existing model
	trainer = _get_trainer(args,dummy_telm(),model_folder,True,prints_file)

	if args.overwrite or (None is trainer) or (None is trainer.model): # if overwriting, or failed to load a model: get new model + trainer
		if args.overwrite:
			print("didn't get a model from the folder, making a completely new model",file=prints_file)
		else:
			print("making new model (because not asked to overwrite existing, if any)",file=prints_file)

		model = _get_model(args,alpha,out_classes,non_token,is_classification_task)
		trainer = _get_trainer(args,model,model_folder,False,prints_file)

	elif None in [trainer.optim,trainer.lr_scheduler]: # one of these failed to load - just do them fully
		print("got a model from the folder but found no optimiser/scheduler: making those anew",
				"for this model, from given args",file=prints_file)
		trainer = _get_trainer(args,trainer.model,model_folder,False,prints_file)
	else:
		print("fully loaded existing model, optimiser, and scheduler for this train round",file=prints_file)

	return finish_f(trainer)


def full_run(args):
	print("going to train lang:",args.lang)

	datasets, ds_meta = get_lang(args)
	args, datasets = apply_overfit_or_tiny(args,datasets)
	args, prepped_model, base_folder = args, None, None

	model_folder = get_model_folder(args)
	training_prints_file,dataset_prints_file = get_prints_files(model_folder,args,datasets)
	


	with open(training_prints_file,"a") as f:
		trainer = get_trainer(model_folder,args,f,ds_meta.alpha,
							ds_meta.out_classes,ds_meta.non_token,
							ds_meta.is_classification_task,prepped_model,
							ds_meta.using_tgt_attns)
	batches = get_prepped_batches(args,datasets,trainer.model)
	print_train_info(training_prints_file,dataset_prints_file,model_folder,args,ds_meta,datasets,batches)


	print("finished all preps, and beginning training",flush=True)
	with open(training_prints_file,"a") as tf:
		print("entering training for",args.n_epochs,"epochs",file=tf)
		trainer.train_for_epochs(batches["train"], batches["valid"], batches["test"],args.n_epochs,training_out=tf)
	
if __name__ == "__main__":
	args = get_args()
	full_run(args)