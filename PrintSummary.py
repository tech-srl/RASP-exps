import LoadTransformer
import argparse
from Helper_Functions import mean, things_in_path, prepare_directory, clean_val, pad_to_length
import sys
import Minilangs

parser = argparse.ArgumentParser()
parser.add_argument('--lang',default="all",type=str) 
parser.add_argument('--from-subfolder',type=str,default=None)

def get_all_langs(subfolder):
	langs_folder = "lms"
	if not None is subfolder:
		langs_folder += "/"+subfolder
	all_subs = things_in_path(langs_folder,only_folders=True)
	return [f for f in all_subs if f in Minilangs.minilangs]

def get_list_of_pairs(lang,subfolder):
	res = LoadTransformer.load_any_transformer_for_lang(lang,report_folder=True,
			load_all=True,from_subfolder=subfolder)
	if isinstance(res,tuple): # single model-folder pair
		res = [res]
	else: # should be a map generating model-folder pairs
		res = list(res)
	res = [r for r in res if (not None is r[0])] # remove models that failed to load
	return res

def example_model_folder_pairs(lang,subfolder=None):
	args = parser.parse_args()
	args.lang = lang
	args.subfolder = subfolder
	return get_list_of_pairs(args)

def get_all_layer_head_combinations(models):
	combs = set()
	for m in models:
		combs.add((m.nlayers,m.nheads))
	return list(combs)

def get_all_for_comb(models,comb):
	return [m for m in models if (m.nlayers,m.nheads)==comb]

def loss(m,group="test",e=-1):
	if m.training_attn:
		l = m.metrics[group+"_seq_losses"]
	else:
		l = m.metrics[group+"_losses"]
	if (not l) or (e>0 and len(l)<e):
		return 100
	return l[e]

def acc(m,group="test",e=-1):
	l = m.metrics[group+"_accs"] # acc always just on seq, so dont need if
	if (not l) or (e>0 and len(l)<e):
		return 0
	return l[e]

def num_epochs(m):
	return len(m.metrics["train_losses"])

def group_summary(models,group="test",e=-1):
	lloss = lambda m: loss(m,group=group,e=e)
	lacc = lambda m: acc(m,group=group,e=e)
	losses = list(map(lloss,models))
	accs = list(map(lacc,models))
	best_loss, avg_loss, worst_loss = min(losses), mean(losses), max(losses)
	best_acc, avg_acc, worst_acc = max(accs), mean(accs), min(accs)
	return best_loss, avg_loss, worst_loss, best_acc, avg_acc, worst_acc
		
def print_group_summary(models,group="test",e=-1,f=sys.stdout,tiny=False):
	best_loss, avg_loss, worst_loss, best_acc, avg_acc, worst_acc = group_summary(models,group=group,e=e)
	epstr = "[last ep]" if e==-1 else ("[ep "+str(e)+"]")
	if not tiny:
		print(epstr,group," losses (best//avg//worst):\t",my_num_str(best_loss),
				" // \t",my_num_str(avg_loss)," // \t",my_num_str(worst_loss),file=f)
	print(epstr,group,"accs   (best//avg//worst):\t",my_num_str(best_acc),
				" // \t",my_num_str(avg_acc)," // \t",my_num_str(worst_acc),file=f)
	if not tiny:
		print("",file=f)
		print("all",epstr,group,"losses:     \t",", ".join(list(my_num_str(loss(m,group=group,e=e)) for m in models)),file=f)
		print("all",epstr,group,"test accs:  \t",", ".join(list(my_num_str(acc(m,group=group,e=e)) for m in models)),file=f)

def my_num_str(v):
	res = pad_to_length(str(clean_val(v,5)),10)
	if res.startswith("1.0") and v<1:
		res = "1.0-" # signal that not precisely 1.0, even if very close
	return res

def load_and_print_lang_summary(lang,subfolder,f=sys.stdout,with_individuals=False,tiny=False):
	print("="*30,file=f)
	print("\t\tlang:",lang,file=f)
	print("="*30,file=f)
	model_folder_pairs = get_list_of_pairs(lang,subfolder)
	models_folders = {m:f for m,f in model_folder_pairs}
	all_models = list(models_folders.keys())
	all_combs = get_all_layer_head_combinations(all_models)
	for c in all_combs:
		nlayers, nheads = c
		models = get_all_for_comb(all_models,c)
		print("==========",file=f)
		print("summary for models with layers/heads",nlayers,"/",nheads,": (",len(models),"models)",file=f)
		print("==========",file=f)
		print_group_summary(models,group="test",e=-1,f=f,tiny=tiny)
		if not tiny:
			print("======",file=f)
			print_group_summary(models,group="val",e=20,f=f)
			print("======",file=f)
			print("num epochs:                    \t",", ".join(list(my_num_str(num_epochs(m)) for m in models)),file=f)
			if with_individuals:
				print("===",file=f)
				print("individuals:",file=f)
				for m in models:
					print(models_folders[m],file=f)
					print("test loss:",my_num_str(loss(m,group="test",e=-1)),",\t\ttest acc:",my_num_str(acc(m,group="test",e=-1)),file=f)
				print("===",file=f)
		print("==========",file=f)	
	print("="*30,file=f)


if __name__ == "__main__":
	args = parser.parse_args()
	summarynest = (args.from_subfolder + "/") if not None is args.from_subfolder else ""
	out_file_no_indivs = "summaries/"+summarynest+args.lang+".txt"
	out_file_with_indivs = "summaries/"+summarynest+args.lang+"_with_individuals.txt"
	out_file_tiny = "summaries/"+summarynest+args.lang+"_tiny.txt"
	prepare_directory(out_file_with_indivs,includes_filename=True) # that'll cover the path for noindivs too - they're going to the same folder
	with open(out_file_no_indivs,"w") as f_noind, open(out_file_with_indivs,"w") as f_ind, open(out_file_tiny,"w") as f_tiny:
		if args.lang == "all":
			langs = get_all_langs(args.from_subfolder)
		else:
			langs = [args.lang]
		for l in langs:
			load_and_print_lang_summary(l,args.from_subfolder,f=f_noind,with_individuals=False)
			load_and_print_lang_summary(l,args.from_subfolder,f=f_ind,with_individuals=True)
			load_and_print_lang_summary(l,args.from_subfolder,f=f_tiny,tiny=True)

	