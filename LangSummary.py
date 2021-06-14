import LoadTransformer
import DrawSequence
import argparse
from Helper_Functions import prepare_directory, things_in_path
from Helper_Functions import clean_val as _clean_val
from copy import copy

def padto(s,width):
	if len(s)>width:
		return s
	d = width-len(s)
	n1 = int(d/2)
	n2 = d-n1
	return (" "*n1) + s+ (" "*n2)

def clean_val(v,width,digits=5):
	if not isinstance(v,str):
		v = _clean_val(v,digits=digits)
	return padto(str(v),width)

def get_all_langs():
	return things_in_path("lms",only_folders=True)

def make_txt(lang="all",seqs=[],valwidth_factor=-1):
	if lang=="all":
		langs = get_all_langs()
	else:
		langs=[lang]
	for l in langs:
		_make_txt(l,seqs=seqs,valwidth_factor=valwidth_factor)


def _make_txt(lang,seqs=[],valwidth_factor=-1):
	all_models = LoadTransformer.load_any_transformer_for_lang(lang,load_all=True,report_folder=True)
	infos = {}
	id2f = {}
	for a,f in all_models:
		if None is a:
			continue
		i = len(infos)
		f = f.split("/")[-1]
		infos[f] = model_info(a,seqs)
		infos[f]["id"] = i
		id2f[i]=f
		is_classifier = a.is_classifier
		some_f = f
	nfolders = len(infos)
	if not nfolders:
		return
	stats = ["id","test acc","heads","layers","dim","ff","epochs","gamma","lr","position","train e20","val e20"]
	stats += sorted([k for k in infos[some_f].keys() if (k not in stats) and (not k=="seq_outs")]) # f is last loaded model folder, fine to be arbitrary here
	min_stat_width=6
	alll_seqs = copy(seqs)
	for f in infos:
		if alll_seqs and isinstance(infos[f]["seq_outs"][0],str):
			alll_seqs += infos[f]["seq_outs"]
	seq_widths = max([len(s) for s in alll_seqs]+[1])+3
	val_widths = 6 if is_classifier else seq_widths # check from last model, they'll all be the same on same task

	filename = "summaries/"+lang
	prepare_directory(filename)
	with open(filename,"w") as f:
		print("lang:",lang,"\n\n",file=f)
		stat_widths = [max(len(s)+1,min_stat_width) for s in stats]
		print(*[clean_val(s,width) for s,width in zip(stats,stat_widths)],"\n",file=f)
		for i in range(nfolders):
			folder = id2f[i]
			if i>0 and i%3==0:
				print("",file=f) # break up lines for readability
			print(*[clean_val(infos[folder][s],width) for s,width in zip(stats,stat_widths)],file=f)

		print("\n\nsome example seqs:\n",file=f)
		print(clean_val("seq",seq_widths),*[clean_val("out-"+str(i),val_widths) for i in range(nfolders)],file=f)
		for si,s in enumerate(seqs):
			print(clean_val(s,seq_widths),*[clean_val(infos[id2f[i]]["seq_outs"][si],val_widths) \
												for i in range(nfolders)],file=f)

		print("\n\nfolders:",file=f)
		for i in range(nfolders):
			print(i,":\t\t",id2f[i],file=f)


def get_list_val(l,i=-1,d=-1):
	try:
		return l[i]
	except:
		return d

def model_info(model,seqs):
	res = {}
	test_accs = model.metrics["test_accs"]
	res["test acc"] = get_list_val(model.metrics["test_accs"])
	res["train acc"] = get_list_val(model.metrics["train_accs"])
	res["val acc"] = get_list_val(model.metrics["val_accs"])
	res["train e20"] = get_list_val(model.metrics["train_accs"],19)
	res["val e20"] = get_list_val(model.metrics["val_accs"],19)
	# this may happen if run this script while some models are still training
	res["epochs"] = len(model.metrics["train_accs"])
	res["heads"] = model.nheads
	res["layers"] = model.nlayers
	res["dim"] = model.d_model
	res["ff"] = model.ff
	res["position"] = model.positional_encoding_type
	
	res["gamma"] = getattr(model,"last_gamma",-1)
	res["lr"] = getattr(model,"last_initial_lr",0.0003)
	res["seq_outs"] = [model.classify(s) for s in seqs]
	return res

if __name__ == "__main__":
    # execute only if run as a script
    make_txt()