from TransformerClassifier import dummy_telm
from Helper_Functions import things_in_path, load_pytorch_model

def load_any_transformer_for_lang(lang,must_contain=None,must_not_contain=None,loud=False,\
	write_lang_name=True,report_folder=False,load_all=False,from_subfolder=None,nested1_too=False):
	main_folder = "lms"
	if from_subfolder:
		main_folder += "/" + from_subfolder
	all_folders = things_in_path(main_folder,only_folders=True)
	relevant = [main_folder+"/"+f for f in all_folders if f==lang]
	if nested1_too:
		# as some might have been subfolders containing several langs, check those too:
		for f in all_folders:
			poss_relevant = things_in_path(main_folder+"/"+f,only_folders=True)
			relevant += [main_folder+"/"+f+"/"+pr for pr in poss_relevant if pr==lang]
	if loud:
		print("for lang:",lang,"got relevant subfolders:",relevant)
	transformer_folders = [relevant[0]+"/"+p for p in things_in_path(relevant[0],only_folders=True)]
	for r in relevant[1:]:
		transformer_folders += [r+"/"+p for p in things_in_path(r,only_folders=True)]
	if loud:
		print("got relevant transformer folders:\n","\n".join(transformer_folders))
	if not None is must_contain:
		transformer_folders = [t for t in transformer_folders if must_contain in t]
		if loud:
			print("filtered for folders containing",must_contain,", now have only transformer folders:\n",transformer_folders)
	if not None is must_not_contain:
		transformer_folders = [t for t in transformer_folders if not must_not_contain in t]
		if loud:
			print("filtered for folders not containing",must_not_contain,", now have only transformer folders:\n",transformer_folders)
	
	def get_model(folder):
		res = load_pytorch_model(dummy_telm(),folder,quiet=not loud)
		if write_lang_name and not None is res: # res can be None if loading from 
		# a folder during training, in particular before it's managed to store any model
			res.lang_name = lang
		return res

	def get_result(folder):
		res = get_model(folder)
		if report_folder:
			return res, folder
		else:
			return res

	if load_all:
		return map(get_result,transformer_folders)
	else:
		folder = transformer_folders[-1]
		print("loading transformer from:",folder)
		return get_result(folder)


