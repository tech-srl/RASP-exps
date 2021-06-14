import LoadTransformer
import DrawSequence
import argparse
import Minilangs

parser = argparse.ArgumentParser()
parser.add_argument('--lang',type=str) 
parser.add_argument('--sequence',type=str)
parser.add_argument('--must-contain',type=str,default=None,help="only loads from transformer in path containing this as substring")
parser.add_argument('--with-numbers',type=int,default=-1,help="when with-numbers=d for d>0, overlay the attention heatmap with the actual softmaxed scores at \
						each point, with d digits after the decimal point.")
parser.add_argument('--size',type=int,default=5,help="size of the image")
parser.add_argument('--load-all',action='store_true',help="make the heatmaps for all transformers trained on the requested language")
parser.add_argument('--nested1-too',action='store_true')
parser.add_argument('--from-subfolder',type=str,default=None,help="subfolder to look in")
parser.add_argument('--skip-colorbar',action='store_true')

args = parser.parse_args()

def draw(model,folder,reset_main_txt):
	if None is model:
		return # can happen if loading model that is still being trained
	model_args = folder.split("/")[-1]
	DrawSequence.process_and_draw(model,args.sequence,hyperparams_str=model_args,
		with_numbers=args.with_numbers,size=args.size,
		gold=Minilangs.minilangs.get_classifier(args.lang),
		reset_main_txt=reset_main_txt,into_subfolder=args.from_subfolder,with_colorbar=not args.skip_colorbar)

af = LoadTransformer.load_any_transformer_for_lang(args.lang,
			must_contain=args.must_contain,report_folder=True,
			load_all=args.load_all,from_subfolder=args.from_subfolder,nested1_too=args.nested1_too)

if isinstance(af,tuple):
	a,f = af
	draw(a,f,True)
else:
	for i,(a,f) in enumerate(af):
		draw(a,f,i==0)

