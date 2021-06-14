from torch.utils.data import Dataset
import string
import random
from collections import Counter
from TransformerClassifierTrainer import prep_batch

#### functions to help make datasets ####
class SampleGenerator:
    def __init__(self):
        self.determinised = False

class DeterminisedFunc(SampleGenerator): # hopefully this won't be local and then it can be pickled for multiprocessing
    def __init__(self,f):
        super(DeterminisedFunc,self).__init__()
        self.f = f
        self.f_determinised = isinstance(f,SampleGenerator) and f.determinised # used when messing with an already determinised func, in some way other than determinising it, and want to remember it was already this way
        self.determinised = True
    def __call__(self,i,*args,**kwargs):
        if self.f_determinised:
            return self.f(i,*args,**kwargs)
        old_state = random.getstate() # save current random state
        random.seed(i)
        res = self.f(*args,**kwargs)
        random.setstate(old_state) # don't mess with the overall randomness
        return res

class AlternatingGenerator(SampleGenerator):
    def __init__(self,gens):
        super(AlternatingGenerator,self).__init__()
        self.gens = [DeterminisedFunc(g) for g in gens]
        self.determinised = True
    def __call__(self,i):
        g = self.gens[i%len(self.gens)]
        return g(i)

def alternating_generator(gens):
    return DeterminisedFunc(AlternatingGenerator(gens)) # AlternatingGenerator figures out determinising

def completely_random(alpha,n,reps_prob=0):
    def choose_random(s):
        if (reps_prob>0) and len(s)>0:
            if random.random() <= reps_prob:
                return random.choice(s)
        return random.choice(alpha)

    if isinstance(alpha,str):
        s = ""
        for _ in range(n):
            s += choose_random(s)
    else:
        s = []
        for _ in range(n):
            s += [choose_random(s)]
        s = tuple(s)
    return s

class SpecificRandomGenerator(SampleGenerator):
    def __init__(self,alpha,shortlen,longlen,reps_prob,with_eos,with_bos,non_token):
        super(SpecificRandomGenerator,self).__init__()
        self.alpha = alpha
        self.shortlen, self.longlen = shortlen, longlen
        self.reps_prob = reps_prob
        self.with_eos, self.with_bos = with_eos, with_bos
        self.non_token = non_token
    def __call__(self):
        s = completely_random(self.alpha,random.randint(self.shortlen,self.longlen),
                    reps_prob=self.reps_prob)
        if self.with_eos:
            s += self.non_token
        if self.with_bos:
            s = self.non_token + s
        return s


def randoms_generator(with_eos=False,alpha=string.ascii_lowercase,
                                non_token="§",shortlen=0,
                                longlen=100,with_bos=False,reps_prob=0):
    return SpecificRandomGenerator(alpha,shortlen,longlen,
                    reps_prob,with_eos,with_bos,non_token)


OUT_INTS = 0

class XYPairs_Generator_Dataset(Dataset):
    # no BOS, intsdataset will add it
    def __init__(self, generator,classifier,non_token,n,base,name=""):
        super().__init__()
        self.generator = generator if isinstance(generator,DeterminisedFunc) else DeterminisedFunc(generator)
        self.n = n
        self.base = base
        self.classifier = classifier
        self.non_token = non_token
        self.name = name

    def __getitem__(self, index):
        if index == self.n:
            raise StopIteration
        assert (index >= 0) and (index < self.n), "attempted to get item #"+str(index)+" from dataset of size: "+str(self.n)

        s = self.generator(index+self.base)
        return s, self.classifier(s)

    def __len__(self):
        return self.n

    def crop(self,n):
        self.n = min(self.n,n)

class XYPairs_Dataset(Dataset):
    # no BOS, intsdataset will add it
    def __init__(self, samples,classifier,non_token,name=""):
        super().__init__()
        self.samples = [(x,classifier(x)) for x in samples]
        self.n = len(samples)
        self.name = name
        self.classifier = classifier # useful for output anaylsis later (compare target to output)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.n

    def crop(self,n):
        self.samples = self.samples[:n]
        self.n = len(self.samples)

def identity_f(y):
    return y


class IntsDataset(Dataset):
    # adds BOS
    def seq2int(self,x):
        res = [self.char2int[c] for c in x]
        if self.add_BOS_to_input:
            return [self.non_token_index] + res
        else:
            return res

    def y2int(self,y):
        if self.training_attn:
            y,attn = y  # split away given attention
        if self.is_classifier:
            res = self.single_out_to_int(y)
        else:
            res = [self.single_out_to_int(c) for c in y]
        if self.training_attn:
            res = res,attn
        return res

    def convert_sample(self,xy):  
        x,y = xy
        # print("converting sample with x:",x)      
        # print("y length:",len(y))
        res = (self.seq2int(x), self.y2int(y))
        return res

    def out2int_f(self,y):
        return self.out2int[y]


    def sort_out_classes(self,model):
        self.out_classes = model.out_classes
        if self.out_classes == OUT_INTS:
            self.single_out_to_int = identity_f
        else:
            self.out2int = model.out2int
            self.single_out_to_int = self.out2int_f

    def __init__(self, ds, model):
        super().__init__()
        self.add_BOS_to_input = model.add_BOS_to_input
        self.training_attn = model.training_attn


        if not self.training_attn:
            self.delay_convert = (len(ds) >= 5e5) and isinstance(ds,XYPairs_Generator_Dataset)
        else:
            self.delay_convert = (len(ds) >= 2e3) and isinstance(ds,XYPairs_Generator_Dataset)
        self.non_token_index = model.non_token_index
        self.char2int = model.char2int

        self.sort_out_classes(model)
        self.is_classifier = model.is_classifier
        a= [ s for s in ds]
        self.samples = ds if self.delay_convert else [ self.convert_sample(s) for s in ds]         
        self.n = len(self.samples)
        self.name = ds.name
        self.dummy_load = False
        
        

    def __getitem__(self, index):
        if self.dummy_load:
            return 0
        # if index%100000 == 0:
        #     print("getting index",index,"from",self.name,"set (dataset size:",self.n,")",flush=True)
        s = self.samples[index]
        if self.delay_convert:
            return self.convert_sample(s)
        else:
            return s

    def __len__(self):
        return self.n

    def prep_batch(self,x):
        return prep_batch(x,self.non_token_index)

class GetLangs:
    def __init__(self,lang_funcs):
        self.lang_funcs = lang_funcs
    def __contains__(self,lang):
        return lang in self.lang_funcs
    def __getitem__(self,lang):
        old_state = random.getstate() # save current random state
        random.seed(42) # move to same thing always for consistency across runs (these are our constant datasets here, even if never stored in a txt file)
        res = self.lang_funcs[lang]()
        random.setstate(old_state) # don't mess with the overall randomness
        return res
    def __setitem__(self,lang,f):
        self.lang_funcs[lang] = f
    def keys(self):
        return self.lang_funcs.keys()
    def get_classifier(self,lang):
        if lang not in self.keys():
            return None
        return self[lang][0]["train"].classifier
    def peep_lang(self,lang,subset="train",start=0,stop=10,just_print=True,print_attn_cap=20,ignore_over=-1):
        a = self[lang]
        b = a[0][subset]
        res = []
        for i in range(start,stop):
            example = b[i]
            if ignore_over>0 and len(example[0])>ignore_over:
                continue
            res.append(example)
        def quickstr(l):
            if False not in [len(str(s))==1 for s in l]:
                return ''.join(str(s) for s in l)
            else:
                return str(l)
        for p in res:
            x,y = p
            if ignore_over>0 and len(x)>ignore_over:
                continue
            with_attn = (len(y)==2) and isinstance(y[1],dict)
            if with_attn:
                y, d = y
            print("=================")
            print(quickstr(x),"\n"+quickstr(y))
            if with_attn and len(x)<print_attn_cap:
                for l in d:
                    for h in d[l]:
                        print("layer",l,"head",h,":")
                        m = d[l][h]
                        for v in m:
                            print(*list(map(int,v)))
            print("=================")
        if not just_print:
            return res



    def check_policy(self,lang,policy):
        def policy_acc(pairs):
            def count_match(s1,s2):
                return sum([a==b for a,b in zip(s1,s2)])
            total_correct = sum(count_match(p[1],policy(p[0])) for p in pairs)
            total_len = sum(len(p[0]) for p in pairs)
            return total_correct/total_len
        print("policy accs:\ntrain:",policy_acc(self[lang][0]["train"]),
                            "\nval:",policy_acc(self[lang][0]["valid"]),
                            "\ntest:",policy_acc(self[lang][0]["test"]))



    def class_dist(self,lang,subset="train",print_res=True):
        a=self[lang]
        is_classification_task = a[-1]
        if not is_classification_task:
            print("not classification task!")
            return
        b=a[0][subset] # options: train, test, valid
        c=[e[1] for e in b]
        d=Counter(c)
        if print_res:
            for k in sorted(d.keys()):
                print(k,":\t",d[k])
            if set(d.keys())==set([True,False]):
                print("T/(T+F) ratio:",d[True]/(d[True]+d[False]))
        else:
            return d




minilang_funcs = {}
minilangs = GetLangs(minilang_funcs) # minilang funcs will be filled in this file. minilangs should see it without problems


def makesets(samples,classifier,non_token,train_size,val_size,test_size,dont_shuffle=False):
    assert train_size+val_size+test_size <= len(samples), "not enough samples, only "+str(len(samples))+\
            " when trying to make train, test, val of sizes: "+str(train_size)+", "+str(test_size)+", "+str(val_size)
    datasets = {}
    if not dont_shuffle:
    	random.shuffle(samples) # VERY important if created first positives then negatives or something
    datasets["train"] = XYPairs_Dataset(samples[:train_size],classifier,non_token,name="train")
    datasets["valid"] = XYPairs_Dataset(samples[train_size:train_size+val_size],classifier,non_token,name="val")
    datasets["test"] = XYPairs_Dataset(samples[train_size+val_size:train_size+val_size+test_size],classifier,non_token,name="test")
    return datasets


def makesets_generator(generator,classifier,non_token,train_size,val_size,test_size): # no shuffle implemented for this one yet
    datasets = {}
    # def __init__(self, generator,classifier,non_token,n,base):
    datasets["train"] = XYPairs_Generator_Dataset(generator,classifier,non_token,train_size,0,name="train")
    datasets["valid"] = XYPairs_Generator_Dataset(generator,classifier,non_token,val_size,train_size,name="val")
    datasets["test"] = XYPairs_Generator_Dataset(generator,classifier,non_token,test_size,train_size+val_size,name="test")
    return datasets
