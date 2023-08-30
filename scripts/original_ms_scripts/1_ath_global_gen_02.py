# %% [markdown]
# # Athena Global Generation
# Large Scale Empirical Analysis 

# %%
from pathlib import Path
import csv
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import functools

pd.options.display.float_format = '{:.2f}'.format

# %%
from tokenizers import ByteLevelBPETokenizer
import torch
import importlib
from fairseq.models.transformer import TransformerModel

# %%
import warnings
from matplotlib import colors
import os
from rationalization import rationalize_lm, rationalize_conditional_model

# %%
def param_default():
    corpus = 'fm_fc_ms_ff' #<-- Scope
    data_path = Path('../athena-datasets/' + corpus + '/')
    data_path_raw = Path('../athena-datasets/' + corpus + '/raw/')
    tokenizer_path = Path('../tokenizer/')
    return {
        'bpe_path' : tokenizer_path / 'universal_tokenizer/universal_tokenizer/roberta_aug_spaces',
        'eval_raw': [data_path_raw / 'eval/input.methods.txt',
                        data_path_raw / 'eval/output.tests.txt'],
        'test_raw': [data_path_raw / 'test/input.methods.txt', 
                        data_path_raw / 'test/output.tests.txt'],
        'train_raw': [data_path_raw / 'train/input.methods.txt', 
                        data_path_raw / 'train/output.tests.txt'],
        'data_labels' : ['test_raw'],#['eval_raw','test_raw','train_raw'], <----- Just Test
        'output_pandas' : data_path / 'pandas/',
        'out_processed' : '/datasets/out_processed/',
        'model_name_or_path' : 'models/checkpoint_dir_01/models/', #Model Path
        'checkpoint_file': 'checkpoint_best.pt', #Model
        'data_preprocessed':'/home/davidna/data/dummy/sequential-rationales/fairseq/fairseq/data-bin/bins/',
        'output_results' : 'results/' 
    }

# %%
params = param_default()
params['checkpoint_file']

# %%
params['eval_raw']

# %%
#Setting experiments 
#! export CUDA_VISIBLE_DEVICES="1"

# %% [markdown]
# ## Universal Tokenizer

# %%
def load_tokenizer(bpe_path):
    return ByteLevelBPETokenizer(str(bpe_path)+'-vocab.json',str(bpe_path)+'-merges.txt')

# %%
def lazy_decode(bpe_java):
    return bpe_java.replace(' ','').replace('Ġ',' ').replace('Ċ','\n')

# %%
def prettify_java(minified_java):
    "tries to undo Michele's minification. Works decently, although for loops and sets get newlines inserted, and there are no empty lines or comments"
    minified_java = minified_java.replace('{','{\n').replace('}','}\n').replace(';',';\n')
    num_indents = 0
    pretty_java = ''
    for line in minified_java.splitlines():
        if line.lstrip().startswith('}'):
            num_indents -= 1
        pretty_java += num_indents*'    '+line+'\n'
        if line.endswith('{'):
            num_indents += 1
        if line.endswith('}') and not line.lstrip().startswith('}'):
            num_indents -= 1
    return pretty_java

# %%
tokenizer = load_tokenizer(params['bpe_path'])

# %% [markdown]
# ## Data Loading and Testing

# %%
#export
def method_size_vector( method_vector ):
    '''Return the size of the tokens for a give method based on id
        Assuming that method_vector is an array of tokens
    '''
    input_ids = [ len(mtd) for mtd in method_vector ]
    return input_ids

# %%
def super_set_code():
    data = {}
    for label in params['data_labels']:
        for val,path_data in enumerate(params[ label ]):
            df = pd.read_csv( path_data, sep="\n", header=None, names=[label+str(val)]) #reading file
            df[label+'_bpe'+str(val)] = [ enc.tokens for enc in tokenizer.encode_batch( df[label+str(val)].values ) ] #bpe
            df['method_size'+str(val)] = method_size_vector( df[label+'_bpe'+str(val)].values ) #counting tokens
            data[label+str(val)] =  df  
        #data[-1].columns = [ label ]
    return data

# %%
# Loading Json Sets
def load_checkpoint_1():
    super_df = {}
    for label in params['data_labels']:
        for val, _ in enumerate(params[ label ]):
            super_df[ label+str(val) ] = pd.read_json( params['output_pandas'] / (label+str(val) +'.json')  )
            print("read:",label+str(val))
    return super_df

# %%
super_data = load_checkpoint_1()

# %%
super_data['test_raw0'].head(1) #Source

# %%
#Size Statistics of Source Set
super_data['test_raw0'].method_size0.describe()

# %%
SET_METHOD_SIZE = 100 #<---- HARDCODED
super_data['test_raw0'][super_data['test_raw0'].method_size0 <= SET_METHOD_SIZE ].method_size0.describe()

# %%
#Target Set
super_data['test_raw1'].head(1) #Target

# %% [markdown]
# ## Model Loading and Testing

# %%
#Loading a pretrain model
model = TransformerModel.from_pretrained(
  model_name_or_path = params['model_name_or_path'],
  checkpoint_file = params['checkpoint_file'],
  #data_name_or_path = params['data_preprocessed']
)

# %%
## Move model to GPU if available and trigger evaluation mode
if torch.cuda.is_available():
  model.cuda()
model.eval()

# %%
model.model = model.models[0]

# %%
model.device

# %%
def joining_encode_tokens( arr_tokens, model ):
    if len(arr_tokens) > SET_METHOD_SIZE:
        arr_tokens = arr_tokens[0:SET_METHOD_SIZE]
    focal_code = " ".join(arr_tokens)
    return model.encode( focal_code )

# %%
#Sampling without replacement
#Testing size: 78388
#Sampling size with 95% of confidence and 3% Error = 1053 ~ 1000
def code_sampling(df_super_data ,  FLAG_SAMPLING = True, SIZE_SAMPLING = 1000, random_state = 3): #<---- HARDCODED
    
    df_sampled_code = df_super_data['test_raw0'][df_super_data['test_raw0'].method_size0 <= SET_METHOD_SIZE ].sample(
            n = SIZE_SAMPLING,
            replace = False,
            random_state = random_state # For reproducibility
    )

    if FLAG_SAMPLING:
        df_sampled_code['input_tokens'] = [ joining_encode_tokens(arr_sample, model=model) for arr_sample in df_sampled_code.test_raw_bpe0.values ]
        #df_sampled_code['origin_pos'] = df_sampled_code.index
    else:
        df_sampled_code['input_tokens_pos'] = [ joining_encode_tokens(arr_sample, model=model) for arr_sample in df_super_data['test_raw0'].test_raw_bpe0.values]
        #df_sampled_code['origin'] = df_sampled_code.index
    return df_sampled_code

# %%
df_sampled_code = code_sampling(
    df_super_data = super_data,
    SIZE_SAMPLING = 1000
)

# %%
df_sampled_code.head()

# %%
#df_sampled_code.reset_index()

# %%
#super_data['test_raw0'].filter( items = df_sampled_code.origin_pos.values, axis=0 ) #<-------- Retrieving original Data

# %%
len(df_sampled_code.input_tokens.values)

# %%
df_sampled_code.input_tokens.values[0]

# %%
SAMPLES = 30 #<---- Hardocoded
MAX_GEN_TOK = 100

# %%
def df_sample_generation(
    df_sampled_code, 
    model, 
    n=1, 
    max_gen_tok = 100
    ):
    generated_input = lambda input,model,n,max_gen_tok: model.generate( 
        input,
        beam = n, 
        maxlen = max_gen_tok, ##WARNING, This parameter is not working
        #max_length = n, 
        do_sample = False, 
        pad_token_id = 50256 ) ## HARDCODED
    arr_generated_code = np.array([ generated_input(input, model=model, n=n, 
                                max_gen_tok=max_gen_tok ) for input in df_sampled_code.input_tokens.values ]).T
    
    #dict_generated_code = { i: [j['tokens'].cpu().data.numpy()[:max_gen_tok] for j in samples] for i,samples in enumerate(arr_generated_code) } #Max Token Generation
    dict_generated_code = { i: [j['tokens'].cpu().data.numpy() for j in samples] for i,samples in enumerate(arr_generated_code) }
    dict_generated_code['source_sampling'] = [ i.cpu().data.numpy() for i in df_sampled_code.input_tokens.values] 
    #return arr_generated_code
    df_temp = pd.DataFrame().from_dict( data=dict_generated_code ) # DataFrame from Generation
    df_temp = pd.concat([df_sampled_code.reset_index(), df_temp ], axis=1) #Index before concating
    #return pd.DataFrame().from_dict( data=dict_generated_code )
    return df_temp

# %%
#TODO limit the number of tokens generated
#WARNING TIME CONSUMING
df_generated_input = df_sample_generation( 
    df_sampled_code = df_sampled_code, 
    model = model, 
    n = SAMPLES, 
    max_gen_tok = MAX_GEN_TOK 
)
# [ sample_generation(input, model=model) for input in input_tokens[:2] ]

# %%
df_generated_input

# %% [markdown]
# ### Statistics and Checkpoint

# %%
np_len_method = [ (np.array([ len(gen_method) for gen_method in df_generated_input[j] ]).mean(),
                   np.array([ len(gen_method) for gen_method in df_generated_input[j] ]).std()  )
                    for j in range(30) ]

# %%
np_len_method

# %%
#Checkpoint of Generation
def checkpoint_generation( df , name = '1_generation_[max:100]_02.json' ):
    df.drop('input_tokens', axis=1).to_json( params['output_results'] + name )
    pass

# %%
checkpoint_generation( df = df_generated_input )

# %%
df_generated_input = pd.read_json( params['output_results'] + '1_generation_[max:100]_02.json' )

# %%
df_generated_input.head()

# %%
#tst decoding
decoded = model.decode(df_generated_input['1'][0])
decoded

# %%
prettify_java( lazy_decode( decoded ) )

# %%
## MEMORy DEALLOCATION
torch.cuda.empty_cache()


