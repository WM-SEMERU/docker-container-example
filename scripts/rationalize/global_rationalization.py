# %% [markdown]
# # Rationalization @ Global Granularity
# > GPT-2 based global rationalization

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
from sacrebleu.metrics import BLEU

# %%
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch

# %%
import warnings
import importlib
from matplotlib import colors
import os

# %%
import sys
sys.path.insert(1, '/workspaces/code-rationales/sequential-rationales/huggingface')
from rationalization import rationalize_lm

# %%
def param_default():
    return {
        'model_name' : '/workspaces/code-rationales/data/codeparrot-small/checkpoints/checkpoint-29000', 
        'cache_dir': '/workspaces/code-rationales/datax/df_cache_dir',
        #'dataset' : 'code_completion_random_cut_5k_30_512_tokens',
        #'dataset' : 'code_completion_docstring_random_cut_3.8k_30_150_tokens',
        #'dataset' : 'code_completion_docstring_signature_3.8k_30_150_tokens',
        'dataset' : 'code_completion_docstring_5k_30_150_tokens',
        'sampling_results': '/workspaces/code-rationales/data/sampling/gpt',
        'rational_results': '/workspaces/code-rationales/data/rationales/gpt',
    }

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Model Loading and Testing

# %%
model = AutoModelForCausalLM.from_pretrained(
            param_default()['model_name'],
            cache_dir=param_default()['cache_dir'])

# %%
model.to(device)
model.eval()

# %% [markdown]
# ## Tokenizer Loading and Testing

# %%
tokenizer = AutoTokenizer.from_pretrained(param_default()['model_name'])

# %% [markdown]
# ## Data Loading and Testing

# %%
#Loading Code Generation
df_generated_input = pd.read_csv( param_default()['sampling_results'] + '/' + param_default()['dataset'] +'.csv', index_col=0)

# %%
df_generated_input.columns[5:] #Tensor Columns

# %%
df_generated_input.head()

# %%
df_generated_input.shape

# %%
#tst decoding
decoded = tokenizer.decode(eval(df_generated_input['0'][1]))
decoded

# %% [markdown]
# ## Running Rationales

# %%
#Statistics
np.mean( [len(eval(i)) for i in df_generated_input['0'].values] )

# %%
#TODO Run the distribution of each experiment. The mean value of tokens or size for each experiment. 
np.mean( [len(eval(i)) for i in df_generated_input['input_ids'].values] )

# %%
len(df_generated_input['0'].values[1])

# %%
MAX_TOKEN_SIZE = df_generated_input['size'].max() #Hardocoded!!

# %%
#If the model is not fine-tuned or compatible, it will rise an error
#This function works for one tensor of source token and one tensor of target tokens
def rationalize_model(model, tokenizer, input_ids, verbose=True):
    all_rationales, log = rationalize_lm(
        model = model,
        input_ids = input_ids[:MAX_TOKEN_SIZE],
        tokenizer = tokenizer,
        verbose = verbose,
        max_steps=1024 #Max number of steps for greedy rationalization
    )
    return all_rationales, log 

# %%
#tst <------- Test Case 2
def tst_rationalize_model():
    torch.cuda.empty_cache() #Cleaning Cache
    #WARNING TIME CONSUMING
    all_rationales, log = rationalize_model(
        model=model, 
        tokenizer=tokenizer, 
        input_ids=torch.tensor(eval(df_generated_input['0'][0])).to(model.device),
        verbose=False
    )
    pass
tst_rationalize_model()


# %%
def run_multiple_rational(
    model,
    tokenizer, 
    arr_target_tokens, 
    seq_id, #mapping sequence id
    verbose=True
):
    arr_log = []
    for index, val in enumerate(arr_target_tokens):
        all_rationales, log = rationalize_model(
            model=model, 
            tokenizer=tokenizer, 
            input_ids=val,
            verbose=False
        )
        arr_log.append(log)
    arr_code_rationales = [ log['rationalization'] for log in arr_log ] #extracting just rationalizations
    arr_from_sentence = [ list(np.full( len(val), seq_id[arr_i] )) #arr_i maps to the real sequence id
                            for arr_i, val in enumerate(arr_code_rationales)]
    arr_code_rationales = sum( arr_code_rationales, [] ) #flatting
    arr_from_sentence = sum( arr_from_sentence, [] ) #flatting
    return arr_code_rationales, arr_from_sentence

# %%
import gc

# %%
#tst <------- Test Case 2
def tst_run_multiple_rationa():
    gc.collect()
    torch.cuda.empty_cache() #Cleaning Cache
    t_dict_generated_input = { exp : [ torch.tensor(eval(s)).to(model.device) for 
                s in df_generated_input[exp].values ] for exp in df_generated_input.columns[5:]  }
    
    arr_rations, seq_id = run_multiple_rational(
        model = model,
        tokenizer = tokenizer,
        arr_target_tokens =  t_dict_generated_input['0'][:2], 
        seq_id = list( range(2,4) ),
        verbose = False
        )
    return arr_rations, seq_id
tst_arr_rations, seq_id = tst_run_multiple_rationa()

# %%
def pandas_rationales( arr_code_rationales, arr_from_sentence ):
    #Creating pandas_1 {p_rationale}
    rational = lambda list_log,typeset: [ (dict_tok['added_token_text'],round(dict_tok['true_token_prob'],6)) for dict_tok in list_log if dict_tok['from']==typeset]
    log = lambda log_row: [(log_dict['added_token_text'],log_dict['true_token_prob']) for log_dict in log_row] #Typeset

    log_position = lambda log_row: [log_dict['added_token_position'] for log_dict in log_row] #Position of the Rationale
    log_prediction = lambda log_row: [log_dict['true_token_prob'] for log_dict in log_row] #Rationale Prob

    p_rationale = pd.DataFrame()

    p_rationale['goal_token'] = [dict_token['goal_word'] for dict_token in arr_code_rationales]
    p_rationale['from_seq_id'] = arr_from_sentence

    p_rationale['typesets_tgt'] = [ log(log_row) for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]
    
    p_rationale['rationale_pos_tgt'] = [ log_position(log_row) for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]
    p_rationale['rationale_prob_tgt'] = [ log_prediction(log_row) for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]


    return p_rationale

# %%
#Running Rationalization
def run_code_rational( 
        df_generated_input,
        tensor_size, #Control the size of the experiment
        experiment = '5',
        batch_size = 100, 
        model = model, 
        verbose = True 
    ):

    arr_rationals = []
    arr_from_seq = []

    for i in range( 0 , tensor_size , batch_size ):
        print('************************' + str(i) + '************************')
        t_generated_input = df_generated_input[experiment].values[i:i+batch_size]
        t_generated_input = [ torch.tensor(eval(s)).to(model.device) for s in t_generated_input]

        t_arr_rationals,t_arr_from_seq = run_multiple_rational(
            model = model,
            tokenizer = tokenizer,
            arr_target_tokens =  t_generated_input, 
            seq_id = list(range(i,i+batch_size)),
            verbose = verbose
        )

        arr_rationals = arr_rationals + t_arr_rationals
        arr_from_seq = arr_from_seq + t_arr_from_seq

        gc.collect()
        torch.cuda.empty_cache() #Cleaning Cache

    #keys_tensor = list( dict_generated_input.keys() )
    #keys_tensor = keys_tensor[:1] #HardCoded Ratios
    #dict_arr_rations = { key : for key in keys_tensor}
    #torch.cuda.empty_cache() #Cleaning Cache
    print("Experiment Finished: " + experiment)
    return pandas_rationales( arr_rationals, arr_from_seq )

# %%
#tst
def tst_run_code_rational_sampling_set(exp='0'):
    gc.collect()
    torch.cuda.empty_cache()
    tensor_n = 3 #df_generated_input.shape[0]
    EXP = exp
    BATCH = 1
    test_arr_rationals = run_code_rational( 
            df_generated_input = df_generated_input.sample( n = tensor_n, replace = False, random_state=2),
            tensor_size = tensor_n,
            experiment = EXP,
            batch_size = BATCH, 
            model = model, 
            verbose = False 
        )
    return test_arr_rationals
df_test_run = tst_run_code_rational_sampling_set()

# %%
#tst
df_test_run[ df_test_run['from_seq_id'] == 1]

# %%
def run_code_rational_all_set(exp, tensor_n = 100, BATCH = 10): #When Tensor_n and batch differs then 'from_seq_id' is lost
    gc.collect()
    torch.cuda.empty_cache()
    EXP = exp
    test_arr_rationals = run_code_rational( 
            df_generated_input = df_generated_input,
            tensor_size = tensor_n,
            experiment = EXP,
            batch_size = BATCH, 
            model = model, 
            verbose = False 
        )
    #Saving process
    print('Saving process')
    test_arr_rationals.to_csv(param_default()['rational_results'] + '/' + param_default()['dataset'] + '/' + '[t_'+str(tensor_n)+']_[max_tgt_'+str(MAX_TOKEN_SIZE)+']_[exp:' + str(EXP) +']_.csv')
    return test_arr_rationals


# %%
#tst
#df_test_run = run_code_rational_all_set(exp='0')

# %%
for i in df_generated_input.columns[5:]: #Only Generated Sequences 
    df_test_run = run_code_rational_all_set(exp=i, tensor_n=df_generated_input.shape[0])

# %%
df_test_run.head(1)

# %%
#Running all Experiments
def exp_run_all_rationales():
    dict_arr_rations = { key : run_code_rational(
        df_generated_input = df_generated_input,
        experiment = key,
        batch_size = 10, 
        model = model, 
        verbose = False 
    ) for key in df_generated_input.columns[5:] }
    return dict_arr_rations

# %%
#arr_df_rationale = [pandas_rationales(dict_arr_rations[key]) for key in dict_arr_rations.keys()]


