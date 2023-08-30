# %% [markdown]
# # Athena Rationales Global
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
params['output_results']

# %% [markdown]
# ## Rationalizations Utilities

# %%
rationalization = importlib.import_module("sequential-rationales.huggingface.rationalization")
rationalize = rationalization.rationalize_lm
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Universal Tokenizer

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
#Setting experiments 
#! export CUDA_VISIBLE_DEVICES="0,1"

# %%
## Move model to GPU if available and trigger evaluation mode
def model_activate(model = model):
  if torch.cuda.is_available():
    model.cuda()
    model.eval()
    model.model = model.models[0]
    model.device
    print("Model Activated")
  pass

# %% [markdown]
# ## Data Loading and Testing

# %%
#Loading Code Generation
df_generated_input = pd.read_json( params['output_results'] + '1_generation_[max:100]_02.json' )

# %%
df_generated_input.columns[4:] #Tensor Columns

# %%
print('df readit')
df_generated_input.head()

# %%
df_generated_input.shape

# %%
#tst decoding
decoded = model.decode(df_generated_input['0'][0])
decoded

# %%
prettify_java( lazy_decode( decoded ) )

# %% [markdown]
# ## Running Rationales

# %%
#Statistics
np.mean( [len(i) for i in df_generated_input['0'].values] )

# %%
#TODO Run the distribution of each experiment. The mean value of tokens or size for each experiment. 
np.mean( [len(i) for i in df_generated_input['source_sampling'].values] )

# %%
len(df_generated_input['0'].values[2])

# %%
MAX_TOKEN_SIZE = 100 #Hardocoded!!

# %%
#If the model is not fine-tuned or compatible, it will rise an error
#Bear in mind that Athena is a Translation model (not a language one)
#This function works for one tensor of source token and one tensor of target tokens
def rationalize_model(t_source_tokens, t_target_tokens, model, verbose=True):
    all_source_rationales, all_target_rationales, log = rationalize_conditional_model(
        model = model, 
        source_tokens = t_source_tokens, #[:MAX_TOKEN_SIZE],
        target_tokens = t_target_tokens[:MAX_TOKEN_SIZE], 
        verbose=verbose,
        max_steps=1024 #Max number of steps for greedy rationalization
    )
    return all_source_rationales, all_target_rationales, log 

# %%
#tst <--- TestCase1
def tst_rationalize_model():
    gc.collect()
    torch.cuda.empty_cache() #Cleaning Cache
    model_activate(model = model)

    t_dict_generated_input = { exp : [ torch.tensor(s).to(model.device) for 
                s in df_generated_input[exp].values ] for exp in df_generated_input.columns[4:]  } #Tensor Columns only

    rationalize_model(  
        t_source_tokens =  t_dict_generated_input['source_sampling'][0],
        t_target_tokens =  t_dict_generated_input['0'][0],
        model = model 
    )
    pass

#tst_rationalize_model()

# %%
def run_multiple_rational(
    arr_source_tokens, 
    arr_target_tokens, 
    model, 
    seq_id, #mapping sequence id
    verbose=True
):
    arr_log = []
    for index,val in enumerate( arr_source_tokens ):
        _, _, log = rationalize_model(
            t_source_tokens = val, 
            t_target_tokens = arr_target_tokens[index], 
            model = model,
            verbose = verbose )
        arr_log.append(log)
    arr_code_rationales = [ log['rationalizations'] for log in arr_log ] #extracting just rationalizations
    arr_from_sentence = [ list(np.full( len(val), seq_id[arr_i] )) #arr_i maps to the real sequence id
                            for arr_i, val in enumerate(arr_code_rationales)]
    
    arr_code_rationales = sum( arr_code_rationales, [] ) #flatting
    arr_from_sentence = sum( arr_from_sentence, [] ) #flatting
    
    return arr_code_rationales, arr_from_sentence
    #return arr_code_rationales

# %%
import gc

# %%
#tst <------- Test Case 2
def tst_run_multiple_rationa():
    
    gc.collect()
    torch.cuda.empty_cache() #Cleaning Cache
    model_activate(model = model)

    t_dict_generated_input = { exp : [ torch.tensor(s).to(model.device) for 
                s in df_generated_input[exp].values ] for exp in df_generated_input.columns[4:]  }
    
    arr_rations, seq_id = run_multiple_rational(
        arr_source_tokens =  t_dict_generated_input['source_sampling'][:2], #With 2 Sequences  
        arr_target_tokens =  t_dict_generated_input['0'][:2], 
        model = model,
        seq_id = list( range(2,4) ),
        verbose = False
        )
    return arr_rations, seq_id
#tst_arr_rations, seq_id = tst_run_multiple_rationa()

# %%
def pandas_rationales( arr_code_rationales, arr_from_sentence ):
    #Creating pandas_1 {p_rationale}
    rational = lambda list_log,typeset: [ (dict_tok['added_token_text'],round(dict_tok['true_token_prob'],6)) for dict_tok in list_log if dict_tok['from']==typeset]
    log_from = lambda log_row,typeset: [(log_dict['added_token_text'],log_dict['true_token_prob']) for log_dict in log_row if log_dict['from']==typeset] #Typeset

    log_position = lambda log_row,typeset: [log_dict['added_token_position'] for log_dict in log_row if log_dict['from']==typeset] #Position of the Rationale
    log_prediction = lambda log_row,typeset: [log_dict['true_token_prob'] for log_dict in log_row if log_dict['from']==typeset] #Rationale Prob

    p_rationale = pd.DataFrame()

    p_rationale['goal_token'] = [dict_token['goal_word'] for dict_token in arr_code_rationales]
    p_rationale['from_seq_id'] = arr_from_sentence

    p_rationale['typesets_tgt'] = [ log_from(log_row,'target') for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]
    p_rationale['typesets_src'] = [ log_from(log_row,'source') for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]


    p_rationale['rationale_pos_tgt'] = [ log_position(log_row,'target') for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]
    p_rationale['rationale_pos_src'] = [ log_position(log_row,'source') for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]

    p_rationale['rationale_prob_tgt'] = [ log_prediction(log_row,'target') for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]
    p_rationale['rationale_prob_src'] = [ log_prediction(log_row,'source') for log_row in [dict_token['log'] for dict_token in arr_code_rationales]]

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
        model_activate(model = model)
        print('************************' + str(i) + '************************')
        t_generated_input = df_generated_input[ experiment ].values[i:i+batch_size]
        t_source_sampling = df_generated_input['source_sampling'].values[i:i+batch_size]

        t_generated_input = [ torch.tensor(s).to(model.device) for s in t_generated_input]
        t_source_sampling = [ torch.tensor(s).to(model.device) for s in t_source_sampling]

        
        t_arr_rationals,t_arr_from_seq = run_multiple_rational(
            arr_source_tokens =  t_source_sampling, 
            arr_target_tokens =  t_generated_input, 
            model = model,
            seq_id = list( range(i,i+batch_size) ),
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
gc.collect()
torch.cuda.empty_cache()

# %%
torch.cuda.is_available()

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
    #Saving process
    #print('Saving process')
    #test_arr_rationals.to_json( params['output_results'] + 'rationales_[t_1000]_[max_100]_02_' + EXP )
    return test_arr_rationals
#df_test_run = tst_run_code_rational_sampling_set()

# %%
#tst
#df_test_run[ df_test_run['from_seq_id'] == 1]

# %%
def run_code_rational_all_set(exp, tensor_n = 1000, BATCH = 100): #When Tensor_n and batch differs then 'from_seq_id' is lost
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
    test_arr_rationals.to_json( params['output_results'] + 'rationales_1_gen_02/' + 'rationales_[t_1000]_[max_src_100]_[max_tgt_100]_02_[exp:' + EXP +']_.json' )
    return test_arr_rationals


# %%
#tst
#df_test_run = run_code_rational_all_set(exp='0')

# %%
for i in df_generated_input.columns[4:-1]: #Only Generated Sequences 
    df_test_run = run_code_rational_all_set(i)

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
    ) for key in df_generated_input.columns[4:-1] }
    return dict_arr_rations

# %%
#arr_df_rationale = [pandas_rationales(dict_arr_rations[key]) for key in dict_arr_rations.keys()]


