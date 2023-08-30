# %% [markdown]
# # Mapping Function
# > Labels a given token following a tailored taxonomy

# %%
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json
import random
import numpy as np

# %%
from code_rationales.loader import download_grammars
from tree_sitter import Language, Parser
import code_rationales

# %%
import nltk

# %% [markdown]
# ## Setup

# %%
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

# %%
def param_default():
    return {
        #'dataset' : 'code_completion_random_cut_5k_30_512_tokens',
        #'dataset' : 'code_completion_docstring_random_cut_3.8k_30_150_tokens',
        #'dataset' : 'code_completion_docstring_signature_3.8k_30_150_tokens',
        'dataset' : 'code_completion_docstring_5k_30_150_tokens',
        'rational_results': '/workspaces/code-rationales/data/rationales/gpt',
        'global_ast_results': '/workspaces/code-rationales/data/global_ast_results/gpt',
        'global_taxonomy_results': '/workspaces/code-rationales/data/global_taxonomy_results/gpt',
        'delimiter_sequence': '',
        'num_samples' : 100, 
        'size_samples' : 157,
        'num_experiments': 30, 
        'bootstrapping' : 500
    }
params = param_default()

# %% [markdown]
# ## Taxonomy Definitions

# %%
#Programming Language Taxonomy
def pl_taxonomy_python() -> dict:
    return {
  "punctuation": ['{', '}', '[', ']', '(', ')','\"', ',', '.', '...', ';', ':'], 
  "exceptions": ['raise_statement','catch', 'try', 'finally', 'throw', 'throws', 'except'],
  "oop": ['def','class','instanceof','interface','private','protected','public','abstract','extends','package','this','implements','import','new','super'],
  "asserts": ['assert'],
  "types": ['tuple','set','list','pair','subscript','type','none','dictionary','integer','native','static','synchronized','transient','volatile','void','final','enum','byte','char','float','boolean','double','int','long','short','strictfp'],
  "conditionals": ['else', 'if', 'switch', 'case', 'default'],
  "loops": ['break', 'do', 'for', 'while', 'continue'],
  "operators": ['as','yield','is','@','in','and','or','not','**','slice','%','+','<','>','=','+','-','*','/','%','++','--','!','==','!=','>=','<=','&&','||','?',':','~','<<','>>','>>>','&','^','|','//'],
  "indentation": ['\n','\t'],
  "bool": ['true', 'false'], 
  "functional":['lambda','lambda_parameters'],
  "with" : ['with','with_item','with_statement','with_clause'], 
  "return" :['return'],
  "structural" : ['attribute','module', 'argument_list','parenthesized_expression','pattern_list','class_definition','function_definition','block'],
  "statements" : ['return_statement','break_statement','assignment','while_statement','expression_statement','assert_statement'],
  "expression": ['call','exec','async','ellipsis','unary_operator','binary_operator','as_pattern_target','boolean_operator','as_pattern','comparison_operator','conditional_expression','named_expression','not_operator','primary_expression','as_pattern'],
  "errors": ["ERROR"],
  "identifier":["identifier"],  
  "comment":["comment"],
  "string": ['string','interpolation','string_content','string_end','string_start','escape_sequence'], 
  "unknown": []
}

# %%
def nl_pos_taxonomy() -> dict: return {
    "nl_verb" : ['VBN', 'VBG', 'VBZ', 'VBP', 'VBD', 'VB'],
    "nl_noun" : ['NN', 'NNPS', 'NNS', 'NNP'],
    "nl_pronoun" : ['WP', 'PRP', 'PRP$', 'WP','WP$'], 
    "nl_adverb" : ['RBS','RBR', 'RB', 'WRB'], 
    "nl_adjetive" : ['JJR', 'JJS', 'JJ'], 
    "nl_determier" : ['DT','WDT','PDT'], 
    "nl_preposition" : ['IN', 'TO'],
    "nl_particle" : ['RP'],
    "nl_modal" : ['MD'],
    "nl_conjunction" : ['CC'],
    "nl_cardinal" : ['CD'],
    "nl_list": ['LS'],
    "nl_other" : ['FW', 'EX', 'SYM' , 'UH', 'POS', "''", '--',':', '(', ')', '.', ',', '``', '$']
}

# %% [markdown]
# ## AST Mapping

# %%
languages=['python', 'java']
download_grammars(languages)

# %%
def unroll_node_types(
    nested_node_types: dict  # node_types from tree-sitter
) -> list: # list of node types
    def iterate_and_unroll_dict(nested_node_types: dict, all_node_types: set):
        for key, value in nested_node_types.items():
            if key == 'type' and type(value) == str:
                all_node_types.add(value)
            if type(value) == dict:
                iterate_and_unroll_dict(value, all_node_types)
            if type(value) == list:
                for element in value:
                    iterate_and_unroll_dict(element, all_node_types) 
    all_node_types = set()
    for dictionary in nested_node_types:
        iterate_and_unroll_dict(dictionary, all_node_types)
    all_node_types.add('ERROR')
    return list(all_node_types)

# %%
def create_parser(lang: str):
    # Grab the node types from the tree-sitter language
    language = Language(f"{code_rationales.__path__[0]}/grammars/tree-sitter-languages.so", lang)
    node_path = f"{code_rationales.__path__[0]}/grammars/tree-sitter-{lang}/src/node-types.json"
    with open(node_path) as f:
            node_types = json.load(f)
    node_types = unroll_node_types(node_types)
    # Create a parser for the language
    parser = Parser()
    parser.set_language(language)
    return parser, node_types

# %%
def traverse(
    node,       # tree-sitter node
) -> None:
    """Traverse in a recursive way, a tree-sitter node and append results to a list."""
    results = []
    def traverse_tree(node, results):
        if node.type == 'string':
            results.append(node)
            return
        for n in node.children:
            traverse_tree(n, results)
        if not node.children:
            results.append(node)
    traverse_tree(node, results)
    return results

# %%
def convert_to_offset(
    point,              #point to convert
    lines: list         #list of lines in the source code
    ):
        """Convert the point to an offset"""
        row, column = point
        chars_in_rows = sum(map(len, lines[:row])) + row
        chars_in_columns = len(lines[row][:column])
        offset = chars_in_rows + chars_in_columns
        return offset

# %%
def get_node_span(node, lines):
    """Get the span position of the node in the code string"""
    start_span = convert_to_offset(node.start_point, lines)
    end_span = convert_to_offset(node.end_point, lines)
    return start_span, end_span
    

# %%
def is_token_span_in_node_span(tok_span, node_span):
    if (node_span[0] <= tok_span[0] and tok_span[0] < node_span[1]) or (node_span[0] < tok_span[1] and tok_span[1] <= node_span[1]):
        return True
    return False

# %%
def get_token_type(
    tok_span: tuple, # (start, end) position of a token in tokenizer
    nodes: list,     # list of tree-sitter nodes
    lines: list,     # list of lines in the code
) -> tuple: # (parent_type, token_type) of the token
    """Get the parent AST type and token AST type of a token."""
    node_spans = [get_node_span(node, lines) for node in nodes]
    for i, span in enumerate(node_spans):
        if is_token_span_in_node_span(tok_span, span):
            return nodes[i].parent.type, nodes[i].type

# %%
def get_token_nodes(
    tok_span: tuple, # (start, end) position of a token in tokenizer
    node,            # tree-sitter node
    lines: list,     # list of lines in the code
) -> list: 
    """Get all AST types for the given token span"""
    results = []
    def traverse_and_get_types(tok_span, node, lines, results) -> None:
        node_span = get_node_span(node, lines)
        if (node_span[0] <= tok_span[0] and tok_span[0] < node_span[1]) or (node_span[0] < tok_span[1] and tok_span[1] <= node_span[1]):
            results.append(node)
        for n in node.children:
            traverse_and_get_types(tok_span, n, lines, results)
    traverse_and_get_types(tok_span, node, lines, results)
    return results

# %%
def get_nodes_by_type(
    node, 
    node_types: list
) -> list :
    def traverse_and_search(node, node_types, results):
        if node.type in node_types:
            results.append(node)
            return
        for n in node.children:
            traverse_and_search(n, node_types ,results)
    results = []
    traverse_and_search(node, node_types, results)
    return results

# %% [markdown]
# ## Taxonomy Mapping 

# %%
def clean_results(global_results):
    def clean_dictonary(result_dict):
        clean_dict = result_dict.copy()
        for key, value in result_dict.items():
            if not value:
                clean_dict.pop(key)
        return clean_dict
    for key, value in global_results.items():
        global_results[key] = clean_dictonary(value)
    return clean_dictonary(global_results)

# %%
def search_category_by_token(taxonomy_dict: dict, token_type: str):
    for key, value in taxonomy_dict.items():
        if token_type in value:
            return key
    return 'unknown'

# %%
def map_to_taxonomy(taxonomy_dict: dict, result_dict: dict):
    result_dict = result_dict.copy()
    mappings = {token: {category : [] for category in taxonomy_dict.keys()} for token in taxonomy_dict.keys()}
    for target_token, value in result_dict.items():
        for source_token, probs in value.items():
            mappings[search_category_by_token(taxonomy_dict, target_token)][search_category_by_token(taxonomy_dict, source_token)]+=probs
    return clean_results(mappings)

# %%
def map_global_results_to_taxonomy(taxonomy_dict:dict, global_results: dict):
    return dict(zip(global_results.keys(), map(lambda aggegrations: map_to_taxonomy(taxonomy_dict, aggegrations), global_results.values())))

# %% [markdown]
# ##  Rationals Tagging

# %%
calculate_right_span = lambda start_idx, end_idx, df : len(''.join(map(str, df.loc[start_idx:end_idx, 'goal_token'].tolist())))
calculate_span = lambda right_span, token : (right_span-len(str(token)), right_span)

# %%
def add_auxiliary_columns_to_experiment_result(df, delimiter_sequence: str):
    initial_token = eval(df['typesets_tgt'][0])[0][0]
    ### TOKEN TYPE COLUMN
    token_type_column = ['src'] * len(df)
    sequence = initial_token
    for idx, goal_token in enumerate(df['goal_token']):
        if delimiter_sequence not in sequence:
            token_type_column[idx] = 'nl'
            sequence+=goal_token
    df['token_type'] = token_type_column
    ### TOKEN SPAN COLUMN - CHECK FOR DATASETS
    src_initial_token_idx = df[df['token_type']== 'src'].first_valid_index()
    df['span'] = [None] * len(df[:src_initial_token_idx]) + [calculate_span(calculate_right_span(src_initial_token_idx, index, df), token) for index, token in df[src_initial_token_idx:]['goal_token'].items()]


# %%
def fill_nl_tags_in_experiment_result(df, nl_ast_types, nl_pos_types, parser):
    initial_token = eval(df['typesets_tgt'][0])[0][0]
    ##### POS TAGS FOR NL PART
    target_nl = initial_token + ''.join(df[df['token_type'] == 'nl']['goal_token'].map(lambda value: str(value)))
    pos_tags = nltk.pos_tag(nltk.word_tokenize(target_nl))
    for idx in range(df[df['token_type']== 'src'].first_valid_index()):
        nl_tags = list(map(lambda tag: tag[1] if tag[1] in nl_pos_types else None, filter(lambda tag: tag[0] in str(df['goal_token'][idx]), pos_tags)))
        if nl_tags: df.at[idx, 'tags'] = df['tags'][idx] + [nl_tags[-1]]
    ##### POS TAGS FOR CODE PART
    target_code = ''.join(df[df['token_type'] == 'src']['goal_token'].map(lambda value: str(value)))
    nl_target_nodes = get_nodes_by_type(parser.parse(bytes(target_code, 'utf8')).root_node, nl_ast_types)
    for token_idx in range(df[df['token_type'] == 'src'].first_valid_index(), len(df['span'])):
                for nl_target_node in nl_target_nodes:
                    if is_token_span_in_node_span(df['span'][token_idx], get_node_span(nl_target_node, target_code.split("\n"))) and \
                            str(df['goal_token'][token_idx]) in nl_target_node.text.decode('utf-8'):
                            tagged_token_list = list(filter(lambda tagged_token: tagged_token[0] in str(df['goal_token'][token_idx]), \
                                                        nltk.pos_tag( nltk.word_tokenize(nl_target_node.text.decode('utf-8')))))
                            if len(tagged_token_list)>0 and tagged_token_list[0][1] in nl_pos_types and tagged_token_list[0][1] not in df['tags'][token_idx]: df['tags'][token_idx].append(tagged_token_list[0][1])

# %%
def fill_ast_tags_in_experiment_result(df, parser):
    initial_token = eval(df['typesets_tgt'][0])[0][0] if df[df['token_type'] == 'src'].first_valid_index() == 0 else ''
    target_code = initial_token + ''.join(df[df['token_type'] == 'src']['goal_token'].map(lambda value: str(value)))
    src_initial_token_idx = df[df['token_type']== 'src'].first_valid_index()
    target_ast = parser.parse(bytes(target_code, 'utf8')).root_node
    for token_idx in range(src_initial_token_idx, len(df)):
        df.at[token_idx, 'tags'] = df['tags'][token_idx] + list(map(lambda node: node.type, get_token_nodes(df['span'][token_idx], target_ast, target_code.split("\n"))))

# %%
def tag_rationals(experiment_paths: list, nl_ast_types: list, nl_pos_types: list, delimiter_sequence: str, parser):
    experiments = {}
    for exp_idx, experiment_path in enumerate(experiment_paths):
        experiment_results = []
        df_experiment = pd.read_csv(experiment_path, index_col=0)
        experiment_rational_results = [df_experiment[(df_experiment['from_seq_id'] == sample_idx) | \
                                                     (df_experiment['from_seq_id'] == str(sample_idx))].reset_index() \
                                                    for sample_idx in range(params['num_samples'])]
        print('*'*10 +'Tagging rationals for exp: ' +str(exp_idx) + '*'*10)
        for experiment_rational_result in experiment_rational_results:
            add_auxiliary_columns_to_experiment_result(experiment_rational_result, delimiter_sequence)
            experiment_rational_result['tags'] = [[]]*len(experiment_rational_result)
            fill_nl_tags_in_experiment_result(experiment_rational_result, nl_ast_types, nl_pos_types, parser)
            fill_ast_tags_in_experiment_result(experiment_rational_result, parser)
            experiment_results.append(experiment_rational_result)
        experiments[exp_idx] = experiment_results
    return experiments
            

# %% [markdown]
# ## Rationals Aggregation

# %%
def aggregate_rationals(global_tagged_results: dict, ast_node_types: list, nl_pos_types: list):
    aggregation_results = {exp_id: None for exp_id in global_tagged_results.keys()}
    for exp_idx, experiment_results in global_tagged_results.items():
        experiment_aggregation_results = {node_type : {node_type : [] for node_type in ast_node_types + nl_pos_types} for node_type in ast_node_types + nl_pos_types}
        print('*'*10 +'Aggregrating rationals for exp: ' +str(exp_idx) + '*'*10)
        for result_idx, experiment_result in enumerate(experiment_results):
            for target_idx in range(len(experiment_result)):
                for rational_idx, rational_pos in enumerate(eval(experiment_result['rationale_pos_tgt'][target_idx])):
                    try:
                        [experiment_aggregation_results[target_tag][rational_tag].append(eval(experiment_result['rationale_prob_tgt'][target_idx])[rational_idx]) if target_tag and rational_tag else None \
                         for rational_tag in experiment_result['tags'][rational_pos] for target_tag in experiment_result['tags'][target_idx]]
                    except Exception as e:
                        print('An Error Occurred')
            print('r-'+str(result_idx))
        aggregation_results[exp_idx] = clean_results(experiment_aggregation_results)
    return aggregation_results

# %% [markdown]
# ## Bootstrapping

# %%
def bootstrapping( np_data, np_func, size ):
    """Create a bootstrap sample given data and a function
    For instance, a bootstrap sample of means, or mediands. 
    The bootstrap replicates are a long as the original size
    we can choose any observation more than once (resampling with replacement:np.random.choice)
    """
    
    #Cleaning NaNs
    #np_data_clean = np_data[ np.logical_not( np.isnan(np_data) ) ] 
    
    #The size of the bootstrap replicate is as big as size
    #Creating the boostrap replicates as long as the orignal data size
    #This strategy might work as imputation 
    bootstrap_repl = [ np_func( np.random.choice( np_data, size=len(np_data) ) ) for i in range( size ) ]
    
    #logging.info("Covariate: " + cov) #Empirical Mean
    #logging.info("Empirical Mean: " + str(np.mean(np_data_clean))) #Empirical Mean
    #logging.info("Bootstrapped Mean: " + str( np.mean(bootstrap_repl) ) ) #Bootstrapped Mean
    
    return np.array( bootstrap_repl )

# %%
def bootstrap_samples_global_results(global_results: dict, size: int):
    for exp_id in global_results.keys():
        experiment_result = global_results[exp_id]
        for target_type, target_value in global_results[exp_id].items():
            for source_type, source_value in target_value.items():
                experiment_result[target_type][source_type] = bootstrapping(source_value, np.mean, size).tolist()
        global_results[exp_id] = experiment_result

# %% [markdown]
# ## Running Experiment

# %%
### Retrieve experiments
get_experiment_path =  lambda samples, size, exp: params['rational_results'] + '/' + params['dataset'] + '/' + '[t_'+str(samples)+']_[max_tgt_'+str(size)+']_[exp:'+str(exp)+'].csv'
experiment_paths = [get_experiment_path(params['num_samples'], params['size_samples'], exp) for exp in range(params['num_experiments'])]
### Define parser
parser, node_types = create_parser('python')
### Defines pos tags 
pos_types = list(nltk.data.load('help/tagsets/upenn_tagset.pickle'))

# %%
###TAG EXPERIMENTS RESULTS - TAKES TIME
nl_ast_types = ['comment','identifier','string']
global_tagged_results = tag_rationals(experiment_paths, nl_ast_types, pos_types, params['delimiter_sequence'], parser)

# %%
###AGGREGATE RATIONALS - TAKES TIME
global_aggregated_results = aggregate_rationals(global_tagged_results, node_types, pos_types)

# %%
###GROUP AGGREGATES BY TAXONOMY
taxonomy = {**pl_taxonomy_python(), **nl_pos_taxonomy()}
global_taxonomy_results = map_global_results_to_taxonomy(taxonomy, global_aggregated_results.copy())

# %%
### BOOTSTRAPPING - TAKES TIME
bootstrap_samples_global_results(global_taxonomy_results, params['bootstrapping'])

# %% [markdown]
# ## Storing Results

# %%
for exp_id, exp_aggregation in global_aggregated_results.items():
    with open(params['global_ast_results'] + '/' + params['dataset'] +'_exp_' + str(exp_id) +'.txt', 'w') as file:
        file.write(json.dumps(exp_aggregation))

# %%
for exp_id, exp_aggregation in global_taxonomy_results.items():
    with open(params['global_taxonomy_results'] + '/' + params['dataset'] +'_exp_' + str(exp_id) +'.txt', 'w') as file:
        file.write(json.dumps(exp_aggregation))


