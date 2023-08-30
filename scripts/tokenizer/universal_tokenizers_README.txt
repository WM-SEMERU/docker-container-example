See the section on tokenizing from my notes https://www.overleaf.com/project/5e7536614b0d3600011d3c43

We use the Huggingface ByteLevel tokenizer, which is very fast, handles whitespace well by default, is reversible, and extends the gpt2/roberta/bart tokenizer

The universal tokenizer is formed by extending the gpt2/roberta/bart tokenizer by adding whitespace tokens (e.g. the 12-space token).
I plan on using the universal tokenizer for all my future models

The pythonNEWLINEs tokenizer was trained after first replacing '\n' with 'NEWLINE'
I used it for the python pretrained model (and the method-generation model finetuned from it)

In order to run on Fairseq, you also need to supply the fairseq dict.txt in addition to tokenizing
I normally tokenize so there are 1021 tokens per line (less than 1024 because tokens like <bos> and <eos> are added by fairseq)