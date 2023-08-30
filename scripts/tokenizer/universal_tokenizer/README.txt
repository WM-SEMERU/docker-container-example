This universal tokenizer was formed by adding whitespace tokens to the end of the GPT2/RoBERTa/BART tokenizer.
In order to use the pretrained fairseq models (besides the original Python and Java seq2seq models that used NEWLINE tokens),
you'll need to use this universal tokenizer and fairseq dict (don't let fairseq generate a dict for you -- use this one!)

