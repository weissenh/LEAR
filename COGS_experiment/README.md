# Learning Algebraic Recombination for Compositional Generalization - COGS

This repository is the extention implementation on COGS, based on the basic idea in our paper "Learning Algebraic Recombination for Compositional Generalization". 


## Requirements

Our code is officially supported by Python 3.7. The main dependency is `pytorch`.
You could install all requirements by the following command:

```bash
pip install -r requirements.txt
```

## Data Files

- `/cogs_data/`: COGS dataset containing train, dev and test. 
  - need to unzip it first
  - source: https://github.com/najoungkim/COGS
  - note: doesn't contain the larger `train_100.tsv`
- `/preprocess/`: Contains encode tokens, decode tokens and phrase table.
  - `caus_predicate`: output tokens defining caus predicates (triggered by verbs on the input side)
  - `unac_predicate`: output tokens defining unac predicates (triggered by verbs on the input side)
  - `encode_tokens.txt`: input tokens (all tokens on the input side, even including punctuation, copula verb etc)
  - `enct2dect`: json-object mapping input tokens to output tokens (phrase table): described in appendix D, [page 13f](https://aclanthology.org/2021.findings-acl.97.pdf#page=13) in Liu et al. 2021.
  - `entity`: output tokens defining entities (triggered by nouns on the input side)
  - `example2type`: json-object mapping sentences from the gen set to generalization types
  - **to do**: how were the preprocess files generated? code not published

## Training

To train our model on COGS datasets, you could use this command:

```bash
python3 main.py --mode train --checkpoint <model_dir> --task cogs
```

📋 Note that `<model_dir>` specifies the store folder of model checkpoints.

The corresponding log and model weights will be stored in the path `checkpoint/logs/` and `checkpoint/models/` respectively

Example call:  
```bash
python3 main.py --mode train --checkpoint model2 --task cogs
```
will create `./checkpoint/logs/model2.log` and a few `.mdl` files in `./checkpoint/models/model2/`.

The `checkpoint/logs` and `checkpoint/models` relative paths are hard-coded in 
`main.py:prepare_arguments` in the `args` dictionary variable,
but you can also provide your own paths, using one or both of the optional `--model-dir` and `--logs-path`options:
`--model-dir ./checkpoint/models/mymodel --logs-path ./checkpoint/logs/mymodel`, 
If you provide both (model dir and logs path), then you still need to provide 
the `--checkpoint` option, although it isn't used internally then.

*Tipp*: by default 2 is used as the random seed, change this by adding `--random-seed INTEGER`


## Evaluation

The accuracy on gen set will be printed as the pre-last line in log after training.
E.g. use `tail -n 2 ./checkpoint/logs/<model_dir>.log`  

**NB:**   
during training (and including the final run on the gen set performed by a `validate()` call), 
skipped sentences (`skip_count`) are excluded from the accuracy computation.
A sentence might get skipped if it contains an out-of-vocabulary token 
(i.e. never seen word in train set), e.g. 'monastery' or 'gardner' don't appear 
in `train.tsv` at all, but in the dev set and/or gen set.  
PW: changed code for testing (but not for training!) to include skipped 
sentences in the accuracy computation 
(`accuracy_meter` now also called in the try-except-continue part).

You can also use a trained model as your checkpoint and test it on the generalization set:
```bash
python3 main.py --mode test --checkpoint <model_dir> --task cogs
```

Example call:  
```bash
python3 main.py --mode test --checkpoint ./checkpoint/models/model2/0-final.mdl --task cogs
```


--------------------

For comparison find the results on the COGS generalization set reported in Liu et al. 2021, [Table 4 on page 1134](https://aclanthology.org/2021.findings-acl.97.pdf#page=6) below:

| Model                 |      Accuracy |
| :-------------------- | ------------: |
| LeAR                  | 97.7 (+- 0.7) |
| w/o Abstraction       | 94.5 (+- 2.8) |
| w/o Semantic locality | 94.0 (+- 3.6) |
| w/o Tree-LSTM         | 80.7 (+- 4.3) |


## References  
- Liu et al. (2021). [Learning Algebraic Recombination for Compositional Generalization](https://aclanthology.org/2021.findings-acl.97/).
  In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*. pp. 1129--1144. DOI: [10.18653/v1/2021.findings-acl.97](http://dx.doi.org/10.18653/v1/2021.findings-acl.97)
