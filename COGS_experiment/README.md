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
  - `enct2dect`: json-object mapping input tokens to output tokens (phrase table/alignment file): described in appendix D, [page 13f](https://aclanthology.org/2021.findings-acl.97.pdf#page=13) in Liu et al. 2021.
  - `entity`: output tokens defining entities (triggered by nouns on the input side)
  - `example2type`: json-object mapping sentences from the gen set to generalization types
  - **to do**: how were the preprocess files generated? code not published, PW working on own reimplementation: `do_preprocessing.py`

These file paths and file names are hard-coded in various places inside `main.py`

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


**Customizations**  
1. *Specifying own paths for model weights and log files*:  
The `checkpoint/logs` and `checkpoint/models` relative paths are hard-coded in 
`main.py` in global variables at the top of the file and present the default 
values for `--model-dir-prefix` and `--logs-path-prefix` command line options 
respectively. Note that these two are only prefixes, so you still have provide
the `--checkpoint` option, as it will specify the file or folder name within 
logs and model directory respectively. So internally it is used
`--model-dir-prefix ./checkpoint/models/ --logs-path-prefix ./checkpoint/logs/`
2. *Changing the random seed*:  
The default is random seed 2, you can change it by adding `--random-seed SOMEINTEGER`
3. *Further options*:  
Run `python3 main.py --help` to list all available options.


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

Note that the output is not a TSV file with logical form like the gold file.
Rather so far the code can only output the internal representation used by LeAR,
see `gen_right.txt` and `gen_wrong.txt` files (output when run it test mode):
```
obj_pp_to_subj_pp
a girl beside a nest liked that the pickle on a book doubled
Gold LF:  * pickle ( x _ 8 ) ; girl ( x _ 1 ) and girl . nmod . beside ( x _ 1 , x _ 4 ) and nest ( x _ 4 ) and like . agent ( x _ 5 , x _ 1 ) and like . ccomp ( x _ 5 , x _ 12 ) and pickle . nmod . on ( x _ 8 , x _ 11 ) and book ( x _ 11 ) and double . theme ( x _ 12 , x _ 8 )
Pred:  like girl beside nest None None ccomp double pickle on book None None
Gold:  like girl beside nest None None ccomp double None pickle on book None

```
The accuracy is therefore also not calculated on the logical forms directly,
but on these internal representations (see `model.py:HRLModel.get_reward()` about line 900, and e.g. `main.py:test()`about line 408: accuracy only 1 if reward 1 else 0)

By the way, the effect of the determiner 
(`*` for definite, but not indefinite determiners in the logical form)
is not considered here:
- `model.py:HRLModel.process_output()`, that's about Line 979, tied to that:
- the output above: notice no difference in internal representation between indefinite 'a girl' and definite 'the pickle'
- and that words with no alignments ignored: `forward()` of bottom abstractor: only if alignment found ("the" not in enct2dect found: no alignment) added to bottom-span 
- also not found in semantic (P/E classes side)

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
