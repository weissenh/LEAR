#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script is NOT part of the official code submission
# and NOT written by one of the paper's authors.
# author: weissenh
# (trying to reproduce the files under /preprocess,
# with the goal to also apply LeAR to train_100.tsv)
"""
Files under /preprocess/
- caus_predicate
- unac_predicate
- entity
- encode_tokens.txt
- enct2dect
- example2type
"""


import sys  # for argc,argv and exit
import os.path as path
import json
from collections import Counter


def get_sample_generator(tsv_file: str) -> tuple:
    """
    Iterates over the rows in the tsv file and yields triples

    :param tsv_file: path to tsv file with 3 columns
    :return: (yield generator) triples, represnting one row in the tsv file
    """
    with open(tsv_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if len(line) == 0:
                continue
            columns = line.split("\t")
            assert(len(columns) == 3)  # input sentence, logical form, gen type
            # sentence, logical_form, gentype = tuple(columns)
            yield tuple(columns)


def create_encode_tokens(train_file: str, output_file: str):
    """
    Create encode_tokens.txt : all tokens from the input site

    :param train_file: input file from which to collect tokens of all sentences
    :param output_file: where to write one token per line
    :return: None
    """
    encode_tokens = set()
    for file in [train_file]:  # , test_file, , dev_file
        for sentence, logical_form, _ in get_sample_generator(file):
            tokens = sentence.lower().split()
            encode_tokens.update(tokens)
    with open(output_file, "w", encoding="utf-8") as outf:
        for token in encode_tokens:
            outf.write(token+"\n")
        # outf.write("\n")
    print(f"encode_tokens.txt: seen {len(encode_tokens)} tokens")
    return


# todo also add dev set?
# todo refactor? test this?
def create_enct2dect(train_file: str, output_file: str):
    """
    Create phrase table

    Pre-clean:
    remove some words from input, output (semantically vacous, ..)
    First Step:
    find the output words that exactly co-occur with each input words.
    Formally find (w,v) input-output pairs s.t. for all samples:
    occurs(w, sample) ==> occurs(v, sample)
    Second Step: clean the alignments.
    As there are some input words that only have one candidate (Emma: emma),
    we remove these candidates from other alignment pairs which contains more than one candidates (remove emma).
    Step 3:
    Repeat the second step until all input words only have one alignment.

    731 for train
    732 for train_100 ( 'monastery' )
    :param train_file: input file
    :param output_file: 1-line json-object, dict with encodertoken : decodertoken pairs
    :return: None
    """
    in2out = dict()
    out2in = dict()
    in_counter, out_counter = Counter(), Counter()
    dec_exclude = {"and", ".", "*", ";", "(", ")", ",", "_", "x",
                   "a", "b", "e", "lambda",
                   "agent", "theme", "recipient", "xcomp", "ccomp", "nmod"}
    preps = {"in", "on", "beside"}  # todo why remove prepositions?
    # 0. Get co-occurences of input and output tokens
    for file in [train_file]:  # , test_file, , dev_file
        for sentence, logical_form, _ in get_sample_generator(file):
            sentence, logical_form = sentence.lower(), logical_form.lower()
            enc_tokens = {t for t in sentence.split() if t != '.' and t not in preps}
            dec_tokens = {t for t in logical_form.split() if not(t in dec_exclude or t.isdigit() or t in preps)}
            for enc_token in enc_tokens:
                if enc_token not in in2out:
                    in2out[enc_token] = Counter()
                in2out[enc_token].update(dec_tokens)
                in_counter[enc_token] += 1
            for dec_token in dec_tokens:
                if dec_token not in out2in:
                    out2in[dec_token] = Counter()
                out2in[dec_token].update(enc_tokens)
                out_counter[dec_token] += 1
    # 2. Keep only the (e,d) pairs for which  e ==> d holds (if e present, then d is present)
    exact_cooc = dict()
    for enc_token, dec_counter in in2out.items():
        n_enct = in_counter[enc_token]
        for dec_token, n_encdec in dec_counter.items():
            # n_dect = out_counter[dec_token]
            if n_enct == n_encdec:  # == n_encdec  n_dect ==
                if enc_token not in exact_cooc:
                    exact_cooc[enc_token] = set()
                exact_cooc[enc_token].add(dec_token)
    in2out = exact_cooc
    # 3. Clean alignments: if found 1:1 pair,
    # remove it from the dicts and also all its occurences with other tokens
    finalized_pairs = dict()
    maxlen = max(len(dects) for dects in in2out.values())
    minlen = min(len(dects) for dects in in2out.values())
    loops = 0
    while maxlen >= 1 and minlen == 1 and loops < 5:  # use 'loops' counter to prevent infinite loop
        loops += 1
        in_remove, out_remove = set(), set()
        # find new 1:1 pairs
        for enct, dects in in2out.items():
            if len(dects) == 1:
                in_remove.add(enct)
                out_remove.update(dects)
                finalized_pairs[enct] = dects.pop()
        # remove input and output tokens that are part of these new 1:1 pairs
        for in_token in in_remove:
            del in2out[in_token]
        for enct, dects in in2out.items():
            in2out[enct] = {dect for dect in dects if dect not in out_remove}  # todo what if 0?
        if len(in2out) > 0:
            maxlen = max(len(dects) for dects in in2out.values())
            minlen = min(len(dects) for dects in in2out.values())
        else:
            maxlen, minlen = 0, 1
    if minlen != 1:  # loops == 30 or
        # print(f"WARNING: reach maximum number of loops ({loops}). "
        print(f"WARNING: couldn't get 1:1 mappings for all input tokens (loops done: {loops}).  "
              f"Max number of output tokens per input token: {maxlen}")
    # if loops == 0 and minlen == maxlen == 1:
    #     finalized_pairs = exact_cooc
    print(f"enct2dect: found {len(finalized_pairs)} pairs")  # 731 for train, dev, test, gen?
    with open(output_file, "w", encoding="utf-8") as outfile:
        # outfile.write("{")
        outfile.write(json.dumps(finalized_pairs))
    return


# todo in preprocess/example2type they are additionally sorted by gen type
def create_example2type_file(gen_file: str, output_file: str):
    """
    From the gen file get a 'sentence' : 'generalization_type' dict serialized

    :param gen_file: path to 'gen.tsv' file, e.g. `./cogs_data/gen.tsv`
    :param output_file: output file path, e.g. `./preprocess/example2type`
    :return: None
    """
    seen_samples = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("{")
        for sentence, _, gentype in get_sample_generator(gen_file):
            assert(sentence.endswith(" ."))  # no primitives in gen.tsv, otherwise won't work
            sentence = sentence[:-2].lower()  # exclude " ." and lowercase
            if seen_samples > 0:
                outfile.write(", ")
            outfile.write(f'\"{sentence}\": \"{gentype}\"')
            seen_samples += 1
        outfile.write("}")
    print(f"example2type: seen {seen_samples} samples")
    return


# todo: implement (so far can copy caus(unac from train, for entity add 'monastery')
# I know that ideally this should be aligned to what the logical form is parsed into
def create_pred_ent_files(train_file: str, output_folder: str):
    caus_predicate_file = path.join(output_folder, 'caus_predicate')
    unac_predicate_file = path.join(output_folder, 'unac_predicate')
    # 1 token per line, lowercased, all nouns (proper, common)
    entity_file = path.join(output_folder, 'entity')

    entities = set()
    predicates = set()
    for sentence, logical_form, gentype in get_sample_generator(train_file):
        tokens = logical_form.split()
        tsize = len(tokens)
        # nouns: a 1-size predicate names or with uppercase letter/as argument
        prev = None
        for i, token in enumerate(tokens):
            if token.istitle():  # proper noun identifiable by first letter uppercase
                entities.add(token.lower())
            elif tsize == 1:  # primitive proper noun
                assert(gentype == "primitive")
                entities.add(token)
            elif i == 3 and tsize == 7 and gentype == "primitive":  # primitive common noun:
                # ['LAMBDA', 'a', '.', 'noun', '(', 'a', ')']
                entities.add(token.lower())
            elif prev in {"*", ";", "AND"} and i != tsize-1 and tokens[i+1] == "(":
                # ... AND dolphin ( ...  , ... ; dolphin ( ...
                entities.add(token.lower())
            prev = token
        # verbs: 2-size predicate names
        # todo what's the difference between caus and unac?
        for i, token in enumerate(tokens):
            if token in {"agent", "theme", "recipient", "ccomp", "xcomp"}:
                assert(i >= 2)
                predicates.add(tokens[i-2].lower())
        pass

    with open(entity_file, "w", encoding="utf-8") as outf:
        for token in entities:
            outf.write(token+"\n")
    print(f"entity: {len(entities)} entities found")
    with open(caus_predicate_file, "w", encoding="utf-8") as outf:
        for token in predicates:
            outf.write(token+"\n")
    print(f"predicate: {len(predicates)} predicates found")
    return


def main(argv):
    """call with no arguments for usage info"""
    if len(argv) != 4:
        print("usage: do_preprocessing.py OUTFOLDER COGSDATAFOLDER TRAINFILENAME")
        print("  -> do the preprocessing (generate files of preprocess folder)")
        print("  OUTFOLDER  where to store output files (will overwrite existing ones with same name without warning!)")
        print("  COGSDATAFOLDER  expected to contain 'gen.tsv', 'dev.tsv' and a train tsv file")
        print("  TRAINFILENAME  either 'train.tsv' or 'train_100.tsv' probably")
        sys.exit(1)
    outfolder = argv[1]
    infolder = argv[2]

    # Check if folders exists
    for folder in [outfolder, infolder]:
        if not path.isdir(folder):
            print(f"ERROR: Folder doesn't exits. Exit.  Folder: {folder} ")
            sys.exit(2)
    train_file = argv[3]
    train_file = path.join(infolder, train_file)
    if not path.isfile(train_file):
        print(f"ERROR: Train file wasn't found in input folder:  {train_file}")
        sys.exit(2)


    create_example2type_file(gen_file=path.join(infolder, 'gen.tsv'),
                             output_file=path.join(outfolder, 'example2type'))

    create_enct2dect(train_file=train_file,
                     # dev_file=path.join(infolder, 'dev.tsv'),
                     output_file=path.join(outfolder, 'enct2dect'))

    # encode_tokens.txt  all tokens from input incl 'the', 'beside', '.'
    create_encode_tokens(train_file=train_file,
                         output_file=path.join(outfolder,'encode_tokens.txt'))

    create_pred_ent_files(train_file=train_file, output_folder=outfolder)
    print("Done!")
    return


if __name__ == "__main__":
    main(sys.argv)
