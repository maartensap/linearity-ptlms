#/usr/bin/env python3

import numpy as np
import pandas as pd
import sys, os
import json
import argparse
from tqdm import tqdm

from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
  from IPython import embed
except:
  pass

MAX_SEQ_LEN = 512

np.random.seed(seed=56)

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

def betterSentTokenization(text,wTokenize=word_tokenize,m=2):
  """Sentence tokenization that makes sure there aren't any sentences
  with only one token.
  """
  sents = sent_tokenize(text)
  toks = [wTokenize(s) for s in sents]
  sings = [i for i,s in enumerate(toks) if len(s) <= m]
  if sings:
    for i in reversed(sings):
      sing = sents.pop(i)
      if i !=0:
        sents[i-1] += " "+sing
      else:
        try:
          sents[0] = sing + " " + sents[0]
        except:
          sents = [sing]
  return sents


def _computePplxInContext(tokens,index,model,tok,max_seq_len=MAX_SEQ_LEN):
  # Note: tokens[index:] is the target sentence
  if len(tokens) > max_seq_len:
    log.warn(f"Sequence is too long for this model ({len(tokens)} > {max_seq_len}); truncating the history")
    dec = len(tokens)-max_seq_len
    tokens = tokens[dec:]
    index = max(0,index-dec)
    # assert index > 0
    
  with torch.no_grad():
    input_ids = torch.tensor([tokens])
    # try:
    out = model(input_ids)#,labels=input_ids)
    # except RuntimeError as e:
    #   embed();exit()
    
    if isinstance(model,OpenAIGPTLMHeadModel) or isinstance(model,GPT2LMHeadModel):
      logits = out[0]
    elif isinstance(model,TransfoXLLMHeadModel):
      logits = out[0]
      
    assert logits.shape[:2] == input_ids.shape
    # only taking the probs after index
    logits = logits[0,max(index-1,0):-1,:]
    sm = logits.log_softmax(dim=-1)
    
    if index == 0:
      # no context, we lose information on one token
      tokens = tokens[1:]
      
    assert sm.shape[0] == len(tokens[index:])
    ps = [lps[t].item() for lps,t in zip(sm,tokens[index:])]
    
  return ps


def computePplxInContext(df,tok,model,story_col="story",context_col="summary",
                         sent_col="sents",hist_size=-1,chunk_size=16,
                         max_seq_len=MAX_SEQ_LEN):
  """
  Computes the xent of tokens in a sentence conditioned on history and context:
    p(s_i|context, s_i-k,...,s_i-1)
  context: text to be prepended to every sentence.
  hist_size: how far back does history go? If -1, it goes back all the way.
  """
  if sent_col not in df:
    tqdm.pandas(desc="Tokenizing into sentences",ascii=True)
    df[sent_col] = df[story_col].progress_apply(
      lambda t: betterSentTokenization(t,tok.encode))
  else:
    print("Data already split into sentences.")
  print(df[sent_col].apply(len).describe())

  if context_col == "firstSent":
    df[context_col] = df[sent_col].apply(lambda x: x[0])
    df[sent_col] = df[sent_col].apply(lambda x: x[1:])
  
  # Todo: make sure index is unique
  assert len({i for i in df.index}) == len(df), "df index isn't unique"
  dataD = {}
  for ix, ss, cont in df[[sent_col,context_col]].itertuples():
    for i,s in enumerate(ss):
      hist_ix = 0 if hist_size==-1 else max(0,i-hist_size)
      dataD[(ix,i)] = {
        "cont": cont,
        "hist": ss[hist_ix:i],
        "text": s
      }
  data = pd.DataFrame(dataD).T
  data["contHist"] = data["cont"] + "\n\n" + data["hist"].apply(" ".join) + " "
  
  tqdm.pandas(ascii=True,desc="Tokenizing context+history")
  data["contHist_toks"] = data["contHist"].progress_apply(tok.encode)
  
  tqdm.pandas(ascii=True,desc="Tokenizing sentence text")
  data["text_toks"] = data["text"].progress_apply(tok.encode)
  
  data["toks"] = data["contHist_toks"] + data["text_toks"]
  data["text_ix"] = data["contHist_toks"].apply(len)
  # data["text_xent"] = np.nan
  # data["perTextTokenLogP"] = data["text_xent"].apply(lambda x: [])
  
  tqdm.pandas(ascii=True,desc="Getting probabilities")
  data["perTextTokenLogP"] = data[["toks","text_ix"]].progress_apply(
    lambda x: _computePplxInContext(*x,model,tok,max_seq_len=max_seq_len),axis=1)
  
  data["text_xent"] = -data["perTextTokenLogP"].apply(np.mean)

  data["text_prob"] = data["perTextTokenLogP"].apply(
    lambda x: np.exp(x)).apply(np.mean)
  # data[["perTextTokenLogP","text_xent"]] = data[["toks","text_ix"]].progress_apply(
  #   lambda x: _computePplxInContext(*x,model,tok),axis=1)

  # re-format to make short again
  def reMergeXents(c):
    return pd.Series({
      "text_xents": c["text_xent"].tolist(),
      "text_probss": c["text_prob"].tolist(),
      "perTextTokenLogPs": c["perTextTokenLogP"].tolist(),
      "xent_avg": c["text_xent"].mean(),
      "xent_std": c["text_xent"].std(),
    })
    
  data.index.names = ["doc_ix","sent_ix"]
  # data.reset_index(level=1,inplace=True)
  feats = data.groupby(level=0).apply(reMergeXents)
  feats = feats.reindex(df.index)
  
  assert len(feats) == len(df)
  return feats

def loadInput(args):
  if args.input_story_file and args.input_sentence_file:
    raise ValueError("Please provide only one of --input_story_file or --input_sentence_file")

  elif args.input_story_file:
    log.info(f"Reading {args.input_story_file}")
    if args.input_story_file.endswith(".csv"):
      df = pd.read_csv(args.input_story_file)
    elif args.input_story_file.endswith(".pkl"):
      df = pd.read_pickle(args.input_story_file)
    elif args.input_sentence_file.endswith(".txt"):
      lines = [l.strip() for l in open(args.input_story_file)]
      s = pd.Series(data=lines)
      s.name = args.story_column
      df = s.to_frame()
    else:
      raise ValueError("Unknown data format "+args.input_story_file)
    
    log.info(f"Found {len(df)} stories.")
    
  elif args.input_sentence_file:
    if args.input_sentence_file.endswith(".csv"):
      df = pd.read_csv(args.input_sentence_file)
    elif args.input_sentence_file.endswith(".pkl"):
      df = pd.read_pickle(args.input_story_file)
    elif args.input_sentence_file.endswith(".txt"):
      lines = [l.strip() for l in open(args.input_sentence_file)]
      s = pd.Series(data=lines)
      s.name = args.sentence_column
      df = s.to_frame()
    log.info(f"Found {len(df)} sentences.")
  else:
    raise ValueError("Please provide one of --input_story_file or --input_sentence_file")

  if args.debug:
    df = df.sample(args.debug)
    log.info(f"[DEBUG] Sampling {args.debug} datapoints.")
    
  return df

def splitIntoSentences(df,tok,story_col,sent_col):
  if sent_col not in df:
    tqdm.pandas(desc="Tokenizing into sentences",ascii=True)
    df[sent_col] = df[story_col].progress_apply(
      lambda t: betterSentTokenization(t,tok.encode))
  else:
    print("Data already split into sentences.")
    
  print(df[sent_col].apply(len).describe())
  
  return df

def main(args):
  df = loadInput(args)

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.language_model)

  # Parse into sentences
  df = splitIntoSentences(df,tokenizer,args.story_column,args.sentence_column)

  # Loading model
  model = AutoModelForCausalLM.from_pretrained(args.language_model)

  embed();exit()
  
  if not args.context_col:
    args.context_col = "summary"
    df[args.context_col] = ""
  
  feats = computePplxInContext(
    df,tok,model,story_col=args.column,context_col=args.context_col,
    sent_col=args.sent_col,hist_size=args.hist_size,
    max_seq_len=model.config.max_position_embeddings)
  
  df = pd.concat([df,feats],axis=1,sort=False)
  df.to_pickle(args.output_file)
  log.info(f"Exporting feats to {args.output_file}") 


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_story_file",help="CSV file with one story per line. "
                      "Use this to analyze multiple stories at once.")
  parser.add_argument("--input_sentence_file",help="CSV file with one sentence per line. "
                      "Use this for one story only.")
  
  parser.add_argument("--output_sentence_file",
                      help="Output file with the linearity of each sentence per line.")
  parser.add_argument("--output_aggregate_file",
                      help="Output file with linearity scores aggregated per story.")
  
  parser.add_argument("--context_column", default=None,
                      help="Column that contains the context or 'main event'. "
                      "Optional; if ommitted, will use the empty string.")
  
  parser.add_argument("--sentence_column",default="sents",
                      help="If text already split up into sentences. "
                      "This column should be a json list of words/tokens.")
  parser.add_argument("--history_sizes",nargs="+",type=int,default=[0, -1])
  parser.add_argument("--debug",type=int,default=0)
  parser.add_argument("--story_column",default="story",
                      help="Column for which to compute the linearity scores")
  parser.add_argument("--language_model",default="openai-gpt",
                      help="Which large LM to use to compute perplexity. Options include: "
                      "openai-gpt, gpt2, distilgpt2, transfo-xl, reformer.")
  
  args = parser.parse_args()
  
  # if not args.input_file or not args.output_file:
  #   parser.print_help()
  #   exit(2)
    
  # if args.output_file.endswith(".csv"):
  #   raise ValueError("Please use a .pkl extension for the outputfile")
  
  print(args)
  main(args)
