#/usr/bin/env python3

import numpy as np
import pandas as pd
import sys, os
import json
import argparse
from tqdm import tqdm
from itertools import combinations
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
  from IPython import embed
except:
  pass

MAX_SEQ_LEN = 512
DEFAULT_STORY_ID_COL = "story_id"
DEFAULT_STORY_ID_VALUE = "doc0"
FEAT_COLS = ['text_xents', 'text_probs', 'perTextTokenLogPs',
             'xent_avg', 'xent_std']
SENT_FEAT_COLS = ['text_xents', 'text_probs', 'perTextTokenLogPs']

np.random.seed(seed=56)

# logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
#                     level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__file__)

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
    input_ids = torch.tensor([tokens]).to(model.device)
    out = model(input_ids)
    logits = out[0]
    
    assert logits.shape[:2] == input_ids.shape
    assert logits.shape[-1] == model.config.vocab_size
         

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

  if context_col == "firstSent":
    log.info("Using first sentence as context")
    sents_backup = df[sent_col].copy()
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
  data["contHist"] = data["cont"] + "\n\n" + data["hist"].apply(
    lambda x: " ".join(x) + (" " if len(x) > 0 else ""))
  data["contHist"] = data["contHist"].str.lstrip()
  
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
  
  # embed();exit()
  
  data["text_prob"] = data["perTextTokenLogP"].apply(
    lambda x: np.exp(x)).apply(np.mean)
  # data[["perTextTokenLogP","text_xent"]] = data[["toks","text_ix"]].progress_apply(
  #   lambda x: _computePplxInContext(*x,model,tok),axis=1)

  # re-format to make short again
  def reMergeXents(c):
    return pd.Series({
      "text_xents": c["text_xent"].tolist(),
      "text_probs": c["text_prob"].tolist(),
      "perTextTokenLogPs": c["perTextTokenLogP"].tolist(),
      "xent_avg": c["text_xent"].mean(),
      "xent_std": c["text_xent"].std(),
    })
  
  data.index.names = ["doc_ix","sent_ix"]
  # data.reset_index(level=1,inplace=True)
  feats = data.groupby(level=0).apply(reMergeXents)
  feats = feats.reindex(df.index)

  if context_col == "firstSent":
    log.info("Resetting the sentences")
    df[sent_col] = sents_backup
    
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
    elif args.input_story_file.endswith(".txt"):
      lines = [l.strip() for l in open(args.input_story_file)]
      s = pd.Series(data=lines)
      s.name = args.story_column
      df = s.to_frame()
    else:
      raise ValueError("Unknown data format "+args.input_story_file)
    
    log.info(f"Found {len(df)} stories.")
    
  elif args.input_sentence_file:
    if args.input_sentence_file.endswith(".csv"):
      sent_df = pd.read_csv(args.input_sentence_file)
    elif args.input_sentence_file.endswith(".pkl"):
      sent_df = pd.read_pickle(args.input_story_file)
    elif args.input_sentence_file.endswith(".txt"):
      lines = [l.strip() for l in open(args.input_sentence_file)]
      s = pd.Series(data=lines)
      s.name = args.sentence_column
      sent_df = s.to_frame()
      
    log.info(f"Found {len(sent_df)} sentences.")
    if not args.story_id_column in sent_df:
      sent_df[args.story_id_column] = DEFAULT_STORY_ID_VALUE

    # turn into a one story Dataframe
    aggDict = {args.sentence_column:list}
    if args.context_column:
      aggDict[args.context_column] = "first"
    df = sent_df.groupby(args.story_id_column,as_index=False).agg(aggDict)
    df[args.story_column] = df[args.sentence_column].apply(" ".join)

  else:
    raise ValueError("Please provide one of --input_story_file or --input_sentence_file")
  
  if args.debug:
    df = df.sample(args.debug)
    log.info(f"[DEBUG] Sampling {args.debug} datapoints.")
    
  if not args.context_column:
    args.context_column = "summary"
    df[args.context_column] = ""
    
  return df

def saveOutput(df,fn,inst="stories"):
  if fn.endswith(".pkl"):
    df.to_pickle(fn)
  elif fn.endswith(".csv"):
    listsOrDicts = df.apply(lambda c: c.apply(
      lambda x: isinstance(x,list) or  isinstance(x,dict)).all())
    df = df.copy()
    for c in df.columns[listsOrDicts]:
      df[c] = df[c].apply(json.dumps)
    df.to_csv(fn,index=False)
  else:
    log.warn("Unrecognized file extension, will save as pickle")
    fn = fn+".pkl"
    df.to_pickle(fn)
    
  log.info(f"Exporting features for {len(df)} {inst} to '{fn}'.")

def splitIntoSentences(df,tok,story_col,sent_col):
  if sent_col not in df:
    tqdm.pandas(desc="Tokenizing into sentences",ascii=True)
    df[sent_col] = df[story_col].progress_apply(
      lambda t: betterSentTokenization(t,tok.encode))
  
  return df

def meltFeaturesPerSentence(df,args,story_level_cols=[]):
  sent_feat_cols = [c for c in df.columns if c.split("_hist")[0] in SENT_FEAT_COLS]
  if args.context_column == "firstSent":
    log.warn("Since the context is the 'firstSent', the first sentence will have NaN values")
    # adding nan's to the feature columns
    for c in sent_feat_cols:
      if "perTextToken" in c:
        df[c] = df[c].apply(lambda x: [[np.nan]] + x)
      else:
        df[c] = df[c].apply(lambda x: [np.nan] + x)
      
  cols = [args.sentence_column]+sent_feat_cols

  if not args.story_id_column:
    args.story_id_column = DEFAULT_STORY_ID_COL
    df[args.story_id_column] = df.index
    
  
  data = {(r[args.story_id_column],sIx): {c: r[c][sIx] for c in cols}
          for ix, r in df.iterrows()
          for sIx,_ in enumerate(r[args.sentence_column])}
  
  df_long = pd.DataFrame.from_dict(data,orient="index")
  df_long.index.names = [args.story_id_column, args.sentence_id_column]
  df_long = df_long.reset_index()
  if story_level_cols:
    story_level_cols = pd.Index(story_level_cols).difference(df_long.columns).tolist() + [args.story_id_column]
    df_long = df_long.merge(df[story_level_cols],on=args.story_id_column)

  return df_long

def computeDiffsPerSentence(feats,hist_sizes,col="text_xents_hist"):
  # relevant column to subtract: text_xents_histH
  hist_sizes_str = sorted(["Full" if h == -1 else str(h) for h in hist_sizes])

  for b,c in combinations(hist_sizes_str,r=2):
    b_ = b.replace("Full","-1")
    c_ = c.replace("Full","-1")
    feats[col+b+"Minus"+c] = feats[[col+b_,col+c_]].apply(
      lambda x: np.array(x[0])-np.array(x[1]),axis=1)
  return feats

def main(args):
  print(args)
  df = loadInput(args)

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.language_model)

  # Parse into sentences
  df = splitIntoSentences(df,tokenizer,args.story_column,args.sentence_column)

  # Loading model
  model = AutoModelForCausalLM.from_pretrained(args.language_model)
  model = model.to(args.device)
  
  try:
    max_seq_len = model.config.max_position_embeddings
  except:
    max_seq_len = MAX_SEQ_LEN
  
  feat_list = {}
  for h in tqdm(args.history_sizes,ascii=True,desc="History sizes"):
    print(file=sys.stderr); sys.stderr.flush()
    feats = computePplxInContext(
      df,tokenizer,model, story_col=args.story_column,
      context_col=args.context_column,
      sent_col=args.sentence_column,hist_size=h,
      max_seq_len=max_seq_len)
    feats = feats.rename(columns={c: c+f"_hist{h}" for c in feats})
    feat_list[h] = feats
    print(file=sys.stderr); sys.stderr.flush()
    
  feats = pd.concat(feat_list.values(),axis=1)
  if args.extract_diffs:
    feats = computeDiffsPerSentence(feats,args.history_sizes)
  
  df_out = pd.concat([df,feats],axis=1,sort=False)
  
  if args.output_story_file:
    saveOutput(df_out,args.output_story_file,"stories")
    
  if args.output_sentence_file:
    df_long = meltFeaturesPerSentence(df_out,args,story_level_cols=df.columns.tolist())
    saveOutput(df_long,args.output_sentence_file,"sentences")
    
  log.info("Done")


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_story_file",help="CSV file with one story per line. "
                      "Use this to analyze multiple stories at once.")
  parser.add_argument("--input_sentence_file",help="CSV file with one sentence per line. "
                      "Use this for one story only.")
  
  parser.add_argument("--output_sentence_file",
                      help="Output file with the linearity of each sentence per line.")
  parser.add_argument("--output_story_file",
                      help="Output file with linearity scores aggregated per story.")
  
  parser.add_argument("--story_column",default="story",
                      help="Column for which to compute the linearity scores")
  
  parser.add_argument("--story_id_column",default="story_id",
                      help="Story identifier (optional, defaults to 'story_id'; "
                      "will assign automatically)")
  parser.add_argument("--sentence_id_column",default="sentence_id",
                      help="Sentence identifier (optional, defaults to 'sentence_id'; "
                      "will assign automatically)")
  
  parser.add_argument("--context_column", default=None,
                      help="Column that contains the context or 'main event'. "
                      "Optional; if ommitted, will use the empty string.")  
  parser.add_argument("--sentence_column",default="sents",
                      help="If text already split up into sentences. "
                      "This column should be a json list of words/tokens.")

  parser.add_argument("--history_sizes",nargs="+",type=int,default=[0, -1])

  parser.add_argument("--debug",type=int,default=0)

  parser.add_argument("--language_model",default="openai-gpt",
                      help="Which large LM to use to compute perplexity. Options include: "
                      "openai-gpt, gpt2, distilgpt2, transfo-xl, reformer.")

  parser.add_argument("--extract_diffs",action="store_true")
  parser.add_argument("--device",default="cpu")
  
  args = parser.parse_args()
  
  if not args.output_story_file and not args.output_sentence_file:
    parser.print_usage()
    print()
    raise ValueError("Please provide an output file: --output_story_file or --output_sentence_file")

    
  # if args.output_file.endswith(".csv"):
  #   raise ValueError("Please use a .pkl extension for the outputfile")
  
  log.info(args)
  main(args)
