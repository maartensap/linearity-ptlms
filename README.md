# Linearity of sentences with pretrained LMs

## Setup and requirements
- First, clone the repo: `git clone git@github.com:maartensap/linearity-ptlms.git`.
- Set up a virtualenv: `virtualenv --python python3.6 venv-linearity`.
- Launch the virtualenv: `venv-linearity/bin/activate`.
- Install the required packages:
  - `pip install transformers==3.0.2 nltk==3.5`
  - install pytorch v1.5.1 (varies based on your computing setup): https://pytorch.org/
  - alternatively, use `pip install -r requirements.txt` (could possible break with different torch/cuda versions)


## Step 1: Extracting probabilities of sentences
There are two ways to run the probability extraction script: for only one story, and for multiple stories in bulk.
Note, here I use the term "stories" to denote a document/piece of text.

This script will extract the probabilities of sentences with various amounts of history (i.e., preceding sentences) and some initial context (e.g., gist of a story, summary, main event).
History sizes are indicated with `--history_sizes 0 1 2 ...`, and correspond to how many preceding sentences to condition the probability on. For example: h=2, means your current sentence s_i will be conditioned on s_i-2 and s_i-1. A size of -1 will include the entire history. See the paper for further details.

For all options, run `extractLinearity.py --help`.

### Single story
Option: `--input_sentence_file`
If you only have one story to analyze, this is the option for you. You can input your file in several formats:
- `.csv`: each sentence is a line (column name: `--sentence_column`), with a potential story identifier (`--story_id_column`; should be the same for all sentences)
- `.txt`: each sentence is a line

Example usage: `python extractLinearity.py --input_sentence_file oneStory.csv --sentence_id_column sent_id --history_sizes -1 0 1 2 3 --output_sentence_file oneStory.gptPplx.-1.0.1.2.3.csv`

The output file (`--output_sentence_file`) will contain one line per sentence, with the following columns:
- `story_id`: story identifier, same on every line since this is just one story
- `sent_id`: index of the story (either from the input file, or created automatically)
- `sents`: sentence text
- For each of the history sizes (e.g., h=-1, 0):
  - `text_xents_hist-1`: log probability of sentence in context normalized by sentence length (equivalent to the average log probability of each word in context).
  - `text_probs_hist-1`: average probability of each word in the sentence (note, this isn't simply the exponentiated version of the above).
  - `perTextTokenLogPs_hist-1`: log probabilities for each word/token in the sentence in context. Tokens are determined by the pretrained LM's tokenizer, and could be BPEs.


### Multiple stories
Option: `--input_story_file`
If you have multiple stories to analyze (or you only have one story but not split into sentences), this is the option. Similar input formats supported:
- `.csv`: each story is a line (column name: `--story_column`), with a potential story identifier (`--story_id_column`)
- `.txt`: each story is a line

Sentences will be automatically split (using NLTK's sentence tokenizer with tweaks).

Example usage: `python extractLinearity.py --input_story_file manyStories.csv --history_sizes -1 0 --output_sentence_file manyStories.gpt2Pplx.-1.0.sentenceLevel.csv --output_story_file manyStories.gpt2Pplx.-1.0.pkl --language_model gpt2`

The output file (`--output_story_file`) will contain the same columns as above, plus the following:
- the full story (column name: `--story_column` or `story_id`)
- For each history size, the average (e.g., `xent_avg_hist-1`) and standard deviation (e.g., `xent_std_hist-1`) of the log probability of the sentences in the story (`text_xents_hist-1`).

## Running the analysis script
_coming soon_

## Reference
Please cite [our paper](https://homes.cs.washington.edu/~msap/pdfs/sap2020recollectionImagination.pdf):

Maarten Sap, Eric Horvitz, Yejin Choi, Noah A Smith & James W Pennebaker (2020).
**Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models**. ACL

```
@inproceedings{sap2020recollectionImagination,
  title={Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models},
  author={Sap, Maarten and Horvitz, Eric and Choi, Yejin and Smith, Noah A and Pennebaker, James W},
  year={2020},
  booktitle={ACL},
}
```
