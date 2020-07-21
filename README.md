# Linearity of sentences with pretrained LMs

## Setup and requirements
- First, clone the repo: `git clone git@github.com:maartensap/linearity-ptlms.git`.
- Set up a virtualenv: `virtualenv --python python3.6 venv-linearity`.
- Launch the virtualenv: `venv-linearity/bin/activate`.
- Install the required packages:
  - `pip install transformers==3.0.2 nltk==3.5`
  - install pytorch v1.5.1 (varies based on your computing setup): https://pytorch.org/


## Notes
Script should be able to extract a bunch of linearity scores and export them to a file (csv? or pkl?)
And compare against categories?
math test: $$h_i$$

## Reference
Maarten Sap, Eric Horvitz, Yejin Choi, Noah A Smith & James W Pennebaker (2020).
**Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models**. ACL

Bibtex:
```
@inproceedings{sap2020recollectionImagination,
  title={Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models},
  author={Sap, Maarten and Horvitz, Eric and Choi, Yejin and Smith, Noah A and Pennebaker, James W},
  year={2020},
  booktitle={ACL},
}
```
