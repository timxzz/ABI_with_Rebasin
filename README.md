# A Compact Representation for Bayesian Neural Networks By Removing Permutation Symmetry

<div id="top"></div>

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](http://arxiv.org/abs/2401.00611)

  <span><a href="https://timx.me" target="_blank">Tim&nbsp;Z.&nbsp;Xiao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://wyliu.com" target="_blank">Weiyang&nbsp;Liu</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://robamler.github.io" target="_blank">Robert&nbsp;Bamler</a>
  </span>
  <br/>
  


## About The Project
This is the official GitHub repository for our NeurIPS 2023 UniReps Workshop paper [A Compact Representation for Bayesian Neural Networks By Removing Permutation Symmetry](http://arxiv.org/abs/2401.00611).

### Abstract
> Bayesian neural networks (BNNs) are a principled approach to modeling predic- tive uncertainties in deep learning, which are important in safety-critical applica- tions. Since exact Bayesian inference over the weights in a BNN is intractable, various approximate inference methods exist, among which sampling methods such as Hamiltonian Monte Carlo (HMC) are often considered the gold standard. While HMC provides high-quality samples, it lacks interpretable summary statis- tics because its sample mean and variance is meaningless in neural networks due to permutation symmetry. In this paper, we first show that the role of permutations can be meaningfully quantified by a number of transpositions metric. We then show that the recently proposed rebasin method allows us to summarize HMC samples into a compact representation that provides a meaningful explicit uncer- tainty estimate for each weight in a neural network, thus unifying sampling meth- ods with variational inference. We show that this compact representation allows us to compare trained BNNs directly in weight space across sampling methods and variational inference, and to efficiently prune neural networks trained without explicit Bayesian frameworks by exploiting uncertainty estimates from HMC.

## Environment: 

Python 3.9.12

Other dependencies are in `requirements.txt`


## Training

#### HMC

```bash
python MNIST_HMC.py \
--dataset MNIST \
--config_name test3.2 \
--run_name <RUN_NAME> \
--run_batch_name <RUN_BATCH_NAME>
```

#### Ensemble

```bash
python MNIST_HMC.py
```

#### VI

```bash
python MNIST_VI.py
```

## Evaluations 

#### Loss Barriers

```bash
python MNIST_net_matching.py \
--seed 12 \
--dataset MNIST \
--optimizer SGD \
--num_epochs 50 \
--by_activation \
--run_name <RUN_NAME> \
--run_batch_name <RUN_BATCH_NAME>
```

#### Number of Transpositions (NoTs)

```bash
python MNIST_number_of_transpositions.py \
--seed 30 \
--dataset MNIST \
--num_epochs 50 \
--optimizer SGD \
--num_perm_exps 50 \
--by_activation  \
--run_name <RUN_NAME> \
--run_batch_name <RUN_BATCH_NAME>
```

#### Test Performance

```bash
python MNIST_eval.py
```

#### Test Performance vs. Pruning

```bash
python MNIST_eval_pruning.py
```

## Citation:
Following is the Bibtex if you would like to cite our paper :

```bibtex
@article{xiao2023compact,
  title = {A Compact Representation for Bayesian Neural Networks By Removing Permutation Symmetry},
  author = {Xiao, Tim Z. and Liu, Weiyang and Bamler, Robert},
  journal = {NeurIPS 2023 Workshop on Unifying Representations in Neural Models},
  year = {2023},
}
```