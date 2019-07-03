# Null Space Gradient Descent (NSGD) and Document Space Projected Dueling Bandit Gradient Descent (DBGD-DSP)
This repository contains the code used to produce the experimental results found in "Efficient Exploration of Gradient Space for Online Learning to Rank" and "Variance Reduction in Gradient Exploration for Online Learning to Rank" published at SIGIR 2018 and SIGIR 2019, respectively. It was forked from Harrie Oosterhuis's repository for "Differentiable Unbiased Online Learning to Rank" published at CIKM 2018, at https://github.com/HarrieO/OnlineLearningToRank.

NSGD Algorithm
-------
This algorithm was developed with the intent of increasing the efficiency of exploration of the gradient space for online learning to rank. It does this in a series of 3 steps. First, the null space of previously poorly performing directions is computed, and new directions are sampled from within this null space (this helps to avoid exploring less promising directions repeatedly). Second, a candidate preselection process is done wherein the sampled directions which are most differentiable by the current query's documents are chosen for evaluation. Thirdly, in the event of a tie, a tie-breaking mechanism uses historically difficult queries to reevaluate the candidates and choose a winner.

DBGD-DSP Algorithm
-------
The aim of this algorithm is to act as a wrapper around other DBGD-style algorithms to reduce their ranker variance and improve overall performance in online learning to rank. DBGD-DSP works by modifying the winning ranker after the interleaved test. In particular, it projects the winning gradient into the space spanned by the query-document feature vectors associated with the given query. This reduces the variance in gradient exploration by removing the component of the winning gradient that was orthogonal to the document space, which does not contribute to the loss function and true gradient estimation.

Usage
-------
To run the code to generate experimental results like those found in our papers, you will need to run a command in the following format, using Python 2 (SIGIR2018.py, SIGIR2019.py, and SIGIR2019_NSGD.py are all run similarly):

```
python scripts/SIGIR2019.py [-h] [--n_runs N_RUNS] [--n_impr N_IMPRESSIONS] [--vali]
                    [--vali_in_train] --data_sets DATA_SETS [DATA_SETS ...]
                    [--output_folder OUTPUT_FOLDER] [--log_folder LOG_FOLDER]
                    [--average_folder AVERAGE_FOLDER] [--small_dataset]
                    --click_models CLICK_MODELS [CLICK_MODELS ...]
                    [--print_freq PRINT_FREQ] [--print_logscale]
                    [--print_output] [--max_folds MAX_FOLDS]
                    [--n_proc N_PROCESSING] [--no_run_details]
                    [--n_results N_RESULTS] [--skip_read_bin_data]
                    [--skip_store_bin_data] [--train_only] [--all_train]
                    [--nonrel_test]
```                 

In the command above, parameters within square brackets are optional. In our papers, we used datasets such as MQ2007 and MQ2008 from LETOR 4.0 datasets, the Yahoo! learning to rank challenge dataset, and MSLR-WEB10K dataset. The possible click models are described in our papers: inf = informational, nav = navigational, and per = perfect. 

Citation
--------

If you use this code to produce results for your scientific publication, please refer to our SIGIR 2019 paper and/or SIGIR 2018 paper:

```
@inproceedings{wang2019variance,
  title={Variance Reduction in Gradient Exploration for Online Learning to Rank},
  author={Wang, Huazheng and Kim, Sonwoo and McCord-Snook, Eric and Wu, Qingyun and Wang, Hongning},
  booktitle={The 42nd International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  year={2019},
  organization={ACM}
}

@inproceedings{wang2018efficient,
  title={Efficient exploration of gradient space for online learning to rank},
  author={Wang, Huazheng and Langley, Ramsey and Kim, Sonwoo and McCord-Snook, Eric and Wang, Hongning},
  booktitle={The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  year={2018},
  organization={ACM}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.
