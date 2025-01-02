# <i>GEM</i>: Graph Attention Encoder for Multi-task Depression Severity Detection in Multi-party Conversations
This repository contains the source code of the initial version of <i>GEM</i>, a graph attention network (GAT)-based multitask learning (MTL) pipeline for depression detection and severity classification using multi-party conversation (MPC) data. By modeling utterances as graph nodes and leveraging hierarchical root-sub and sub-sub relationships, the shared GAT layers captured critical depression cues and speaker interactions. Task-specific layers refined the shared representations for both binary depression detection and severity classification, facilitating knowledge transfer across tasks.

## Datasets
- Download [Reddit eRisk 18 T2 2018](https://link.springer.com/chapter/10.1007/978-3-319-98932-7_30) used in the original paper and move to path: ./mpc_data 
- Download [Reddit eRisk 22 T2 2022](https://books.google.co.jp/books?hl=en&lr=&id=LzaFEAAAQBAJ&oi=fnd&pg=PA231&dq=Overview+of+eRisk+2022:+Early+Risk+Prediction+on+the+Internet&ots=LnO4GFgjt7&sig=lgSXnAWqqgjiPUp-jYV3HKIv4z8&redir_esc=y#v=onepage&q=Overview%20of%20eRisk%202022%3A%20Early%20Risk%20Prediction%20on%20the%20Internet&f=false) used in the original paper and move to path: ./mpc_data
- Download [Twitter Depression 2022](https://www.nature.com/articles/s41599-022-01313-2) used in the original paper and move to path: ./mpc_data
- Download [Severity levels corpus](https://dl.acm.org/doi/10.1007/978-3-031-42141-9_1) used in the original paper and move to path: ./severity_data
- Download [DepSeverity corpus](https://dl.acm.org/doi/10.1145/3485447.3512128) used in the original paper and move to path: ./severity_data
- Download [DEPTWEET corpus](https://www.sciencedirect.com/science/article/abs/pii/S0747563222003235) used in the original paper and move to path: ./severity_data

Unzip the corpora and run the following commands to categorize the MPC data into three groups based on the session length (total number of utterances): <i>Len-5</i>, <i>Len-10</i>, and <i>Len-15</i>. <br>
  ```
  cd data/
  python data_preprocess.py
  ```

## Pre-training Phase - Self-supervised Tasks Formation
The training phase of the proposed GAT-based MTL pipeline determines depression detection head (DDH) and severity classification head (SCH). DDH is designed to distinguish depressed and non-depressed utterances along with the depressed users while SCH categorizes the depressed utterances into severity levels. Run the following commands for DDH and SCH.
  ```
  python trainer.py
  ```

## Downstream Tasks Formation
Three downstream tasks, Depressed Utterance Detection (DUD), Depressed Interlocutor Recognition (DIR), and Depression Severity Classification (DSC), are formed based on the outputs of DDH and SCH.

## LLM Prompting
To evaluate DUD and DSC, MentalBERT and DisorBERT were used as 100M-300M-parameter LLMs and MentaLLaMA-7B and Mental-Alpaca were used as open-source 7B-parameter LLMs while GPT-4 was adopted as a closed-source 1.76T-parameter LLM. Run the following commands to facilitate the LLM prompting logic.
  ```
  cd llm_prompting/
  python prompt.py
  ```
To evaluate the proposed GAT-based MTL pipeline with respect to sequential model baselines, graph-based baselines, and LLMs, run the following commands.
  ```
  cd llm_prompting/
  python few_shot.py
  ```
