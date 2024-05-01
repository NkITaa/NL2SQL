# Bachelors Thesis: A Comparison of Chunking and Prompting Strategies for Large Text Data Processing in Language Models

## Abstract
This thesis investigates the effectiveness of chunking and prompting techniques for handling large text datasets in language models, specifically utilizing the Blar AI Framework. Datasets such as Spider and BIRD are employed in this study, and models are trained on prepared data.

## Table of Contents
* [Data Preparation](#Data-Preparation)
* [Model Training](#Model-Training)
* [Inference](#Inference)
* [Setup](#Setup)
* [Accessing Trained Models](#Accessing-Trained-Models)

## Data Preparation
The [Data_Preparing directory](https://github.com/NkITaa/NL2SQL/tree/main/Data_Preparing) contains a script for transforming the raw [Spider](https://yale-lily.github.io//spider) and [BIRD](https://bird-bench.github.io/) datasets into [Prepared_Data CSVs](https://github.com/NkITaa/NL2SQL/tree/main/Prepared_Data).

## Model Training 
The [Training directory](https://github.com/NkITaa/NL2SQL/tree/main/Training) includes code for training relevant models on the prepared datasets. It consists of a generalised [notebook version](https://github.com/NkITaa/NL2SQL/tree/main/Training/notebooks) and a [script version](https://github.com/NkITaa/NL2SQL/tree/main/Training/scripts) for the training of the relevant model

## Inference 
The [Inference directory](https://github.com/NkITaa/NL2SQL/tree/main/Inference) provides code for applying the trained models to make inferences.

## Setup
The [setup_A100s](https://github.com/NkITaa/NL2SQL/blob/main/setup_A100s.rtf) file details the configuration of the Azure virtual machine used in this project.

## Accessing Trained Models
The trained models for this thesis are available on the [Huggingface platform](https://huggingface.co/BotoxBernd) 
