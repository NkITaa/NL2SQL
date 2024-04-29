# Bachelors Thesis: A Comparison of Chunking and Prompting Strategies for Large Text Data Processing in Language Models

##Abstract
This thesis investigates the effectiveness of chunking and prompting techniques for handling large text datasets in language models, specifically utilizing the Blar AI Framework. Datasets such as Spider and BIRD are employed in this study, and models are trained on prepared data.
Table of Contents

Data Preparation: #data-preparation
Model Training: #model-training
Inference: #inference
Results: #results
Setup: #setup
Accessing Trained Models: #accessing-trained-models
Data Preparation

The Data_Preparing directory contains scripts for transforming the raw Spider (Spider: https://yale-lily.github.io//spider) and BIRD (BIRD: https://bird-bench.github.io/) datasets into Prepared_Data CSVs.
Model Training

The Training directory includes code for training relevant models on the prepared datasets.
Inference

The Inference directory provides code for applying the trained models to make inferences.
Results

The Prepared_Data_Predicted directory houses datasets containing the predicted_schema_links generated by the Schema Links Model.
Setup

The setup_A100s file details the configuration of the Azure virtual machine used in this project.
Accessing Trained Models

The trained models for this thesis are available on the Huggingface platform Huggingface: https://huggingface.co/BotoxBernd.
