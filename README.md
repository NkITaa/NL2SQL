Repository for the Bachelors Thesis:
"Ein Vergleich zwischen Chunking- und Prompt-Strategien für die Verarbeitung großer Textdaten in Sprachmodellen: Eine Untersuchung basierend auf dem Blar AI Framework"

The Data_Preparing directory contains the script for preparing the raw Spider [Spider](https://yale-lily.github.io//spider) and BIRD (https://bird-bench.github.io/) datasets into the Prepared_Data CSVs

The Training Directory contains the code in order to train the relevant models on the Prepared_Data data.

The Inference Directory contains the relevant code for infering the trained models

The Prepared_Data_Predicted Directory contains the Datasets where the predicted_schema_links from the Schema Links Model are saved

The setup A100s File contains the Setup of the Azure virtual machine

Lastly the trained models can be found on huggingface: https://huggingface.co/BotoxBernd
