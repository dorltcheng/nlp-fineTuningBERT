# Clinical Decision Support System with Deep Learning (Natural Language Processing)

This project focuses on the Natural Language Processing (NLP) section of the clinical decision supporting system for cardio-pneumological diseases, in collaboration with the Third Eye Intelligence, as part of the Bachelor Year Group Project of MEng Biomedical Engineering (Computational Bioengineering), Imperial College London. The paper for this study can be found here: https://drive.google.com/file/d/1JKRXKfszJk8KMNhxOEtvyNehMl98FHNn/view?usp=sharing

In this section, we aim to fine-tune the BERT-based [Bio+ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) onto all radiology reports available on the [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) such that our models can understand and be able to classify different diagnoses between reports.  All the trained models can be found in [Hugging Face](https://huggingface.co/ICLbioengNLP).

Codes for our deep learning models are written in Python and implemented with PyTorch 1.7.1. 

## Masked Language Modelling

The [CXR_BioClinicalBERT_MLM](https://huggingface.co/ICLbioengNLP/CXR_BioClinicalBERT_MLM) was fine-tuned on a Masked Language Modelling (MLM) task. The model is trained to predict text by attempting to recover the whole word, such that we can validate model’s understanding of radiological contents. The model achieved a perplexity score of 1.0710 after 10 epochs of training. 

The contextualized word embeddings output were converted into sentence embeddings by a mean pooling operation. Sentence embeddings of different radiological reports were semantically compared using the cosine similarity calculation. Examples of the results can be found in the paper and `ReportSimilarity_sections.ipynb`. 

## Text Classification 

The [CXR_BioClinicalBERT_Class](https://huggingface.co/ICLbioengNLP/CXR_BioClinicalBERT_Class) was fine-tuned from the `CXR_BioClinicalBERT_MLM` model on a classification task, such that it can perform multi-label text classification across 13 different cardiopulmonary conditions. Classification evaluation can be found in the paper and `CXR_BioClinicalBERT_Class.ipynb`. Prediction examples can be found in `TC_prediction.ipynb`. 

## Dataset and Pre-processing

The [MIMIC-CXR database (v2.0.0)](https://physionet.org/content/mimic-cxr/2.0.0/) was used for training, which is the largest publicly available Chest X-ray dataset containing 377,110 radiographs and 225,606 associated radiology reports in free-text format. 

Only simple text pre-processing including punctuation and number removal was applied. Typical steps including stemming, lemmatization and stopword removal were avoided in training with highly context-dependent transformer models like BERT. 

