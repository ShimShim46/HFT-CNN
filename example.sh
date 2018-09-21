#! bin/bash
DataDIR=./Sample_data
Train=${DataDIR}/train.txt
Test=${DataDIR}/test.txt
Valid=${DataDIR}/valid.txt

## Embedding Weights Type (fastText .bin and .vec)
EmbeddingWeightsPath=./Word_embedding/
## Network Type (XML-CNN,  CNN-Flat,  CNN-Hierarchy,  CNN-fine-tuning or Pre-process)
ModelType=CNN-Flat
### the limit of the sequence 
USE_WORDS=13
### Tree file path
TreefilePath=./Tree/Amazon_all.tree

mkdir -p CNN
mkdir -p CNN/PARAMS
mkdir -p CNN/LOG
mkdir -p CNN/RESULT
mkdir -p Word_embedding

python train.py ${Train} ${Test} ${Valid} ${EmbeddingWeightsPath} ${ModelType} ${TreefilePath} ${USE_WORDS}