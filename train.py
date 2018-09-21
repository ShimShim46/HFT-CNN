#!/usr/bin/env python
import sys
from collections import defaultdict
import numpy as np
import data_helper
import cnn_train
import random
import scipy.sparse as sp
import tree
import os

def train_problem(currentDepth, upperDepth, classNum, fineTuning, embeddingWeights, inputData, modelType, learning_categories):
    params = {"gpu":0, 
                "outchannels":128,
                "embedding-dimensions":300, 
                "epoch":40, 
                "batchsize":100,
                "unit":1024, 
                "output-dimensions":int(classNum), 
                "fineTuning":int(fineTuning), 
                "currentDepth":currentDepth, 
                "upperDepth":upperDepth, 
                "embeddingWeights": embeddingWeights,
                "inputData": inputData,
                "model-type": modelType,
                "learning_categories": learning_categories
                }
    if params["model-type"] == "XML-CNN":
        params["unit"] = 512 # compact representation
    if (params["model-type"] == "CNN-fine-tuning") and (currentDepth == "1st"):
        params["fineTuning"] = 0

    if (currentDepth == "1st") and ((params["model-type"] == "CNN-fine-tuning") or  (params["model-type"] == "CNN-Hierarchy")):
        network_output = cnn_train.load_top_level_weights(params)
    else:
        network_output = cnn_train.main(params)
    
    return network_output

def make_labels_hie_info_dic(treePath):
        label_hierarchical_info_dic = {}
        with open(treePath, "r") as f:
            for line in f:
                line = line[:-1]
                category = line.split("<")[-1]
                level = len(line.split("<"))
                if category not in label_hierarchical_info_dic:
                        label_hierarchical_info_dic[category] = level
        return label_hierarchical_info_dic

def make_labels_hie_list_dic(labels, label_hierarchical_info_dic):
        layer_category_list_dic = {}
        for i in range(1,max(label_hierarchical_info_dic.values())+1):
                a_set = set([])
                layer_category_list_dic[i] = a_set
        for label in labels:
            layer_category_list_dic[int(label_hierarchical_info_dic[label])].add(label)
    
        return layer_category_list_dic

def make_tree(treeFile_path):
    Tree = tree.make()
    with open(treeFile_path, mode="r") as f:
        for line in f:
            line = line[:-1]
            line = line.split("\t")[0]
            line = line.split("<")
            tree.add(Tree, line)
    return Tree




# Main processing
# ==================================================================
def main():
    random.seed(0)
    np.random.seed(0)

    # Loading data
    # ==========================================================
    print ('-'*50)
    print ('Loading data...')
    train = sys.argv[1]
    test = sys.argv[2]
    validation = sys.argv[3]
    embeddingWeights_path = sys.argv[4]
    modelType = sys.argv[5]
    treeFile_path = sys.argv[6]
    useWords = int(sys.argv[7])

    f_train = open(train, 'r')
    train_lines = f_train.readlines()
    f_test = open(test, 'r')
    test_lines = f_test.readlines()
    f_valid = open(validation, 'r')
    valid_lines = f_valid.readlines()
    f_train.close()
    f_test.close()
    f_valid.close()

    # Building Hierarchical information
    # =========================================================
    category_hie_info_dic = make_labels_hie_info_dic(treeFile_path)
    input_data_dic = data_helper.data_load(train_lines, valid_lines, test_lines, category_hie_info_dic, useWords)
    category_hie_list_dic = make_labels_hie_list_dic(list(input_data_dic['catgy'].keys()), category_hie_info_dic)
    # Loading Word embeddings
    # =========================================================
    print ('-'*50)
    print ("Loading Word embedings...")
    embeddingWeights=data_helper.embedding_weights_load(input_data_dic['vocab'], embeddingWeights_path)

    # Conditions of each model
    # =========================================================
    fineTuning = 0
    if modelType == "XML-CNN" or modelType == "CNN-Flat":
        categorizationType="flat"
        fineTuning = 0
    elif modelType == "CNN-Hierarchy":
        categorizationType="hierarchy"
        fineTuning = 0
    elif modelType == "CNN-fine-tuning":
        categorizationType="hierarchy"
        fineTuning = 1
    elif modelType == "Pre-process":
        categorizationType = "pre-process"
        fineTuning = 0
    else:
        raise TypeError('Unknown model type: %s!' % (modelType))

    
    # Processing in case of pro-processing
    # ========================================================
    if categorizationType == "pre-process":
        print ('-'*50)
        print ("Pre-process for hierarchical categorization...")
        Tree = make_tree(treeFile_path)
        layer = 1
        depth = data_helper.order_n(1)
        upperDepth = None
        learning_categories = sorted(category_hie_list_dic[layer])
        X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst =  data_helper.build_problem(learning_categories=learning_categories,depth=depth, input_data_dic=input_data_dic)
        input_network_data = {"X_trn":X_trn, "Y_trn":Y_trn, "X_val":X_val, "Y_val":Y_val, "X_tst":X_tst, "Y_tst":Y_tst}
        Y_pred = train_problem(currentDepth=depth, upperDepth=upperDepth, classNum=len(learning_categories), fineTuning=fineTuning, embeddingWeights=embeddingWeights, inputData=input_network_data, modelType=modelType, learning_categories=learning_categories)
        print ("Please change model-type to CNN-Hierarchy of CNN-fine-tuning.")
    
    
    # Processing in case of flat categorization
    # ========================================================
    elif categorizationType == "flat":
        print ('-'*50)
        print ("Processing in case of flat categorization...")
        from itertools import chain
        learning_categories = sorted(input_data_dic['catgy'].keys()) ## this order is network's output order.
        X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst =  data_helper.build_problem(learning_categories=learning_categories,depth="flat", input_data_dic=input_data_dic)
        input_network_data = {"X_trn":X_trn, "Y_trn":Y_trn, "X_val":X_val, "Y_val":Y_val, "X_tst":X_tst, "Y_tst":Y_tst}
        Y_pred = train_problem(currentDepth="flat", upperDepth=None, classNum=len(learning_categories), fineTuning=fineTuning, embeddingWeights=embeddingWeights, inputData=input_network_data, modelType=modelType, learning_categories=learning_categories)
        GrandLabels, PredResult = data_helper.get_catgy_mapping(learning_categories, Y_tst, Y_pred, "flat")
        data_helper.write_out_prediction(GrandLabels, PredResult, input_data_dic)
        
    # Processing in case of hierarchical categorization
    # ========================================================
    elif categorizationType == "hierarchy":
        if not os.path.exists("./CNN/PARAMS/parameters_for_multi_label_model_1st.npz"):
            raise FileNotFoundError('Please change "ModelType=CNN-Hierarchy" or "ModelType=CNN-fine-tuning" to "ModelType=Pre-process" in example.sh.')
        print ('-'*50)
        print ("Processing in case of hierarchical categorization...")
        upperDepth = None
        Y_tst_concat = [[] for i in range(len(input_data_dic['test']))]
        Y_pred_concat = [[] for i in range(len(input_data_dic['test']))]
        all_categories = []
        Tree = make_tree(treeFile_path)
        layers = list(category_hie_list_dic.keys())
        for layer in layers:
            depth = data_helper.order_n(layer)
            print ('-'*50)
            print ('Learning and categorization processing of ' + depth + ' layer')
            learning_categories = sorted(category_hie_list_dic[layer])
            X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst =  data_helper.build_problem(learning_categories=learning_categories,depth=depth, input_data_dic=input_data_dic)
            input_network_data = {"X_trn":X_trn, "Y_trn":Y_trn, "X_val":X_val, "Y_val":Y_val, "X_tst":X_tst, "Y_tst":Y_tst}
            Y_pred = train_problem(currentDepth=depth, upperDepth=upperDepth, classNum=len(learning_categories), fineTuning=fineTuning, embeddingWeights=embeddingWeights, inputData=input_network_data, modelType=modelType, learning_categories=learning_categories)
            GrandLabels, PredResult = data_helper.get_catgy_mapping(learning_categories, Y_tst, Y_pred, depth)
            upperDepth = depth
            for i in range(len(input_data_dic['test'])):
                Y_tst_concat[i].extend(GrandLabels[i])
            for i in range(len(input_data_dic['test'])):
                for y in PredResult[i]:
                    if (tree.search_parent(Tree, y) in Y_pred_concat[i]) or (tree.search_parent(Tree, y) == 'root'):
                        Y_pred_concat[i].append(y)

            all_categories += learning_categories
        
        print ('-'*50)
        print ('Final Result')
        data_helper.write_out_prediction(Y_tst_concat, Y_pred_concat, input_data_dic)

if __name__ == "__main__":
        main()
