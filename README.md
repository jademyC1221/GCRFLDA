# GCRFLDA
A method for predicting lncRNA-disease potential associations
GCRFLDA: Scoring lncRNA-disease associations using graph convolution matrix completion with conditional random field

# Take Dataset 1 as an example：

Model Training： 
1. used TenfoldCrossvalidation.m to read lncRNA-disease associations and divided into five subsets for 5-fold cv. 

2. used dataset1tenfoldcvandsideinformation.m to calculate the gaussian interaction profile kernels similarity and cosine similarity of lncRNA and disease as side information. 

3. used GCRFLDA_main.py to train model, in which GCRFLDA_dataset.py was imported to read  and process data, GCRFLDA_model.py to construct GCRFLDA model. 

4. used GCRFLDA_para.py to adjust model parameters. 

Model Predicting: 

1. used casestudy.m to generate data for case study. We took all known associations as positive samples and randomly select the same number of unknown associations as negative samples to form the training set for model training. We took all unknown associations as testing samples. 

2. used GCRFLDA_casestudy.py to predict potential lncRNA-disease associations from all unknown associations, and sorted the potential associations according to predicted scores. 

# Dependency
pytorch 
matlab
