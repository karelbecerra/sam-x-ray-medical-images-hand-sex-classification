## Citation (TBD by 2025)
```
@article{MOLLINEDA2025111060,
title = {Sex classification from hand X-ray images in pediatric patients: How zero-shot Segment Anything Model (SAM) can improve medical image analysis},
journal = {Computers in Biology and Medicine},
volume = {197},
pages = {111060},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2025.111060},
url = {https://www.sciencedirect.com/science/article/pii/S001048252501412X},
author = {Ramón A. Mollineda and Karel Becerra and Boris Mederos},
keywords = {Sex classification, Hand X-ray images, Segment Anything Model, X-ray image segmentation, X-ray image classification, Prehistoric handprints},
abstract = {The potential to classify sex from hand data is a valuable tool in both forensic and anthropological sciences. This work presents possibly the most comprehensive study to date of sex classification from hand X-ray images. The research methodology involves a systematic evaluation of zero-shot Segment Anything Model (SAM) in X-ray image segmentation, a novel hand mask detection algorithm based on geometric criteria leveraging human knowledge (avoiding costly retraining and prompt engineering), the comparison of multiple X-ray image representations including hand bone structure and hand silhouette, a rigorous application of deep learning models and ensemble strategies, visual explainability of decisions by aggregating attribution maps from multiple models, and the transfer of models trained from hand silhouettes to sex prediction of prehistoric handprints. Training and evaluation of deep learning models were performed using the RSNA Pediatric Bone Age dataset, a collection of hand X-ray images from pediatric patients. Results showed very high effectiveness of zero-shot SAM in segmenting X-ray images, the contribution of segmenting before classifying X-ray images, hand sex classification accuracy above 95% on test data, and predictions from ancient handprints highly consistent with previous hypotheses based on sexually dimorphic features. Attention maps highlighted the carpometacarpal joints in the female class and the radiocarpal joint in the male class as sex discriminant traits. These findings are anatomically very close to previous evidence reported under different databases, classification models and visualization techniques.}
}
```
This work includes three primary steps: **Segmentaion, Classification, and Visualization**

# Datasets (available in [Kaggle](https://www.kaggle.com/datasets/karelbecerra/sam-x-ray-medical-images-hand-sex-classification/))
![Original Hand Regions](figures/original-images.png)

# Segmentation, Classification, Visualization
This work includes three primary steps: **Segmentaion, Classification, and Visualization**

## Segmentation
(documentaion in progress: how segmentaion works) 
First step is to apply segmentation (SAM) on x-ray hand images
Segmentation example outcome
![Enhanced Hand Regions](figures/enhanced-hand-regions.png)

## Classification
With different variations of segmented images we proceed to classification: 

### Training
(inprogress: description on how to training)

### Inference
(inprogress: description on how to run inference)

## Visualization
Finally applying CAMs the visualization of results

# Utils
Find Kaggle and Google Colab notebooks ready to vizualice datasets: check **kaggle** and **google-colab** folders

### Kaggle
Notebooks ready to run on Kaggle environment

### Google Colab
Notebooks ready to run on Google Colab environment