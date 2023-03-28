# 5th Place Solution to Google Landmark retrieval 2021 competition (Kaggle)
This repository contains the code for the fifth place solution to the Google Landmark retrieval 2021 competition hosted on Kaggle. The repository is jointly owned by tereka and KS.

# Competition Overview
The competition, hosted on Kaggle, challenged participants to develop models capable of retrieving the landmark images from a vast dataset of 4.1 million Google Landmark images (GLDv2). The primary objective of the competition was to create a retrieval model that can handle large-scale datasets efficiently.

# Links
### Kaggle competition page: https://www.kaggle.com/c/landmark-retrieval-2021/overview
### Inference code: https://www.kaggle.com/ks2019/landmark-retrieval-5th-place-inference-notebook
### ICCV Presentation: https://github.com/kumar-shubham-ml/5th-Place-Solution-to-Google-Landmark-Retrieval-2021/blob/main/ICCV_Presentation.pdf

# Proposed Solution

The proposed solution involves the following steps:

1. Train a model using a backbone and ArcFace module for GLDv2 dataset (4.1M).
2. Use gradient accumulation and mixed-precision training to train large image size with high batch size.
3. Use Adam 15ep training with cosine annealing and extract 512 embedding vectors from each model.
4. Create concatenation vectors (2560).
5. Use pre-compute embedding, which computes in local and uploads all (it uses post-process).
6. Use post-processing to improve the retrieval performance.
## Training
The model architecture is backbone+ArcFace module+Average Pooling, and the backbone list is as follows:

EfficientNet v2s 800
EfficientNet v2m 732
EfficientNet v2m 640
EfficientNet v2l 720
EfficientNet v2xl 640 (training 512)

## Post-Processing
The post-processing step consists of three parts: Bridged Distance Computation, Direct Distance Computation, and Reranking.

### Part A - Bridged Distance Computation

#### Step 1: KNN with cosine distance to get top 300 neighbours 
Compute the distance between each test and index image and landmark id by picking the top 300 neighbors from the train images using KNN (RAPIDS).
Penalize each train image by non-landmark distances, computed using non-landmark images from the 2019 test set. Take the top 10 neighbors for each train image and get the average of their distances.
Compute the distance between each test image and landmark id by picking the top k (k=2) nearest train images belonging to the landmark from the test and averaging them.
Similarly, compute the distance between each index image and landmark id.
#### Step 2: Finding Top 5 Landmarks for each test and index images
Pick the top 5 nearest landmarks for each test and index image with the help of the distance calculated above.
Add a bonus confidence (0.5) to the nearest landmark for each test and index image (distance = distance - 0.5).
#### Step 3: Calculating Bridged Distance

Calculate the bridged distance between test and index images as the minimum of the maximum distance between (test, landmark_id) and (index, landmark_id) for all landmark_ids.

Bridged Distance between test and index images is calculated as: 
         Min( Max((test,landmark_id),(index,landmark_id)) for all landmark_id )
         
### Part B - DBA + Direct Distance Computation
Apply DBA (Database Augmentation) on index and test embeddings and then apply KNN to retrieve top k neighbours

### PART C - Reranking
After computing the direct and bridged distances, we use a reranking approach to further improve the results.

First, we convert all distances into confidences by taking their inverse (i.e., confidence = 1/distance). Then, we combine the direct and bridged confidences using a power average approach, where we raise each confidence to the power of 3 and take the average.

The resulting confidence score is then used to rank the index images for each test image, and we select the top 100 index images with the highest confidence score.

## Postprocessing Optimisation
All of the hyperparameters used in the postprocessing step were optimized on the 2019 index/test dataset.

# Hardware
We would like to give a special mention to the Google TPU Research Program, which provided us with access to V3 TPUs on GCP. This allowed us to train larger models in the last few days of the competition and ultimately led to our fifth-place finish.

If you would like to reproduce our results or use our code for your own projects, please see the links provided in the repository for more information.
