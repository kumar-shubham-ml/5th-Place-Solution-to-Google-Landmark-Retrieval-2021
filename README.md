# 5th Place Solution to Google Landmark retrieval 2021 competition (Kaggle)

This repo contains training and inference code for fifth place solution to Google Landmark retrieval 2021 competition hosted on Kaggle. This repo is jointly owned by [tereka](https://github.com/tereka114) and [KS](https://github.com/kumar-shubham-ml)

Brief Summary about competition & proposed solution -> To be added

Competition: https://www.kaggle.com/c/landmark-retrieval-2021/overview
Inference Code: https://www.kaggle.com/ks2019/landmark-retrieval-5th-place-inference-notebook

## Train

Backbone + ArcFace, GLDv2 Training(4.1M)
Post Process(WDA, bridged confidence etc….)
Solution
Models
Model architecture is backbone+ArcFace module+Average Pooling.
The backbone list is here.

EfficientNet v2s 800
EfficientNet v2m 732
EfficientNet v2m 640
EfficientNet v2l 720
EfficientNet v2xl 640(training 512)
Training data is gldv2(4.1M). in order to train large image size with high batch size, we used gradient accumulation and mixed-precision training,(above v2s)

We use Adam 15ep training with Cosine Annealing and extract 512 embedding vectors from each model. After we create concatenation vectors(2560).

our first result is KNN of GPU(cuml) method.
KNN in GPU is faster than CPU, it's a great performance.

Also, we use pre-compute embedding, it computes in local and uploads all.(it use post-process)

## Post-Process

Many a times we can't match index and test images by direct KNN (for example: indoor/outdoor images). Our postprocessing is based on using train images to find these connections (we call this bridged connections). This postprocessing alone gives boost of 0.085-0.09 on public/private lb.

### PART A - Bridged Distance Computation

#### Step 1: Computing Distance between (test and landmark) and (index and landmark)

1. So, for each test and index image, we pick top 300 neighbours from train images using KNN (RAPIDS)
2. We penalise each train image by non landmark distances. Non landmark scores for each train image is computed using non landmark images from 2019 test set. We take top 10 neighbours for each train image and get the average of their distances. (2020 Recognition 1st place solution)
3. We compute distance between each test image and landmark id -> Pick top k (k=2) nearest train images belonging to landmark from test and average them
4. Similarly, we compute distance between each index image and landmark id 

#### Step 2: Finding Top 5 Landmarks for each test and index images
5. We pick top 5 nearest landmarks for each test and index images with the help of distance calculated above
6. We add bonus confidence (0.5) to the nearest landmark for each test and index image (distance = distance - 0.5)

#### Step 3: Calculating Bridged Distance
7. Bridged Distance between test and index images is calculated as: 

         Min( Max((test,landmark_id),(index,landmark_id)) for all landmark_id )

### PART B - Direct Distance Computation

1. We apply DBA (Database Augmentation) on index and test embeddings and then apply KNN.  We found that the direct connections have very high precision at high confidences but bridged connections bring more neighbours.

### PART C - Reranking

1. We converted all distances into confidence -> Confidence = 1-distance
2. We aggregated direct and bridged confidence using power average approach (3rd place solution for recognition 2020)

        Final Confidence = Direct Confidence**3 + Bridged Confidence**3
        
3. We pick top 100 index images for each test image using the confidence score calculated above. 

### Postprocessing Optimisation:

All the post-processing hyperparmeters were optimised on 2019 index/test dataset.


# Hardware
Special mention to Google TPU Research Program  (https://sites.research.google/trc/) which helped us in training bigger models in the last few days of the competition by supporting us with V3 TPUs on GCP. 
