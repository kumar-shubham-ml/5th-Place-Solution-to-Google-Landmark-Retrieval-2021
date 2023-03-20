This repository contains the training and inference code for the fifth place solution in the Google Landmark Retrieval 2021 competition hosted on Kaggle. The solution was jointly created by the repository's owners, tereka and KS.

The competition involved the retrieval of landmarks from a dataset of over four million images. The proposed solution involves using a backbone + ArcFace module + Average Pooling architecture with efficientNet v2s 800, v2m 732, v2m 640, v2l 720, and v2xl 640 as the backbone models. Gradient accumulation and mixed-precision training were employed to enable training with high batch sizes and large image sizes.

The training data was GLDv2 (4.1M), and 512 embedding vectors were extracted from each model. These vectors were concatenated into vectors of size 2560. Pre-computed embedding was used, which computed embeddings locally and then uploaded them.

Post-processing involved using train images to find bridged connections since direct KNN matches between index and test images were not always possible, such as for indoor/outdoor images. The post-processing involved three parts: bridged distance computation, direct distance computation, and re-ranking.

In the first part, distances between test and landmark, and index and landmark were computed using a penalization technique. In the second part, the top five nearest landmarks were selected for each test and index image, and a bonus confidence of 0.5 was added to the nearest landmark. In the third part, bridged distance between test and index images was calculated using a power average approach. Direct distance computation involved applying Database Augmentation (DBA) on index and test embeddings and then applying KNN. The final confidence was calculated by aggregating the direct and bridged confidence scores.

Bridged Distance between test and index images is calculated as: 

         Min( Max((test,landmark_id),(index,landmark_id)) for all landmark_id )

All the post-processing hyperparameters were optimized on the 2019 index/test dataset. The solution was trained using Adam 15ep training with Cosine Annealing. The training was carried out using Google TPU Research Program's V3 TPUs on GCP in the last few days of the competition.

The repository also contains a presentation on the solution, an inference code, and the backbone models used.

#### Competition: https://www.kaggle.com/c/landmark-retrieval-2021/overview
#### Inference Code: https://www.kaggle.com/ks2019/landmark-retrieval-5th-place-inference-notebook
#### ICCV Presentation: https://github.com/kumar-shubham-ml/5th-Place-Solution-to-Google-Landmark-Retrieval-2021/blob/main/ICCV_Presentation.pdf
