#Introduction

This repository Uses OpenCV pre-trained Caffe deep learning model to recognize faces.
Dynamically Creating DataSet of Images to label, using Bing Search API

To Get BIng Search API Endpoints and API Key Follow below URLs to Register and fetch key:

Go to https://azure.microsoft.com/en-us/try/cognitive-services/?api=bing-image-search-api and register yourself

Once registered visit: https://azure.microsoft.com/en-us/try/cognitive-services/my-apis/

#Requiste Libraries:

pip install requests
pip install opencv-contrib-python
pip install scikit-learn
 
#Steps to Proceed:
 
Create a file bing_search_image_api.python
 
Create Folder dataset/amirkhan
 
Run the program 
 
#Creates our Data set with below search query in provided path
python search_bing_api.py --query "Amir Khan" --output dataset/amirkhan
 
#Face Recognition Steps: 
 
Embedding data to a dictionary and then serialize the data  in a pickle file with available Datasets
 
python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model \ 
--embedding-model openface_nn4.small2.v1.t7
 
Train Using  Linear Support Vector Machine model on top of embeddings
 
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
 
Test input images to detect faces:
 
python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 \ 
--recognizer output/recognizer.pickle --le output/le.pickle --image images/amir.jpg
