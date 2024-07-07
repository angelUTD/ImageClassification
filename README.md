# CS 4391 - Scene Recognition: Report

## About the Data 

 - Images are already split into train and test sets
 - Relatively small resolutions (~250 x 250)
 - Mostly grayscale images
 - Three image classes:
   * Bedroom
   * Coast
   * Forest

## Preprocessing
 1. Every image is explicitly converted into grayscale
 2. The images are resized to (50 x 50) and (200 x 200) and saved in new folders (small) and (large) respectively.
 3. Using OpenCV, the SIFT and histogram features are extracted and saved to a file.



Old file structure:
```
.\data\
    |
    |--Train
    |    |--Bedroom
    |    |--Coast
    |    |--Forest
    |
    |--Test
    |    |--Bedroom
    |    |--Coast
    |    |--Forest
```
New file structure:
```
.\data\
    |--Small
    |    |--Train
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
    |    |
    |    |--Test
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
    |
    |--Large
    |    |--Train
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
    |    |
    |    |--Test
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
```
# Feature Engineering

## Image Extraction
To represent the images' pixel values directly, we had to flatten the images and put them in a format suitible for existing machine learning functions. Because the images were only in grayscale, we did not need to worry about extra color channels and we could simply flatten the image and append it to a matrix-like object. In this case, the the matrix of flattend images were saved as numpy.ndarrays.
## HIST Extraction
Using OpenCV's calcHist() function, we were able create a histogram for each image. Once the histgram was created it was flattend and appended to a csv file.
## Sift Extraction
In the function 'siftExtract()' is where the SIFT features that are used are extracted and stored in a pickle file is done. As seen in the diagram above some looping was needed in order to acces all the folders that contained the images. While looping through We used the sift.detectandcompute method available in open cv. We then stored the features in a list in order to sort through them and make them the all the same length, padding the ones that are too small with zeros. We then stored those descriptors along with the image scene in the pickle file, one for the test images and one for the train images.

# Machine Learning Models
## <u>KNN</u>
### Pixel Values and HIST
To learn the image based on raw image data and histrogram features, we used OpenCV's built-in KNearest model. Both models were trained using a k value of 3.
### SIFT
The function 'knnClass()' uses the sift features to train and test the Nearest Neighbor Classifier. It first extracts the features stored in the pickle file mentioned above, and reshapes the array so that it can be used in the sklearn's function. After the prep on the data is complete we then create an instance of KNeighborsClassifier called knn. Then the training data along with the training label are fitted into the model using 'knn.fit()' After which it is then fed the test image's data in order to predict which scene it falls under. We then pass the confusion matrix from the predicted results to a function to print the accuracy, False negative, and false positive percentages. 
## <u>Linear SVM classifier for Sift Features</u>
This was implemented in the same way as the Nearest Neighbor classifier mentioned above. First the data was loaded from the pickle files and then processed. It was then fed into the svm model provided from the sklearn library. Then it was tested on the testing image's features, the confusion was then fed to the 'printResults()' function. 
# Results 
The evaluate the methods, we calculated the accuracy, false positive rate, and false negative rate. These were calculated using the confusion matrix for each model.

From the results below, it can be seen that the SVM classifier had the highest average accuracy over each class, however KNN with pixels values has the lowest false positive rate and KNN with SIFT has the lowest false negative rate.
```
KNN using pixel values:
         bedroom  coast  forest
bedroom       11    101       4
coast          9    239      12
forest        14    173      41

Accuracy:  0.4817880794701987
       bedroom      coast     forest   Average
FPR  11.057692  93.835616  12.698413  39.19724
FNR  26.515152   6.730769  39.121339  24.12242

KNN using histogram features:
         bedroom  coast  forest
bedroom      126     55      51
coast        252    160     108
forest       143     62     251

Accuracy:  0.4445364238410596
       bedroom      coast     forest    Average
FPR  69.911504  37.620579  34.120172  47.217418
FNR  16.485226  40.133779  27.628032  28.082346

KNN using SIFT features
         bedroom  coast  forest
bedroom        2    101      13
coast          6    235      19
forest         9     29     190

Accuracy:  0.706953642384106
       bedroom      coast     forest    Average
FPR  23.809524  85.526316  23.021583  44.119141
FNR  21.072089   5.530973   8.172043  11.591702

SVM using SIFT features
         bedroom  coast  forest
bedroom       93     17       6
coast         57    196       7
forest        34      1     193

Accuracy:  0.7980132450331126
       bedroom      coast     forest    Average
FPR  91.919192  31.034483  14.942529  45.965401
FNR   4.554455  11.721612   6.769826   7.681964
```