import os
import cv2 as cv
import csv
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

LABEL = {'bedroom': 0, 'coast': 1, 'forest': 2}
SIZES = {"small": (50, 50), "large": (200, 200)}


def preprocess():
    for set_type in ["Test", "Train"]:
        set_path = os.path.join("data", set_type)
        for class_type in os.listdir(set_path):
            class_path = os.path.join(set_path, class_type)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                if (img is None):
                    print("Image Not Found")
                    exit(0)
                for size, shape in SIZES.items():
                    resized = cv.resize(img, shape)
                    new_path = os.path.join(
                        "new_data", size, set_type.lower(), class_type.lower())
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    cv.imwrite(os.path.join(new_path, file), resized)
    return



def histExtract():
    
    fieldnames  = [x for x in range(256)]
    fieldnames.append('Y')

    set_type = ['train', 'test']
    for type in set_type:
        with open(f'hist_{type}.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for size in ["large", "small"]:
        
        size_path = os.path.join("new_data", size)
        for set_type in os.listdir(size_path):

            csvFile = open(f'hist_{set_type}.csv', 'a')
            writer = csv.writer(csvFile, delimiter=',', lineterminator='\n')

            set_path = os.path.join(size_path, set_type)
            for class_type in os.listdir(set_path):
                
                class_path = os.path.join(set_path, class_type)
                i = 0
                for file in os.listdir(class_path):
                    file_path = os.path.join(class_path, file)
                    img  = cv.imread(file_path)
                    if(img is None):
                        print(f'Image: {file_path} Not Found')
                        exit(0)

                    histSize = 256
                    histRange = (0, 256) # the upper boundary is exclusive
                    accumulate = False
                    hist = cv.calcHist(img, [0], None, [histSize], histRange, accumulate=accumulate)
                    hist = hist.reshape((256,))
                    hist = np.append(hist,LABEL[class_type])
                    writer.writerow(hist.astype(np.float32))
            csvFile.close()

    train = pd.read_csv('hist_train.csv')
    train_label = train.pop('Y').to_numpy(np.int32)
    train = train.to_numpy(np.float32)

    test = pd.read_csv('hist_test.csv')
    test_label = test.pop('Y').to_numpy(np.int32)
    test = test.to_numpy(np.float32)
    return train, train_label, test, test_label


def imgExtract(size):

    for i, set_type in enumerate(["train", "test"]):
        examples, labels = None, None
        set_path = os.path.join("new_data",size, set_type)
        for class_type in os.listdir(set_path):
            class_path = os.path.join(set_path, class_type)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                if(img is None):
                    print("Image Not Found")
                    exit(0)
                img = img.flatten()
                if(examples is None):
                    examples = img.astype(np.float32)
                    labels = np.asarray(LABEL[class_type])
                else:
                    examples = np.vstack([examples, img.astype(np.float32)])
                    labels = np.append(labels, LABEL[class_type])
                
        if i == 0:
            train = examples
            train_label = labels
        else:
            test = examples
            test_label = labels
    return train, train_label, test, test_label
   

def nearestNeighbor(train, train_label, test):
    
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_label)
    _, results, _, _ = knn.findNearest(test, 3)
    return results


def  siftExtract():
    desciptors = []
    sift = cv.SIFT_create()
    max_descriptor = 0
    desciptors2 = []

    for set_type in ["large"]:
        set_path1 = os.path.join("new_data", set_type, 'test')
        for folderName in os.listdir(set_path1):
            folderPath = os.path.join(set_path1, folderName)
            for file in os.listdir(folderPath):
                file_path = os.path.join(folderPath, file)
                img  = cv.imread(file_path)
                keypoints, descriptor = sift.detectAndCompute(img, None)
                if descriptor is not None:
                    descriptor = np.array(descriptor)
                    desciptors2.append((folderName, descriptor))
                    max_descriptor = max(max_descriptor, descriptor.shape[0])
    

    for i in range(len(desciptors2)):
        descriptor = desciptors2[i][1]
        if descriptor.shape[0] < max_descriptor:
            padded_descriptor = np.zeros((max_descriptor, descriptor.shape[1]))
            padded_descriptor[:descriptor.shape[0], :] = descriptor
            desciptors2[i] = (desciptors2[i][0], padded_descriptor)  
          

    with open('siftTest.pickle', 'wb') as f:
        pickle.dump(desciptors2, f)



    for set_type in ["large", "small"]:
        set_path1 = os.path.join("new_data", set_type, 'train')
        for folderName in os.listdir(set_path1):
            folderPath = os.path.join(set_path1, folderName)
            for file in os.listdir(folderPath):
                file_path = os.path.join(folderPath, file)
                img  = cv.imread(file_path)
                keypoints, descriptor = sift.detectAndCompute(img, None)
                if descriptor is not None:
                    descriptor = np.array(descriptor)
                    desciptors.append((folderName, descriptor))
                    max_descriptor = max(max_descriptor, descriptor.shape[0])
    
    for i in range(len(desciptors)):
        descriptor = desciptors[i][1]
        if descriptor.shape[0] < max_descriptor:
            padded_descriptor = np.zeros((max_descriptor, descriptor.shape[1]))
            padded_descriptor[:descriptor.shape[0], :] = descriptor
            desciptors[i] = (desciptors[i][0], padded_descriptor)
    

    with open('siftTrain.pickle', 'wb') as f:
        pickle.dump(desciptors, f)


#Takes in the confusion matrix and computes the percentages
def printSummary(actual, pred):
    print()
    confusion = np.zeros((3,3))
    for i in range(len(actual)):
        confusion[actual[i]][pred[i]] += 1
    
    acc = 0
    for i in range(3):
        acc += confusion[i][i]
    acc = acc/len(actual)
    print('accuracy: %.3f' % acc)
    
    confusion = pd.DataFrame(confusion, columns=list(LABEL.keys()), index=list(LABEL.keys()), dtype='int32')
    print(confusion)
    print()

def printResults(y_act, y_pred):

    matrix = confusion_matrix(y_act, y_pred)

    FP = matrix.sum(axis=0) - np.diag(matrix)  
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix).sum()
    TN = matrix.sum() - (FP + FN + TP)

    # Overall accuracy
    ACC = (TP)/(matrix.sum())
    # False positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)

    matrix = pd.DataFrame(matrix, columns=list(LABEL.keys()), index=list(LABEL.keys()), dtype='int32')
    print(matrix, '\n')
    print("Accuracy: ", ACC)
    result = pd.DataFrame([FPR, FNR], columns=list(LABEL.keys()),index=['FPR', 'FNR'])

    result['Average'] = result.mean(axis=1)
    print(result*100, '\n')



def knnClass(trainFile, testFile):

    with open(trainFile, 'rb') as f:
        trainData = pickle.load(f)
    
    with open(testFile, 'rb') as f:
        testData = pickle.load(f)


    X_train = np.array([x[1] for x in trainData])
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.array([x[0] for x in trainData])

    X_test = np.array([x[1] for x in testData])
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = np.array([x[0] for x in testData])

    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    printResults(y_test, prediction)

def smvClassifier(trainFile, testFile):
    with open(trainFile, 'rb') as f:
        trainData = pickle.load(f)
    
    with open(testFile, 'rb') as f:
        testData = pickle.load(f)


    X_train = np.array([x[1] for x in trainData])
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.array([x[0] for x in trainData])

    X_test = np.array([x[1] for x in testData])
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = np.array([x[0] for x in testData])

    smvC = svm.SVC(kernel='linear', C = 1, gamma= 'auto')
    smvC.fit(X_train, y_train)
    prediction = smvC.predict(X_test)
    printResults(y_test, prediction)
    


def main():
    if not os.path.exists("new_data"):
        preprocess()

    train, train_label, test, test_label = imgExtract('small')
    pred = nearestNeighbor(train, train_label, test)
    pred = pred.reshape(pred.shape[0],)
    pred = pred.astype(np.int32)

    print('KNN using pixel values:')
    printResults(test_label, pred)
    train, train_label, test, test_label = histExtract() 
    pred = nearestNeighbor(train, train_label, test)
    pred = pred.reshape(pred.shape[0],)
    pred = pred.astype(np.int32)

    print('KNN using histogram features:')
    printResults(test_label, pred)
    siftExtract()

    print('KNN using SIFT features')
    knnClass('siftTrain.pickle', 'siftTest.pickle')

    print('SVM using SIFT features')
    smvClassifier('siftTrain.pickle', 'siftTest.pickle')

if __name__ == "__main__":
    main()
