# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 23:39:35 2025

@author: dayoub
"""


import numpy as np
import math

# x_step = 16
# frame_len = 16
target_classes = [1,2,3,4,5,6,7,8,9,
                  20,30,40,50,60,70,
                  100,200,300,400,500,#600,700,
                  1000,2000,3000,4000,5000]

def lastNoneZeroIndex_x_6_6(data):
         last = data.shape[0]-1
         for i in range(data.shape[0]-1, 0, -1):
             if ((data[i,:,:] != 0).all()):
                 if i<last:
                     last = i
                 break
         return last

def loadDatasetFiles():
    print('loading the datasets')
    d2 = np.load('./twoDigitsContDataset.npy')
    d2 = np.moveaxis(d2, 1, 2)
    d3 = np.load('./threeDigitsContDataset.npy')
    d3 = np.moveaxis(d3, 1, 2)
    d4 = np.load('./fourDigitsContDataset.npy')
    d4 = np.moveaxis(d4, 1, 2) 
    d6 = np.load('./sixDigitsContDataset.npy')
    d6 = np.moveaxis(d6, 1, 2)
    
    # d2 = d2[:,:114:2,:,:] + d2[:,1:114:2,:,:] / math.sqrt(2)
    # d3 = d3[:,:144:2,:,:] + d3[:,1:144:2,:,:] / math.sqrt(2)
    # d4 = d4[:,::2,:,:] + d4[:,1::2,:,:] / math.sqrt(2)
    # d6 = d6[:,::2,:,:] + d6[:,1::2,:,:] / math.sqrt(2)
    
    d2i = np.load('./twoDigFirstClassEndSecondClassStart.npy')#//2
    d3indices = np.load('./threeDigsStartAndEnds.npy')#//2
    d4indices = np.load('./fourDigsStartAndEnds.npy')#//2
  
    
    d2label = np.load('./twoDigitsContDatasetLabels.npy')
    d3label = np.load('./threeDigitsContDatasetLabels.npy')
    d4label = np.load('./fourDigitsContDatasetLabels.npy')
    d6l = np.load('./sixDigitsContDatasetLabels.npy')
    d2label = d2label.astype('int')
    d3label = d3label.astype('int')
    d4label = d4label.astype('int')
    d6l = d6l.astype('int')
    
    
   
    
    d2indices = np.zeros((d2i.shape[0], 4))
    d2indices[:,1:3] = d2i
    for i in range(d2indices.shape[0]):
            d2indices[i, 3] = lastNoneZeroIndex_x_6_6(d2[i])

    d2indices = d2indices.astype('int')
    d3indices = d3indices.astype('int')
    d4indices = d4indices.astype('int')
    
    d4 = d4[:430]
    d4label = d4label[:430]
    
    d2l = d2label
    d2i = d2indices
    d3l = d3label
    d3i = d3indices
    d4l = d4label
    d4i = d4indices
    
    return d2, d3, d4, d6, d2l, d3l, d4l, d6l, d2i, d3i, d4i

d2ds, d3ds, d4ds, d6ds, d2Lbl, d3Lbl, d4Lbl, d6Lbl, d2Ind, d3Ind, d4Ind = loadDatasetFiles()




def removeUnwantedClasses():
    mask = np.all(np.isin(d2Lbl, target_classes), axis=1)
    a1 = d2ds[mask]
    b1 = d2Lbl[mask]
    c1 = d2Ind[mask]
    
    mask = np.all(np.isin(d3Lbl, target_classes), axis=1)
    a2 = d3ds[mask]
    b2 = d3Lbl[mask]
    c2 = d3Ind[mask]
    
    mask = np.all(np.isin(d4Lbl, target_classes), axis=1)
    a3 = d4ds[mask]
    b3 = d4Lbl[mask]
    c3 = d4Ind[mask]
    
    mask = np.all(np.isin(d6Lbl, target_classes), axis=1)
    a4 = d6ds[mask]
    b4 = d6Lbl[mask]
    
    return a1,a2,a3,a4,b1,b2,b3,b4, c1,c2,c3



d2ds, d3ds, d4ds, d6ds, d2Lbl, d3Lbl, d4Lbl, d6Lbl, d2Ind, d3Ind, d4Ind = removeUnwantedClasses()

def create_slice_data(data, labels, indices):
        lnzi  = lastNoneZeroIndex_x_6_6(data)
        data = data[:lnzi+1]
        
        slices = []
        lbls = []
        for i in range(0, data.shape[0], x_step):
            if i+frame_len > data.shape[0]:
                slices.append(data[data.shape[0]-frame_len:data.shape[0]])
            else:
                slices.append(data[i:i+frame_len])
            
            offset = i + 2
            
            if len(labels) == 2:
                if offset <= indices[1]:
                    lbls.append(labels[0])
                elif offset < indices[2]:
                    lbls.append(0)
                else:
                    lbls.append(labels[1])
                
            elif len(labels) == 3:
                if offset <= indices[1]:
                    lbls.append(labels[0])
                    
                elif offset < indices[2]:
                    lbls.append(0)
                    
                elif offset < indices[3]:
                    lbls.append(labels[1])
                    
                elif offset < indices[4]:
                    lbls.append(0)
                    
                else:
                     lbls.append(labels[2])
                     
            elif len(labels) == 4:
                if offset <= indices[1]:
                    lbls.append(labels[0])
                    
                elif offset < indices[2]:
                    lbls.append(0)
                    
                elif offset < indices[3]:
                    lbls.append(labels[1])
                    
                elif offset < indices[4]:
                    lbls.append(0)
                    
                elif offset <= indices[5]:
                     lbls.append(labels[2])
                     
                elif offset < indices[6]:
                    lbls.append(0)
                    
                else:
                     lbls.append(labels[3])
            
                
        
        return slices, lbls


def createTrainDataset():
    newDataset = []
    newLabels = []
    
    for i in range(d2ds.shape[0]):
        curData = d2ds[i]
        
        data, labels = create_slice_data(curData, d2Lbl[i], d2Ind[i])
        # if len(data)!=len(labels):
        #     print(i)
        #     return data, labels
        newDataset += data
        newLabels  += labels
    
    for i in range(d3ds.shape[0]):
        curData = d3ds[i]
        
        data, labels = create_slice_data(curData, d3Lbl[i], d3Ind[i])
        # if len(data)!=len(labels):
        #     print(i)
        #     return data, labels
        newDataset += data
        newLabels  += labels
    
    
    for i in range(d4ds.shape[0]):
        curData = d4ds[i]
        
        data, labels = create_slice_data(curData, d4Lbl[i], d4Ind[i])
        # if len(data)!=len(labels):
        #     print(i)
        #     return data, labels
        newDataset += data
        newLabels  += labels
            
    return newDataset, newLabels


def wer(reference, hypothesis) -> float:
    
        # Initialize the matrix
        d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.int32)
        
        # Fill base case (distance to empty string)
        for i in range(len(reference) + 1):
            d[i][0] = i
        for j in range(len(hypothesis) + 1):
            d[0][j] = j
        
        # Compute edit distance
        for i in range(1, len(reference) + 1):
            for j in range(1, len(hypothesis) + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    d[i][j] = d[i - 1][j - 1]  # No operation needed
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,    # Deletion
                        d[i][j - 1] + 1,    # Insertion
                        d[i - 1][j - 1] + 1 # Substitution
                    )
        
        # WER calculation
        wer_value = d[len(reference)][len(hypothesis)] / max(len(reference), 1)
        return wer_value

def WER(reference, hypothesis) -> float:
    
        d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]  
        
        for i in range(len(reference) + 1):  
            for j in range(len(hypothesis) + 1):  
                if i == 0:  
                    d[i][j] = j  
                elif j == 0:  
                    d[i][j] = i  
                else:  
                    d[i][j] = min(d[i-1][j] + 1,    # حذف  
                                  d[i][j-1] + 1,    # اضافه  
                                  d[i-1][j-1] + (reference[i-1] != hypothesis[j-1]))  # جایگزینی  

        error_count = d[len(reference)][len(hypothesis)]/max(len(reference), 1)
        
     
        correct_words = []  
        incorrect_words = []  
        
        ref_index = len(reference)  
        hyp_index = len(hypothesis)  

        while ref_index > 0 and hyp_index > 0:  
            if reference[ref_index - 1] == hypothesis[hyp_index - 1]:  
                correct_words.append(reference[ref_index - 1])  
                ref_index -= 1  
                hyp_index -= 1  
            elif d[ref_index][hyp_index] == d[ref_index - 1][hyp_index] + 1:  
                incorrect_words.append(reference[ref_index - 1])  # کلمه در مرجع وجود دارد اما در تشخیص نیست  
                ref_index -= 1  
            elif d[ref_index][hyp_index] == d[ref_index][hyp_index - 1] + 1:  
                hyp_index -= 1  # کلمه اضافی در تشخیص  
            else:  
                incorrect_words.append(reference[ref_index - 1])  # کلمه جایگزین شده  
                ref_index -= 1  
                hyp_index -= 1  

        while ref_index > 0:  
            incorrect_words.append(reference[ref_index - 1])  
            ref_index -= 1  

        while hyp_index > 0:  
            hyp_index -= 1  

        correct_words.reverse()  # نظم کلمات صحیح را برگردانیم  

        return error_count, correct_words, incorrect_words 
    
from collections import Counter
def get_cat(clss):
        # cat1 = [1,2,3,4,5,6,7,8,9]
        # cat2 = [10,11,12,13,14,15,16,17,18,19]
        # cat3 = [20,30,40,50,60,70]
        # cat4 = [100,200,300,400,500,600,700]
        # cat5 = [1000,2000,3000,4000,5000,6000,7000]
        cat1 = [1,2,3,4,5,6,7,8,9]
        
        cat2 = [20,21,22,23,24,25]
        cat3 = [26,27,28,29,30,31,32]
        cat4 = [33]
        if clss in cat1:
            return 1
        if clss in cat2:
            return 2
        if clss in cat3:
            return 3
        if clss in cat4:
            return 4
        # if clss in cat5:
        #     return 5


def majority(A):
    A = [x for x in A if x!=0]
    if A == []:
        return [0]
    C = []
    B = []
    
    current_category = get_cat(A[0])
    temp_group = []
    
    for num in A:
        category = get_cat(num)
        if category == current_category:
            temp_group.append(num)
        else:
            B.append(temp_group)
            temp_group = [num]
            current_category = category
    
    if temp_group:
        B.append(temp_group)
    
    for group in B:
        if not group:
            continue
        count = Counter(group)
        max_freq = max(count.values())
        most_frequent = [num for num, freq in count.items() if freq == max_freq]
        C.extend(most_frequent)
    
    return C


def createTestDataset():
    
    for i in range(d6ds.shape[0]):
        curData = d6ds[i]
        lnzi  = lastNoneZeroIndex_x_6_6(curData)
        curData = curData[:lnzi+1]
        slices = []
        for j in range(0, lnzi, x_step):
            if j+frame_len > lnzi:
                    slices.append(curData[lnzi-frame_len:lnzi])
            else:
                    slices.append(curData[j:j+frame_len])
        
        # x_test_multi_digit.append(slices)
        for sl in slices:
            x_test_multi_digit.append(sl)
        x_test_frame_size.append(len(slices))





from tensorflow.keras import layers, models 
def residual_block(inputs, filters, stride=1):
    shortcut = inputs
    x = layers.Conv2D(filters, kernel_size=(3,1), strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=(3,1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def create_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, kernel_size=(2,1), strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    
    x = residual_block(x, filters=64)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    return model
# dot_img_file = '/tmp/model_1.png'
# keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)   

werArray = []
for frame_len in [16, 18,20]:#8,10,,40
    for x_step in [frame_len//2,frame_len]:

        x_train, y_train = createTrainDataset()
        x_train = np.array(x_train)
        clss = np.unique(y_train)
        
        for i in range(len(y_train)):
            idx = np.where(clss==y_train[i])[0][0]
            y_train[i] = idx
        
        y_train = np.array(y_train, dtype=int)
        x_test_multi_digit = []
        y_test_multi_digit = []
        x_test_frame_size = []
        
        num_classes = len(target_classes) + 1
        input_shape = x_train.shape[1:]
        resnet_model = create_resnet(input_shape, num_classes)
        resnet_model.compile(optimizer='adam', 
                              loss = 'sparse_categorical_crossentropy', 
                              metrics=['accuracy'])
        
        net_hist = resnet_model.fit(x_train, y_train, batch_size=5, epochs=10, validation_split=0.2)
        
        
        
        createTestDataset()
        
        y_pred= np.argmax(resnet_model.predict(np.array(x_test_multi_digit)), axis=1)
        
        allTrueResult = [[target_classes.index(x)+1 for x in sublist] for sublist in d6Lbl]  
        allPredResult = []
        
        first = 0
        for i in range(len(x_test_frame_size)):
            allPredResult.append(y_pred[first:first+x_test_frame_size[i]])
            first += x_test_frame_size[i]
        
        
        werall = 0
        
        for i in range(len(allPredResult)):
        
            ref = majority(list(allTrueResult[i]))
            hyp = majority(list(allPredResult[i]))
            
            werall += wer(ref, hyp)
            
        werall /= len(allTrueResult)
        
        print("final WER: ", werall)
        
        werArray.append([werall, frame_len, x_step])



werSum = 0
corWord = []
incorWord = []

for i1 in range(len(allPredResult)):
    ref = majority(list(allTrueResult[i1]))
    hyp = majority(list(allPredResult[i1]))
    w, c, inc = WER(ref, hyp)
    werSum += w
    corWord.append(c)
    incorWord.append(inc)

werall = werSum / len(allPredResult)

print("final WER: ", werall)


lstCor = []
lstIncor = []
for l in range(len(corWord)):
    for m in corWord[l]:
        lstCor.append(m)

for l in range(len(incorWord)):
    for m in incorWord[l]:
        lstIncor.append(m)
lstCor = np.array(lstCor)
lstIncor=np.array(lstIncor)

itmCor, cntCor = np.unique(lstCor,return_counts=True)
itmIncor, cntIncor = np.unique(lstIncor,return_counts=True) 























