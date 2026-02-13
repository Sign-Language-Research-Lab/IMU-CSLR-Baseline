# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 19:47:47 2025

@author: dayoub
"""
import tensorflow as tf
import numpy as np
import math
from collections import Counter



def lastNoneZeroIndex_x_6_6(data):
             last = data.shape[0]-1
             for i in range(data.shape[0]-1, 0, -1):
                 if ((data[i,:,:] != 0).all()):
                     if i<last:
                         last = i
                     break
             return last

        
def load_dataset():
            d2 = np.load('./twoDigitsContDataset.npy')
            d3 = np.load('./threeDigitsContDataset.npy')
            d4 = np.load('./fourDigitsContDataset.npy')
            d4 = d4[:430]
            d6 = np.load('./sixDigitsContDataset.npy')
        
            d2 = np.moveaxis(d2, 1, 2)
            d3 = np.moveaxis(d3, 1, 2)
            d4 = np.moveaxis(d4, 1, 2)
            d6 = np.moveaxis(d6, 1, 2)
        
            # d2 = d2//500
            # d3 = d3//500
            # d4 = d4//500
            # d6 = d6//500
        
            d2 = d2[:,:114:2,:,:] + d2[:,1:114:2,:,:] / math.sqrt(2)
            d3 = d3[:,:144:2,:,:] + d3[:,1:144:2,:,:] / math.sqrt(2)
            d4 = d4[:,::2,:,:] + d4[:,1::2,:,:] / math.sqrt(2)
            d6 = d6[:,::2,:,:] + d6[:,1::2,:,:] / math.sqrt(2)
        
        
            d2i = np.load('./twoDigFirstClassEndSecondClassStart.npy')//2
            d3i = np.load('./threeDigsStartAndEnds.npy')//2
            d4i = np.load('./fourDigsStartAndEnds.npy')//2
            d2i = np.astype(d2i, int)
            d3i = np.astype(d3i, int)
            d4i = np.astype(d4i, int)
            d2l = np.load('./twoDigitsContDatasetLabels.npy')
            d3l = np.load('./threeDigitsContDatasetLabels.npy')
            d4l = np.load('./fourDigitsContDatasetLabels.npy')
            d4l = d4l[:430]
            d6l = np.load('./sixDigitsContDatasetLabels.npy')
        
            d2l = np.astype(d2l, int)
            d3l = np.astype(d3l, int)
            d4l = np.astype(d4l, int)
            d6l = np.astype(d6l, int)
        
        
            for i in range(d3.shape[0]):
                start = d3i[i, 0]
                if start > 0:
                    d3[i, :d3.shape[1]-start] = d3[i, start:]
                    d3[i, d3.shape[1]-start:] = 0
                    d3i[i] -= start
                    
            for i in range(d4.shape[0]):
                start = d4i[i, 0]
                if start > 0:
                    d4[i, :d4.shape[1]-start] = d4[i, start:]
                    d4[i, d4.shape[1]-start:] = 0
                    d4i[i] -= start
            
            
            return d2, d3, d4, d6, d2l, d3l, d4l, d6l, d2i, d3i, d4i

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, image_width, image_height, patch_size, num_patches, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return self.projection(patches)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_width,
        image_height,
        patch_size,
        num_layers,
        num_heads,
        mlp_dim,
        num_classes,
        projection_dim,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_width // patch_size) * (image_height // patch_size)
        self.patch_embed = PatchEmbedding(
            image_width=image_width,
            image_height=image_height,
            patch_size=patch_size,
            num_patches=num_patches,
            projection_dim=projection_dim,
        )
        self.transformer_blocks = [TransformerBlock(projection_dim, num_heads, mlp_dim) for _ in range(num_layers)]

        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.patch_embed(x)

        for block in self.transformer_blocks:
           x = block(x, training=training)  # Pass 'training' as a keyword argument

        x = tf.reduce_mean(x, axis=1) # Global average pooling
        x = self.classifier(x)

        return x

def get_train_data_patches(contData, cdl, cdi, stride):
    frames = []
    frm_clss = []
    lnzi = lastNoneZeroIndex_x_6_6(contData)
    
    contData = contData[:lnzi+1]
    
    first = 0
    last = first + patch_in_row
    
    while last < lnzi+1:
        data = contData[first:last]
        frame = np.zeros((image_width, image_height))
        
        for dataIdx in range(data.shape[0]):
            frame[:, dataIdx*6: (dataIdx+1)*6]  = data[dataIdx]
        
        frames.append(frame)
        
        midIndex = (first+last)//2
        
        if cdl.size == 2:
            if  midIndex < cdi[0]:
                clss = cdl[0]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
            elif midIndex < cdi[1]:
                frm_clss.append(0)
            else:
                clss = cdl[1]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
                
                
        elif cdl.size == 3:
            if midIndex < cdi[1]:
                clss = cdl[0]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
            elif midIndex < cdi[2]:
                frm_clss.append(0)
                
            elif midIndex < cdi[3]:
                clss = cdl[1]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
            elif midIndex < cdi[4]:
                frm_clss.append(0)
                
            else:
                clss = cdl[2]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
                
                
        elif cdl.size == 4:
            if midIndex < cdi[1]:
                clss = cdl[0]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
            elif midIndex < cdi[2]:
                frm_clss.append(0)
                
            elif midIndex < cdi[3]:
                clss = cdl[1]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
            elif midIndex < cdi[4]:
                frm_clss.append(0)
            
            elif midIndex < cdi[5]:
                clss = cdl[2]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
            elif midIndex < cdi[6]:
                frm_clss.append(0)
            else:
                clss = cdl[3]
                frm_clss.append(np.where(classes==clss)[0][0]+1)
            
        first += stride
        last += stride
        
    return frames, frm_clss

def make_train_ds_patches(ds, dl, di, stride):
    
        images = []
        sam_clss = []
        for i in range(ds.shape[0]):
            frames, frm_classes = get_train_data_patches(ds[i], dl[i], di[i], stride)
            
            for [f, c] in zip(frames, frm_classes):
                images.append(f)
                sam_clss.append(c)
                
        return images, sam_clss



    
def make_test_ds_patches(ds, stride):
    
        frames = []
        frm_cntr = []
        
        for i in range(ds.shape[0]):
            curData = ds[i]
        
            lnzi = lastNoneZeroIndex_x_6_6(curData)
        
            curData = curData[:lnzi+1]
        
            first = 0
            last = first + patch_in_row
            
            counter = 0
            while last < lnzi+1:
                data = curData[first:last]
                frame = np.zeros((image_width, image_height))
            
                for dataIdx in range(data.shape[0]):
                    frame[:, dataIdx*6: (dataIdx+1)*6]  = data[dataIdx]
                
                first += stride
                last += stride
                counter += 1
                frames.append(frame)
            frm_cntr.append(counter)
            
        return frames, frm_cntr

def WER(reference, hypothesis) -> float:
    
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



def get_cat(clss):
        cat1 = [1,2,3,4,5,6,7,8,9]
        cat2 = [10,11,12,13,14,15,16,17,18,19]
        cat3 = [20,30,40,50,60,70]
        cat4 = [100,200,300,400,500,600,700]
        cat5 = [1000,2000,3000,4000,5000,6000,7000]
        
        if clss in cat1:
            return 1
        if clss in cat2:
            return 2
        if clss in cat3:
            return 3
        if clss in cat4:
            return 4
        if clss in cat5:
            return 5


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



classes = [1,2,3,4,5,6,7,8,9,
           10,11,12,13,14,15,16,17,18,19,
           20,30,40,50,60,70,
           100,200,300,400,500,600,700,
           1000,2000,3000,4000,5000,6000,7000]
target_classes = [1,2,3,4,5,6,7,8,9,
                  20,30,40,50,60,70,
                  100,200,300,400,500,600,700,
                  1000,2000,3000,4000,5000]


def removeUnwantedClasses():
    mask = np.all(np.isin(d2l, target_classes), axis=1)
    a1 = d2[mask]
    b1 = d2l[mask]
    c1 = d2i[mask]
    
    mask = np.all(np.isin(d3l, target_classes), axis=1)
    a2 = d3[mask]
    b2 = d3l[mask]
    c2 = d3i[mask]
    
    mask = np.all(np.isin(d4l, target_classes), axis=1)
    a3 = d4[mask]
    b3 = d4l[mask]
    c3 = d4i[mask]
    
 
    
    return a1,a2,a3, b1,b2,b3, c1,c2,c3


classes = np.array(classes)
num_classes = classes.shape[0]+1 # 1 is for Transition data

d2, d3, d4, d6, d2l, d3l, d4l, d6l, d2i, d3i, d4i =  load_dataset()

d2, d3, d4, d2l, d3l, d4l, d2i, d3i, d4i = removeUnwantedClasses()








for i in range(d6l.shape[0]):
    for j in range(d6l.shape[1]):
        d6l[i, j] = np.where(d6l[i,j]==classes)[0][0]+1

num_layers = 4
num_heads = 8
mlp_dim = 32

projection_dim = 64

patch_in_col = 1
patch_size = 6

werArray = []
for patch_in_row in [4,6,8,9, 10, 12,14,16,18]:#]:9,16,2524, 32, 36,40,44,5#6,8,10,12,14,16,18,20,22,25,30

        num_patch = patch_in_row * patch_in_col
        
        image_width  = patch_in_col * patch_size
        image_height = patch_in_row * patch_size
    
        
        # Create the Vision Transformer model
        vit_classifier = VisionTransformer(
            image_width = image_width,
            image_height = image_height,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            projection_dim=projection_dim,
        )
        
        # Compile the model
        vit_classifier.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        q = patch_in_row//2
        
        for x_step in [q, patch_in_row]: #q,,   q*3, , patch_in_row
            tr2, tr2c = make_train_ds_patches(d2, d2l, d2i, x_step)
            tr3, tr3c = make_train_ds_patches(d3, d3l, d3i, x_step)
            tr4, tr4c = make_train_ds_patches(d4, d4l, d4i, x_step)
            
            x_train = np.array(tr2+tr3+tr4)
            y_train = np.array(tr2c+tr3c+tr4c)
            
            x_train = np.expand_dims(x_train, axis=3)
            
            vit_classifier.fit(x_train, y_train, epochs=10)
            
            x_test, cntr = make_test_ds_patches(d6, x_step)
            
            x_test = np.array(x_test)
            x_test = np.expand_dims(x_test, axis=3)
            results = vit_classifier.predict(x_test)
            y_pred_single = np.argmax(results, axis=1)

            y_pred = []
            idx = 0
            for cc in range(len(cntr)):
                frame_number = cntr[cc]
                y_pred.append(y_pred_single[idx: idx+frame_number])
                idx += frame_number

            werall = 0
            
            for i1 in range(len(y_pred)):
                ref = majority(list(d6l[i1]))
                hyp = majority(list(y_pred[i1]))
                werall += WER(ref, hyp)
            
            werall /= len(y_pred)
            
            print("final WER: ", werall)
    
            werArray.append([patch_in_row, x_step, werall])





