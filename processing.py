# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:24:47 2021

@author: Sérgio Novais, Christian Alves e Denise Ribeiro
"""
import numpy as np
import cv2

def find_threshold( data ):
    threshold = 0
    list_of_thresholds = []
    
    for image in data:
        image = np.array(image) # transforma cada imagem em uma matriz

        for y in image[0]:
            for x in image:
                
                threshold = image[y, x]
                
                if threshold < image[y+1, x]:
                    threshold = image[y+1, x]
                    
                elif threshold < image[y, x+1]:
                    threshold = image[y, x+1]
                
                elif threshold < image[y+1, x+1]:
                    threshold = image[y+1, x+1]
                    
        
        list_of_thresholds.append(threshold)
        
    return sum(list_of_thresholds)/len(list_of_thresholds)
    
    
def binarize( data, threshold ):
    binary_data = []
    for image in data:
        image = np.array(image) # transforma cada imagem em uma matriz

        for y in image[0]:
            for x in image:
                if image[y,x] < threshold:
                    image[y,x] = 0
                
                else:
                    image[y,x] = threshold
                    
        binary_data.append(image)
        
    return np.array(binary_data)


def canny(x, y):
    edges = []
    for i in range(len(x)):
        edges.append(cv2.Canny(x[i], 28, 28))
   
    del x
        
    return np.array(edges), y