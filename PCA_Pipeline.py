#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:27:35 2018

@author: aditya
"""



# Full Pipeline of Analysis for all Subjects, including graphs. All Data can be found in
# Subjects Variable: Including all bands and their scores


# Boxcar Filter = 1101 time points in all, so anything below that, 20 works pretty well for me
# Might want to try with 50, 100, 200
# Hilb_Type = Apply analysis on 'phase' , 'amp' = amplitude.


# So run this line of code for everything analyzed at once.
# Eg. Scores = Lang_Analysis(20, 'amp')
# This line can also be found at the bottom of the script so if you uncomment it 
# And save, you can just run the file, and it will do everything for you. 


# Run and change variables, processing methods, until all bands match up in graph
# With decent separation in accuracy. 


    
import numpy as np
import matplotlib.pyplot as plt

import mne
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
    



def Lang_Analysis(boxcar, hilb_type):
    

    
    
    Subject1 = Analysis('Subject 1', boxcar, hilb_type)
    Subject2 = Analysis('Subject 2', boxcar, hilb_type)
    Subject3 = Analysis('Subject 3', boxcar, hilb_type)
    Subject4 = Analysis('Subject 4', boxcar, hilb_type)
    
    LangAnalysis = Subject1, Subject2, Subject3, Subject4
    
    postAnalysis(LangAnalysis)
    
    return(LangAnalysis)




#Full Pipeline of Analysis for 1 Subject for all frequency bands. 


def Analysis(Subject, boxcar, hilb_type):
    Alpha, Labels = FilterFreq(8,12, Subject)
    L1,R1 = PCA_score(Alpha, Labels, boxcar, hilb_type)
    del(Alpha, Labels)
    Beta, Labels = FilterFreq(13,25, Subject)
    L2,R2 = PCA_score(Beta, Labels, boxcar, hilb_type)
    del(Beta, Labels)
    Gamma, Labels = FilterFreq(30,45, Subject)
    L3,R3 =  PCA_score(Gamma, Labels, boxcar, hilb_type)
    del(Gamma, Labels)
    Theta, Labels = FilterFreq(4,7, Subject)
    L4,R4 = PCA_score(Theta, Labels, boxcar, hilb_type)
    del(Theta,Labels)
    AllData, Labels = FilterFreq(2,50, Subject)
    L5,R5 = PCA_score(AllData,Labels, boxcar, hilb_type)
    del(AllData,Labels)
    scores = {'Alpha' : (L1,R1), 'Beta': (L2,R2), 'Gamma' : (L3,R3), 'Theta': (L4,R4), 'AllData' : (L5, R5)}
    print(Subject)
    return(scores)





# Finds band with most difference between hemispheres (no bias) and ouputs that as the 
# Language Dominant hemisphere with a graph, for all subjects.       
         
def postAnalysis(LangAnalysis):

    # Compare averages between hemispheres to find best band
    def scoreAnalysis(scores):
        old = 0
        for k,v in scores.items():
           compare = (np.abs((np.mean(v[0]) - np.mean(v[1]))), k)
           #If needed do not allow, 2-50 band, and only focus on freq bands. 
           if compare[0] > old and k != 'AllData':
               band = k
               old = compare[0]
               array = v
        
        if np.mean(array[0])>np.mean(array[1]):
            side = 'Left'
        else:
            side = 'Right'
        print(band, old, side)
        return(band,array, old, side)
    
    # 4 Graphs for each Subject with best band
    f, axes = plt.subplots(2,2)
    axes[0,0].set_title('Subject 1')    
    k,v, old, side = scoreAnalysis(LangAnalysis[0])
    axes[0,0].plot(v[0])
    axes[0,0].plot(v[1])
    axes[0,0].text(0.68, 0.7, k)
    
    axes[0,1].set_title('Subject 2')
    k,v, old, side = scoreAnalysis(LangAnalysis[1])
    axes[0,1].plot(v[0])
    axes[0,1].plot(v[1])
    axes[0,1].text(0.5, 0.8, k)
    axes[0,1].legend(('Left', 'Right'),
           loc='upper right')
    
    axes[1,0].set_title('Subject 3')
    k,v, old, side = scoreAnalysis(LangAnalysis[2])
    axes[1,0].plot(v[0])
    axes[1,0].plot(v[1])
    axes[1,0].text(0.5, 0.5, k)
    
    axes[1,1].set_title('Subject 4')
    k,v, old, side = scoreAnalysis(LangAnalysis[3])
    axes[1,1].plot(v[0])
    axes[1,1].plot(v[1])
    axes[1,1].text(0.8, 0.75, k)
    
    plt.show()
    
    g = plt.figure(2)
    plt.bar()
    
    return()
    

    
    

# Filter program, type in subject, and what frequencies you want, and will give you
# Epoched data and its Labels in Data variable and Labels Variable

"""
         x   y
Gamma -> 30,45
Beta -> 13, 25
Alpha  -> 8,12
Theta -> 4, 7
All -> 2 , 50

"""
# Filter bands for your convenience 

def FilterFreq(Low,High, Subject):
    #Run PCA Pipeline
    
    
    rawLH, eventsLH, infoRaw = initialize(Low, High, 'left', Subject)
    
    
    rawRH, eventsRH, infoRaw = initialize(Low, High, 'right', Subject)
    
    PicksR = picks(rawRH)
    PicksL = picks(rawLH)
    
    #Apply Hilbert
    HilbR = Hilbert(rawRH, PicksR)
    HilbL = Hilbert(rawLH, PicksL)
    
    #Obtain Data
    R_Data = Epochs(HilbR, eventsRH, rawRH, PicksR, infoRaw)
    L_Data = Epochs(HilbL, eventsLH, rawLH, PicksL, infoRaw)
    
    #Data is your data, and the events are your labels 
    
    Data = L_Data, R_Data
    Labels = eventsRH, eventsLH
    
    return(Data, Labels)






### Definition of Data Structure : How does it store everything ->
# Data Structures - Left Hemisphere Data, Right Hemisphere Data
# Hemisphere Data - AEFL Data, Word Data
# AEFL and WORD Data contain - Hilb Amp, Hilb Phase
# Eg. Alpha[0][0][0] -> Left Tone Hilb AMP
#     Alpha[1][0][0] -> Right Tone Hilb Amp
#     Alpha[0][0][1] -> Left Tone Hilb Phase
#     Alpha[0][1][0] -> Left Word Hilb Amp
#     Alpha[0][1][1] -> Left Word Hilb Phase
#     Alpha[1][1][0] -> Right Word Hilb Amp


# apply PCA and SVM here, hilb_type allows you to choose if you want to work with 
# amplitude or phase


def PCA_score(Beta, Labels, boxcar, hilb_type):
    
    

    
    labelsR = np.concatenate((Labels[0][0], Labels[0][1]))
    labelsL = np.concatenate((Labels[1][0], Labels[1][1]))
    def Boxcar(data, N):
    
        fdata = np.zeros((data.shape))
        for i in range(data.shape[0]):
            for q in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if k<N:
                        fdata[i,q,k] = data[i,q,k]
                    else:
#                        fdata[i,q,k] = np.sqrt(np.mean(data[i,q,(k-(N-1)):(k+1)] ** 2))
                      fdata[i,q,k] = (np.mean(data[i,q,(k-(N-1)):(k+1)]))

                        # depending on if you want to use RMS vs just mean, 
                        #try out with both, see what works better for you
                        
        return(fdata)
    
    
       
    from mne.decoding import UnsupervisedSpatialFilter
    from sklearn.decomposition import PCA
    #of note, here PCA also allows an unsupervised Spatial Filter
    pcaR = UnsupervisedSpatialFilter(PCA(3), average=False)
    
    if hilb_type == 'amp':
        
        pca_data_AEFL = pcaR.fit_transform(Beta[0][0][0])
        pca_data_WORD = pcaR.fit_transform(Beta[0][1][0])
        pca_L = np.concatenate((pca_data_AEFL, pca_data_WORD))
        pca_data_AEFR = pcaR.fit_transform(Beta[1][0][0])
        pca_data_WORD_R = pcaR.fit_transform(Beta[1][1][0])
        pca_RR = np.concatenate((pca_data_AEFR, pca_data_WORD_R))
    
    if hilb_type == 'phase':
        
        pca_data_AEFL = pcaR.fit_transform(Beta[0][0][1])
        pca_data_WORD = pcaR.fit_transform(Beta[0][1][1])
        pca_L = np.concatenate((pca_data_AEFL, pca_data_WORD))
        pca_data_AEFR = pcaR.fit_transform(Beta[1][0][1])
        pca_data_WORD_R = pcaR.fit_transform(Beta[1][1][1])
        pca_RR = np.concatenate((pca_data_AEFR, pca_data_WORD_R))
    
    
    
    
    
    labelsR = labelsR[0:len(pca_RR)]
    labelsL = labelsL[0:len(pca_L)]
    labels_data = labelsL, labelsR
    
    
    # Apply Boxcar Filter - based on boxcar length specific in upper level functions
    pca_RR = Boxcar(pca_RR, boxcar)
    pca_L = Boxcar(pca_L, boxcar)
    
    
    
#        return(pca_L, pca_RR, labels_data)
#    
#    def SVM(pca_L, pca_RR, labels_data):
    

    from sklearn.svm import SVC  # noqa
    from sklearn.model_selection import ShuffleSplit  # noqa
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    # Applies CSP before applying SVC for better results, point of investigation for 
    # obtaining better accuracies 
    from mne.decoding import CSP  # noqa
    clf = make_pipeline(CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False),
                        SVC(C=1, kernel = 'linear'))



# If you want to apply a LogisticRegression instead, this is the code for it
#
#    from sklearn.preprocessing import StandardScaler
#    from mne.decoding import (SlidingEstimator, cross_val_multiscore)
#    from sklearn.pipeline import make_pipeline
#    from sklearn.linear_model import LogisticRegression

#    
#    
#    clf = make_pipeline(StandardScaler(), LogisticRegression())
#    time_decod = SlidingEstimator(clf, n_jobs=1, scoring='accuracy')
#    L_score = cross_val_multiscore(time_decod, pca_L, labels_data[0][:,-1], cv=cv, n_jobs=1)
#    R_score = cross_val_multiscore(time_decod, pca_RR, labels_data[1][:,-1], cv=cv, n_jobs=1)


    
    # Use cross validation based on shuflesplit, to split into training and testing - Random
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    
    L_score = cross_val_score(clf, pca_L, labels_data[0][:,-1], cv=cv)
    R_score = cross_val_score(clf, pca_RR, labels_data[1][:,-1], cv=cv)



    print('Left Hemisphere Score', np.mean(L_score))
    print('Right Hemisphere Score', np.mean(R_score))
    
    return(L_score, R_score)

#def Boxcar(data, N):
#    
#    fdata = np.zeros((data.shape))
#    for i in range(data.shape[0]):
#        for q in range(data.shape[1]):
#            for k in range(data.shape[2]):
#                if k<N:
#                    fdata[i,q,k] = data[i,q,k]
#                else:
#                    fdata[i,q,k] = np.mean(data[i,q,(k-(N-1)):(k+1)])
#                    
#    return(fdata)
#                    
                




# Main Pipeline - This is for how to construct epochs, apply Hilbert, 
# Filtering of Raw data also involved here. 

def initialize(filterA, filterB, side, Subject): 
    

    data_path = '/Users/aditya/desktop/C001/'
    
    if Subject == 'Subject 1':
        raw_fnameAEF = data_path + 'C001_aef_raw_tsss.fif'
        raw_fnameword = data_path + 'C001_aword_raw_tsss.fif'
    
        raw = mne.io.read_raw_fif(raw_fnameAEF, preload = True)  
    
        rawWORD = mne.io.read_raw_fif(raw_fnameword, preload = True)
    
    if Subject == 'Subject 2':
        raw_fnameAEF = data_path + 'C002_aef_raw_tsss.fif'
        raw_fnameword = data_path + 'C002_aword_raw_tsss.fif'
    
        raw = mne.io.read_raw_fif(raw_fnameAEF, preload = True)  
    
        rawWORD = mne.io.read_raw_fif(raw_fnameword, preload = True)
    
    if Subject == 'Subject 3':
        raw_fnameAEF = data_path + 'C003_aef_raw_tsss.fif'
        raw_fnameword = data_path + 'C003_aword_raw_tsss.fif'
    
        raw = mne.io.read_raw_fif(raw_fnameAEF, preload = True)  
    
        rawWORD = mne.io.read_raw_fif(raw_fnameword, preload = True)    
        rawWORD.drop_channels(ch_names = ('STI001','STI002','STI003','STI004','STI005','STI006','STI007','STI008'))
    
    if Subject == 'Subject 4':
        raw_fnameAEF = data_path + 'C004_aef_raw_tsss.fif'
        raw_fnameword = data_path + 'C004_aword_raw_tsss.fif'
    
        raw = mne.io.read_raw_fif(raw_fnameAEF, preload = True)  
    
        rawWORD = mne.io.read_raw_fif(raw_fnameword, preload = True)
    
    """ 
    intialize arrays
    
    """
    
    rawWORDL = rawWORD
    
    rawAEFL = raw             
    
    x = filterA
    y = filterB
    

    
    #event_id = {'word': 1, 'tone': 0,}
    
    """
    Set up hemispheres
    """
    if side == 'right':
        
    #Right
        rawAEFL.info['bads'] = ['MEG0111', 'MEG0112','MEG0113', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331', 'MEG0332', 'MEG0333', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0611', 'MEG0612', 'MEG0613', 'MEG0621', 'MEG0622', 'MEG0623', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG0641', 'MEG0642', 'MEG0643', 'MEG0711', 'MEG0712', 'MEG0713',  'MEG0741', 'MEG0742', 'MEG0743', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823',  'MEG1011', 'MEG1012', 'MEG1013', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722', 'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933', 'MEG1941', 'MEG1942', 'MEG1943', 'MEG2011', 'MEG2012', 'MEG2013', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 'MEG2141', 'MEG2142', 'MEG2143'] 
        rawWORDL.info['bads'] = ['MEG0111', 'MEG0112','MEG0113', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331', 'MEG0332', 'MEG0333', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0611', 'MEG0612', 'MEG0613', 'MEG0621', 'MEG0622', 'MEG0623', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG0641', 'MEG0642', 'MEG0643', 'MEG0711', 'MEG0712', 'MEG0713',  'MEG0741', 'MEG0742', 'MEG0743', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823',  'MEG1011', 'MEG1012', 'MEG1013', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722', 'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933', 'MEG1941', 'MEG1942', 'MEG1943', 'MEG2011', 'MEG2012', 'MEG2013', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 'MEG2141', 'MEG2142', 'MEG2143'] 

    else:
        
    #Left
        rawAEFL.info['bads'] = ['MEG1922', 'MEG2342', 'MEG0621' , 'MEG0622', 'MEG0623', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG1141', 'MEG1142', 'MEG1143', 'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1411', 'MEG1412', 'MEG1413', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2311', 'MEG2312', 'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341', 'MEG2342', 'MEG2343', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421','MEG2422', 'MEG2423', 'MEG2431', 'MEG2432', 'MEG2433', 'MEG2441', 'MEG2442', 'MEG2443', 'MEG2511', 'MEG2512', 'MEG2513' ,'MEG2521', 'MEG2522', 'MEG2523', 'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643']  
        rawWORDL.info['bads'] = ['MEG1922', 'MEG2342', 'MEG0621' , 'MEG0622', 'MEG0623', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG1141', 'MEG1142', 'MEG1143', 'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1411', 'MEG1412', 'MEG1413', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2311', 'MEG2312', 'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341', 'MEG2342', 'MEG2343', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421','MEG2422', 'MEG2423', 'MEG2431', 'MEG2432', 'MEG2433', 'MEG2441', 'MEG2442', 'MEG2443', 'MEG2511', 'MEG2512', 'MEG2513' ,'MEG2521', 'MEG2522', 'MEG2523', 'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643']  



    #from mne.preprocessing import compute_proj_ecg, compute_proj_eog
    ##projs, events = compute_proj_ecg(rawAEFL, n_grad=1, n_mag=1, n_eeg=0, average=True)
    ##ecg_projs = projs[-2:]
    #projs, events = compute_proj_eog(rawAEFL, n_grad=1, n_mag=1, n_eeg=1, average=True)
    #eog_projs = projs[-3:]
    #rawAEFL.info['projs'] += eog_projs 
    #
    ##projs, events = compute_proj_ecg(rawAEFR, n_grad=1, n_mag=1, n_eeg=0, average=True)
    ##ecg_projs = projs[-2:]
    #projs, events = compute_proj_eog(rawAEFR, n_grad=1, n_mag=1, n_eeg=1, average=True)
    #eog_projs = projs[-3:]
    #rawAEFR.info['projs'] += eog_projs 
    
    
    """
    Create Events
    """
    eventsWORDL = mne.find_events(rawWORDL, stim_channel='STI101')
    eventsAEFL = mne.find_events(rawAEFL, stim_channel='STI101') #joined processes
    eventsWORDL[:, 2] = 0   
    
    
    
    
    
    """
    Create vectors
    """
    
    
    #matrixL = np.zeros([331,48,23,4])
    #matrixR = np.zeros([331,45,23,4])             
    #alphaR = np.zeros([327,45,23])
    #betaL = np.zeros([327,48,23])
    #betaR = np.zeros([327,45,23])
    #gammaR = np.zeros([327,45,23])
    #gammaL = np.zeros([327,48,23])
    #thetaR = np.zeros([327,45,23])
    #thetaL = np.zeros([327,48,23])
    
    """
    Filter
    Created on Thu Mar  1 02:39:43 2018
    
    Filter Function
    
             x   y
    Gamma -> 30,45
    Beta -> 13, 25
    Alpha  -> 8,12
    Theta -> 4, 7
    All -> 2 , 50
    
    @author: aditya
    """
    
    
    rawAEFL.filter(x, y, n_jobs=1,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1,  # in each band and skip "auto" option.
                   fir_design='firwin')
    
    rawWORDL.filter(x, y, n_jobs=1,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1,  # in each band and skip "auto" option.
                   fir_design='firwin')
    
    eventsLH = eventsAEFL, eventsWORDL
    rawLH = rawAEFL, rawWORDL
    
    tmin, tmax = -0.1, 1
    event_idAEF = {'tone': 1}
    event_idWORD = {'word': 0}
    event_id = event_idAEF, event_idWORD
    infoRaw = tmin, tmax, event_id
    
    return(rawLH, eventsLH, infoRaw)









def picks(rawLH):
        
    AEFL_picks = mne.pick_types(rawLH[0].info, meg= 'grad', eeg=False, stim=False, eog=False,
                           exclude='bads')	
    
    WORDL_picks = mne.pick_types(rawLH[1].info, meg= 'grad', eeg=False, stim=False, eog=False,
                           exclude='bads')	
    
    Picks = AEFL_picks, WORDL_picks
    
    return(Picks)




def Hilbert(rawLH, Picks):
    """
    Apply Hilbert on Tone
    """
    
    
    
    hilbAEFL = rawLH[0].apply_hilbert(Picks[0])
    print(hilbAEFL[0][0].dtype)
    
    hilbAEFL_amp = hilbAEFL.copy()
    hilbAEFL_amp.apply_function(np.abs, Picks[0])
    
    hilbAEFL_phase = hilbAEFL.copy()
    hilbAEFL_phase.apply_function(np.angle, Picks[0])
    
    """
    Apply Hilbert on Word
    """
    hilbWORDL = rawLH[1].apply_hilbert(Picks[1])
    print(hilbWORDL[0][0].dtype)
    
    hilbWORDL_amp = hilbWORDL.copy()
    hilbWORDL_amp.apply_function(np.abs, Picks[1])
    
    hilbWORDL_phase = hilbWORDL.copy()
    hilbWORDL_phase.apply_function(np.angle, Picks[1])
    
    HilbAEFL = hilbAEFL, hilbAEFL_amp, hilbAEFL_phase
    HilbWORDL = hilbWORDL, hilbWORDL_amp, hilbWORDL_phase
    HilbL = HilbAEFL,  HilbWORDL
    return(HilbL)


def Epochs(HilbL, eventsLH, rawLH, Picks, infoRaw):
    
    """
    Create Epochs of both datasets
    """
    
       
    epochsAEFL = mne.Epochs(HilbL[0][1], eventsLH[0], infoRaw[2][0], infoRaw[0], infoRaw[1], proj=False,
                        picks=Picks[0], baseline=None, preload=True,
                        verbose=False)
    
    epochsAEFL_phase = mne.Epochs(HilbL[0][2], eventsLH[0], infoRaw[2][0], infoRaw[0], infoRaw[1], proj=False,
                        picks=Picks[0], baseline=None, preload=True,
                        verbose=False)
    
    
    epochsWORDL = mne.Epochs(HilbL[1][1], eventsLH[1], infoRaw[2][1], infoRaw[0], infoRaw[1], proj=False,
                        picks=Picks[1], baseline=None, preload=True,
                        verbose=False)
    
    
    epochsWORDL_phase = mne.Epochs(HilbL[1][2], eventsLH[1], infoRaw[2][1], infoRaw[0], infoRaw[1], proj=False,
                        picks=Picks[1], baseline=None, preload=True,
                        verbose=False)
    
    AEFL_X = epochsAEFL.get_data()
    
    AEFL_X_phase = epochsAEFL_phase.get_data()
    
    WORDL_X = epochsWORDL.get_data()
    
    WORDL_X_phase = epochsWORDL_phase.get_data()
    
    AEFLData = AEFL_X, AEFL_X_phase
    WORDLData = WORDL_X, WORDL_X_phase
    L_Data = AEFLData, WORDLData
    
    return(L_Data)



Scores = Lang_Analysis(20, 'amp')
























  
    






