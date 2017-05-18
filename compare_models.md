MNIST Data: Random Forest vs XGBOOST vs Deep Neural Network
================
Amandeep Rathee
18 May, 2017

-   [Introduction](#introduction)
-   [Overview of notebook](#overview-of-notebook)
-   [Load data](#load-data)
-   [Predictive Modelling](#predictive-modelling)
    -   [Split data](#split-data)
    -   [Random Forest](#random-forest)
    -   [XGBoost](#xgboost)
    -   [Deep Neural Network using H2O](#deep-neural-network-using-h2o)
-   [Result](#result)
-   [Closing Remarks](#closing-remarks)

------------------------------------------------------------------------

Introduction
------------

There was a time when *random forest* was the coolest machine learning algorithm on machine learning competition platforms like **Kaggle** . But things changed and a better version of *gradient boosted trees* came along, with the name *XGBOOST*. The trend seems to continue and *deep learning* methods are replacing XGBOOST especially in competitions where image processing is involved. People are using deep nets recurrently in the following Kaggle competitions:

-   [NOAA Fisheries Steller Sea Lion Population Count](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count)
-   [Invasive Species Monitoring](https://www.kaggle.com/c/invasive-species-monitoring)
-   [Google Cloud & YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m)

Overview of notebook
--------------------

This notebook compares random forest, XGBOOSt and a simple deep neural network build using the H2O package in R. Since the data takes a lot of time to train, only 10% of the data is going to be used here. I'll share how the algorithms performed on full data at the end of the notebook.

------------------------------------------------------------------------

Load data
---------

``` r
digit <- fread("train.csv")
```

    ## 
    Read 71.4% of 42000 rows
    Read 42000 rows and 785 (of 785) columns from 0.072 GB file in 00:00:03

Let us see the distribution of **LABEL** variable.

``` r
prop.table(table(digit$label))*100
```

    ## 
    ##         0         1         2         3         4         5         6 
    ##  9.838095 11.152381  9.945238 10.359524  9.695238  9.035714  9.850000 
    ##         7         8         9 
    ## 10.478571  9.673810  9.971429

Retain only 10% of the data to make things faster.

``` r
digit <- digit[sample(1:nrow(digit), 0.1*nrow(digit), replace = FALSE), ]
```

Predictive Modelling
--------------------

### Split data

Split data in ratio of **80:20**. 80% is for train and the remaining 20% is to test the algorithms' performance.

``` r
digit$label <- factor(digit$label)
set.seed(1234)
split <- sample.split(digit$label, SplitRatio = 0.8)
train <- subset(digit, split == T)
cv <- subset(digit, split == F)
```

------------------------------------------------------------------------

### Random Forest

``` r
set.seed(4)
rf.model <- randomForest(label ~ ., data = train, ntree = 100, nodesize = 50)
rf.predict <- predict(rf.model, cv)
print(rf.cm <- confusionMatrix(rf.predict, cv$label))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1  2  3  4  5  6  7  8  9
    ##          0 77  0  3  2  0  1  1  1  0  1
    ##          1  0 94  0  0  0  1  1  1  1  0
    ##          2  0  0 77  0  2  1  5  0  3  2
    ##          3  0  0  2 74  0  5  0  0  5  0
    ##          4  0  0  3  1 71  1  0  2  1  2
    ##          5  1  0  0  4  0 58  1  0  5  0
    ##          6  1  1  3  1  2  2 73  0  0  0
    ##          7  0  0  1  3  1  0  0 74  0  2
    ##          8  2  0  2  1  1  1  0  2 63  0
    ##          9  0  0  1  2  3  3  0  8  2 77
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.8765         
    ##                  95% CI : (0.8523, 0.898)
    ##     No Information Rate : 0.1128         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.8627         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity           0.95062   0.9895  0.83696  0.84091  0.88750  0.79452
    ## Specificity           0.98817   0.9946  0.98267  0.98408  0.98688  0.98570
    ## Pos Pred Value        0.89535   0.9592  0.85556  0.86047  0.87654  0.84058
    ## Neg Pred Value        0.99471   0.9987  0.98005  0.98148  0.98817  0.98060
    ## Prevalence            0.09620   0.1128  0.10926  0.10451  0.09501  0.08670
    ## Detection Rate        0.09145   0.1116  0.09145  0.08789  0.08432  0.06888
    ## Detection Prevalence  0.10214   0.1164  0.10689  0.10214  0.09620  0.08195
    ## Balanced Accuracy     0.96940   0.9921  0.90981  0.91250  0.93719  0.89011
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity           0.90123  0.84091  0.78750  0.91667
    ## Specificity           0.98686  0.99072  0.98819  0.97493
    ## Pos Pred Value        0.87952  0.91358  0.87500  0.80208
    ## Neg Pred Value        0.98946  0.98160  0.97792  0.99062
    ## Prevalence            0.09620  0.10451  0.09501  0.09976
    ## Detection Rate        0.08670  0.08789  0.07482  0.09145
    ## Detection Prevalence  0.09857  0.09620  0.08551  0.11401
    ## Balanced Accuracy     0.94405  0.91581  0.88784  0.94580

``` r
print(paste("Accuracy of Random Forest:", round(rf.cm$overall[1], 4)))
```

    ## [1] "Accuracy of Random Forest: 0.8765"

------------------------------------------------------------------------

### XGBoost

``` r
# convert every variable to numeric, even the integer variables
train <- as.data.frame(lapply(train, as.numeric))
cv <- as.data.frame(lapply(cv, as.numeric))

# convert data to xgboost format
data.train <- xgb.DMatrix(data = data.matrix(train[, 2:ncol(train)]), label = train$label)
data.cv <- xgb.DMatrix(data = data.matrix(cv[, 2:ncol(cv)]), label = cv$label)

watchlist <- list(train  = data.train, test = data.cv)

parameters <- list(
    # General Parameters
    booster            = "gbtree",          # default = "gbtree"
    silent             = 0,                 # default = 0
    # Booster Parameters
    eta                = 0.3,               # default = 0.3,
    gamma              = 0,                 # default = 0,
    max_depth          = 6,                 # default = 6,
    min_child_weight   = 1,                 # default = 1,
    subsample          = 1,                 # default = 1,
    colsample_bytree   = 1,                 # default = 1,
    colsample_bylevel  = 1,                 # default = 1,
    lambda             = 1,                 # default = 1
    alpha              = 0,                 # default = 0
    # Task Parameters
    objective          = "multi:softmax",   # default = "reg:linear"
    eval_metric        = "merror",
    num_class          = 10,
    seed               = 1234               # reproducability seed
    )

xgb.model <- xgb.train(parameters, data.train, nrounds = 50, watchlist)
```

    ## [1]  train-merror:0.134669   test-merror:0.166567 
    ## [2]  train-merror:0.087587   test-merror:0.116561 
    ## [3]  train-merror:0.070593   test-merror:0.102393 
    ## [4]  train-merror:0.060355   test-merror:0.093940 
    ## [5]  train-merror:0.052498   test-merror:0.087987 
    ## [6]  train-merror:0.047082   test-merror:0.082153 
    ## [7]  train-merror:0.042767   test-merror:0.077985 
    ## [8]  train-merror:0.038124   test-merror:0.074890 
    ## [9]  train-merror:0.033213   test-merror:0.070604 
    ## [10] train-merror:0.029315   test-merror:0.066675 
    ## [11] train-merror:0.026219   test-merror:0.063579 
    ## [12] train-merror:0.023452   test-merror:0.060960 
    ## [13] train-merror:0.021339   test-merror:0.058578 
    ## [14] train-merror:0.018690   test-merror:0.056078 
    ## [15] train-merror:0.016428   test-merror:0.052744 
    ## [16] train-merror:0.014672   test-merror:0.052744 
    ## [17] train-merror:0.013184   test-merror:0.050244 
    ## [18] train-merror:0.011994   test-merror:0.050006 
    ## [19] train-merror:0.010744   test-merror:0.048458 
    ## [20] train-merror:0.009553   test-merror:0.047387 
    ## [21] train-merror:0.008065   test-merror:0.046315 
    ## [22] train-merror:0.007172   test-merror:0.045363 
    ## [23] train-merror:0.006607   test-merror:0.043934 
    ## [24] train-merror:0.005625   test-merror:0.042743 
    ## [25] train-merror:0.004940   test-merror:0.042148 
    ## [26] train-merror:0.004286   test-merror:0.041672 
    ## [27] train-merror:0.003601   test-merror:0.041314 
    ## [28] train-merror:0.003125   test-merror:0.040957 
    ## [29] train-merror:0.002649   test-merror:0.040362 
    ## [30] train-merror:0.002321   test-merror:0.039886 
    ## [31] train-merror:0.002024   test-merror:0.039052 
    ## [32] train-merror:0.001577   test-merror:0.037981 
    ## [33] train-merror:0.001250   test-merror:0.038457 
    ## [34] train-merror:0.001071   test-merror:0.036909 
    ## [35] train-merror:0.000863   test-merror:0.035957 
    ## [36] train-merror:0.000833   test-merror:0.035957 
    ## [37] train-merror:0.000744   test-merror:0.035242 
    ## [38] train-merror:0.000476   test-merror:0.034052 
    ## [39] train-merror:0.000417   test-merror:0.034528 
    ## [40] train-merror:0.000327   test-merror:0.034409 
    ## [41] train-merror:0.000149   test-merror:0.033933 
    ## [42] train-merror:0.000089   test-merror:0.034052 
    ## [43] train-merror:0.000089   test-merror:0.033099 
    ## [44] train-merror:0.000089   test-merror:0.032980 
    ## [45] train-merror:0.000060   test-merror:0.032504 
    ## [46] train-merror:0.000060   test-merror:0.031790 
    ## [47] train-merror:0.000060   test-merror:0.031909 
    ## [48] train-merror:0.000060   test-merror:0.031670 
    ## [49] train-merror:0.000060   test-merror:0.031670 
    ## [50] train-merror:0.000030   test-merror:0.032028

``` r
xgb.predict <- predict(xgb.model, data.cv)
print(xgb.cm <- confusionMatrix(xgb.predict, cv$label))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1   2   3   4   5   6   7   8   9
    ##          0 810   0   3   1   0   3   2   1   3   4
    ##          1   0 925   3   2   1   2   0   6   5   1
    ##          2   0   5 803   7   1   4   1   9   3   3
    ##          3   0   1   6 833   0   8   0   3   2   4
    ##          4   3   4   1   0 790   5   2   5   2  11
    ##          5   0   0   0   8   0 727   8   0   7   3
    ##          6   4   0   0   0   5   4 811   0   2   0
    ##          7   1   1  11   9   1   2   1 844   0   4
    ##          8   6   0   6   5   0   2   2   2 784   5
    ##          9   2   1   2   5  16   2   0  10   5 803
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.968          
    ##                  95% CI : (0.964, 0.9716)
    ##     No Information Rate : 0.1116         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9644         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity           0.98063   0.9872  0.96168  0.95747  0.97052  0.95784
    ## Specificity           0.99776   0.9973  0.99564  0.99681  0.99565  0.99660
    ## Pos Pred Value        0.97944   0.9788  0.96053  0.97200  0.95990  0.96547
    ## Neg Pred Value        0.99789   0.9984  0.99577  0.99509  0.99683  0.99581
    ## Prevalence            0.09835   0.1116  0.09942  0.10358  0.09692  0.09037
    ## Detection Rate        0.09644   0.1101  0.09561  0.09918  0.09406  0.08656
    ## Detection Prevalence  0.09846   0.1125  0.09954  0.10204  0.09799  0.08965
    ## Balanced Accuracy     0.98919   0.9923  0.97866  0.97714  0.98308  0.97722
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity           0.98065   0.9591  0.96433  0.95823
    ## Specificity           0.99802   0.9960  0.99631  0.99431
    ## Pos Pred Value        0.98184   0.9657  0.96552  0.94917
    ## Neg Pred Value        0.99789   0.9952  0.99618  0.99537
    ## Prevalence            0.09846   0.1048  0.09680  0.09977
    ## Detection Rate        0.09656   0.1005  0.09334  0.09561
    ## Detection Prevalence  0.09835   0.1041  0.09668  0.10073
    ## Balanced Accuracy     0.98934   0.9776  0.98032  0.97627

``` r
print(paste("Accuracy of XGBoost is:", round(xgb.cm$overall[1], 4)))
```

    ## [1] "Accuracy of XGBoost is: 0.968"

------------------------------------------------------------------------

### Deep Neural Network using H2O

``` r
h2o.train <- as.h2o(train)
```

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |=================================================================| 100%

``` r
h2o.cv <- as.h2o(cv)
```

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |=================================================================| 100%

``` r
h2o.model <- h2o.deeplearning(x = setdiff(names(train), c("label")),
                              y = "label",
                              training_frame = h2o.train,
                              standardize = TRUE,         # standardize data
                              hidden = c(100, 100),       # 2 layers of 100 nodes each
                              rate = 0.05,                # learning rate
                              epochs = 25,                # iterations/runs over data
                              seed = 1234                 # reproducability seed
                              )
```

    ## Warning in .h2o.startModelJob(algo, params, h2oRestApiVersion): Dropping constant columns: [pixel729, pixel644, pixel645, pixel448, pixel726, pixel727, pixel728, pixel560, pixel52, pixel760, pixel10, pixel54, pixel53, pixel168, pixel56, pixel169, pixel11, pixel55, pixel14, pixel57, pixel16, pixel15, pixel18, pixel17, pixel19, pixel754, pixel755, pixel756, pixel757, pixel758, pixel759, pixel83, pixel196, pixel82, pixel85, pixel671, pixel84, pixel111, pixel672, pixel112, pixel673, pixel476, pixel753, pixel392, pixel700, pixel701, pixel141, pixel780, pixel30, pixel781, pixel782, pixel420, pixel783, pixel31, pixel421, pixel224, pixel140, pixel699, pixel139, pixel8, pixel9, pixel6, pixel7, pixel4, pixel5, pixel2, pixel3, pixel0, pixel21, pixel1, pixel20, pixel23, pixel532, pixel730, pixel22, pixel731, pixel25, pixel24, pixel27, pixel26, pixel29, pixel28].
    ## rate cannot be specified if adaptive_rate is enabled..

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |==                                                               |   2%
      |                                                                       
      |===                                                              |   5%
      |                                                                       
      |=====                                                            |   7%
      |                                                                       
      |========                                                         |  12%
      |                                                                       
      |===========                                                      |  17%
      |                                                                       
      |=============                                                    |  20%
      |                                                                       
      |==============                                                   |  22%
      |                                                                       
      |==================                                               |  27%
      |                                                                       
      |===================                                              |  30%
      |                                                                       
      |=====================                                            |  32%
      |                                                                       
      |=======================                                          |  35%
      |                                                                       
      |========================                                         |  37%
      |                                                                       
      |==========================                                       |  40%
      |                                                                       
      |===========================                                      |  42%
      |                                                                       
      |=============================                                    |  45%
      |                                                                       
      |================================                                 |  49%
      |                                                                       
      |==================================                               |  52%
      |                                                                       
      |=====================================                            |  57%
      |                                                                       
      |=======================================                          |  59%
      |                                                                       
      |==========================================                       |  64%
      |                                                                       
      |=============================================                    |  69%
      |                                                                       
      |===============================================                  |  72%
      |                                                                       
      |================================================                 |  74%
      |                                                                       
      |=================================================================| 100%

``` r
h2o.predictions <- as.data.frame(h2o.predict(h2o.model, h2o.cv))
```

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |================                                                 |  25%
      |                                                                       
      |=================================================================| 100%

``` r
print(h2o.cm <- confusionMatrix(h2o.predictions$predict, cv$label))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1   2   3   4   5   6   7   8   9
    ##          0 805   0   1   2   3   5   3   1   1   3
    ##          1   0 927   3   0   1   1   2   4   6   2
    ##          2   3   1 809   5   3   1   1   6   2   2
    ##          3   0   3   3 826   0   5   0   4   5   5
    ##          4   2   3   2   0 783   1   6   6   4   6
    ##          5   2   0   1  16   1 715   3   0  10   2
    ##          6   8   0   1   0   1   6 809   0   3   0
    ##          7   1   1   7   9   2   3   0 843   4   8
    ##          8   4   1   8   7   3   9   3   1 775   4
    ##          9   1   1   0   5  17  13   0  15   3 806
    ## 
    ## Overall Statistics
    ##                                        
    ##                Accuracy : 0.9642       
    ##                  95% CI : (0.96, 0.968)
    ##     No Information Rate : 0.1116       
    ##     P-Value [Acc > NIR] : < 2.2e-16    
    ##                                        
    ##                   Kappa : 0.9602       
    ##  Mcnemar's Test P-Value : NA           
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity           0.97458   0.9893  0.96886  0.94943  0.96192  0.94203
    ## Specificity           0.99749   0.9975  0.99683  0.99668  0.99604  0.99542
    ## Pos Pred Value        0.97694   0.9799  0.97119  0.97062  0.96310  0.95333
    ## Neg Pred Value        0.99723   0.9987  0.99656  0.99417  0.99591  0.99425
    ## Prevalence            0.09835   0.1116  0.09942  0.10358  0.09692  0.09037
    ## Detection Rate        0.09584   0.1104  0.09632  0.09835  0.09323  0.08513
    ## Detection Prevalence  0.09811   0.1126  0.09918  0.10132  0.09680  0.08930
    ## Balanced Accuracy     0.98603   0.9934  0.98284  0.97305  0.97898  0.96872
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity           0.97823   0.9580  0.95326  0.96181
    ## Specificity           0.99749   0.9953  0.99473  0.99273
    ## Pos Pred Value        0.97705   0.9601  0.95092  0.93612
    ## Neg Pred Value        0.99762   0.9951  0.99499  0.99575
    ## Prevalence            0.09846   0.1048  0.09680  0.09977
    ## Detection Rate        0.09632   0.1004  0.09227  0.09596
    ## Detection Prevalence  0.09858   0.1045  0.09704  0.10251
    ## Balanced Accuracy     0.98786   0.9766  0.97399  0.97727

``` r
print(paste("Accuracy of Deep neural network is:", round(h2o.cm$overall[1], 4)))
```

    ## [1] "Accuracy of Deep neural network is: 0.9642"

------------------------------------------------------------------------

Result
------

``` r
print(paste("Accuracy of Random Forest:", round(rf.cm$overall[1], 4)))
```

    ## [1] "Accuracy of Random Forest: 0.8765"

``` r
print(paste("Accuracy of XGBoost is:", round(xgb.cm$overall[1], 4)))
```

    ## [1] "Accuracy of XGBoost is: 0.968"

``` r
print(paste("Accuracy of Deep neural network is:", round(h2o.cm$overall[1], 4)))
```

    ## [1] "Accuracy of Deep neural network is: 0.9642"

------------------------------------------------------------------------

While running the above algorithms on full data I got the following accuracies on the same 80:20 split with the following parameters:

------------------------------------------------------------------------

<table>
<colgroup>
<col width="17%" />
<col width="15%" />
<col width="66%" />
</colgroup>
<thead>
<tr class="header">
<th>Model</th>
<th>Accuracy</th>
<th>Parameters</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Random Forest</td>
<td>0.9490</td>
<td>trees = 2000, nodesize = 50</td>
</tr>
<tr class="even">
<td>XGBoost</td>
<td>0.9705</td>
<td>all default parameters, nrounds = 100</td>
</tr>
<tr class="odd">
<td>H2O Deep Net</td>
<td>0.9750</td>
<td>rate = 0.05, epochs = 50, hidden layer = [200, 200, 200]</td>
</tr>
</tbody>
</table>

------------------------------------------------------------------------

The difference between Random Forest and other 2 is significant in this case. XGBoost and Deep Neural Nets outperform it completely. But when it comes to XGBoost vs Deep Neural Networks, there is no significant difference. One reason for this might be the small amount of data taken into account while training the models. Deep neural networks need humongous amount of data to show their relevance.

Closing Remarks
---------------

-   I hope you liked the notebook and it provided some insights about the three popular algorithms.

-   I got my first **silver medal** a few days back and it was a great confidence booster for me. It is hard to believe but I am **ranked \#35** in the Kernels section on Kaggle now. Here's the link to my [Kaggle profile](https://www.kaggle.com/arathee2).
