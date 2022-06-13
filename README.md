# PricePredict
A Housing Price Prediction Model 
Abstract
Estimating cost is the foundation for any business and it proves extremely critical for planning a job’s sched- ule and budget. This paper explores predictions on hous- ing prices based on a given data set , it delves into the process of training models and generating predictions us- ing linear regression and other types of regressions. This project also serves the purpose of testing pre-processing, algorithms, and parameterization in order to find the best housing price estimator. The data set used in this research provides information on houses in a suburban area, and through the course of this research it under goes various attribute adjustments and processing and is fitted to vari- ous models that generate predictions on prices with mod- erately high accuracy.
1 Data Description
The data set used in this research was provided by a Kag- gle competition by the title of ”House Prices - Advanced Regression Techniques”, it is called The Ames Housing dataset and it was compiled by Dean De Cock for the pur- pose of education in data science. It’s a good alternative to the Boston Housing dataset as it is more modernized and expanded. The houses in this data set each posess 81 attributes ranging from the property’s sale price in dol- lars, its lot size in square feet all the way to the type of roofing used in that property. Amongst these 81 attributes, 38 of them were numerical attributes (i.e: ’OverallQual’: Overall material and finish quality) , and 43 categorical attributes such as the height of the basement which are or- dered from ”Excellent” to ”No Basement”. This data set
contains certain attributes with missing values, especially the numerical attributes, it also contains some ordinal at- tributes that have values ranging from ”Ex” which stands for ”Excellent” to ”Po” which stands for ”Poor” such as the ”ExterQual” attribute which evaluates the quality of the material on the exterior of the property. This data set also contains attributes for the location of each house which need to undergo moderate pre-processing in order to be later fitted to our models. The nature of these at- tributes and a good understanding of how to process their values is a very critical step in order to utilize the integrity of the data set and increase the accuracy of our models.
Table 1 shows the attributes used in this research which were classified into broader classes. The table also shows the type of values they contain.
Table 1: Table of Attributes used in this research
Attribute Class Example Type
sqfAtts ’1stFlrSF’ Numerical yearAtts YearBuilt’ Numerical qualitativeAtts ’OverallQual’ Ordinal
Housing Price Prediction Models Hamza Cherqaoui, Ihab El Kerdoudi
DePauw University Greencastle, IN 46135, U.S.A.
 1
locationAtts conditionAtts ExtAtts roomAtts RoofAtts
’Street’ ’Condition1’ ’Exterior1st’ ’TotRmsAbvGrd’ ’RoofStyle’
Nominal Nominal Nominal Nominal Nominal
Working with this data set has required the use of many pre-processing techniques and helped make the data more adequate to our model when fitting and predicting.
2 Experiment 2.1 Pre-Processing
1 def 2
3
4
5
6
7
8 9}
ordToNumForQual ( trainDF , testDF , col ) :
trainDF[col] = trainDF[col ].map(lambda v: 4 if v==”Ex” else v)
trainDF[col] = trainDF[col ].map(lambda v: 3 if v==”Gd” else v)
trainDF[col] = trainDF[col ].map(lambda v: 2 if v==”TA” else v)
trainDF[col] = trainDF[col ].map(lambda v: 1 if v==”Fa” else v)
trainDF[col] = trainDF[col ].map(lambda v: 0 if v==”Po” else v)
 As mentioned in the previous section, the data set at hand comprised many attributes with missing values, thus the first process that was implemented in this project is to handle this case of missing values especially in numerical attributes. We started this step of the pre-processing by retrieving all the attributes that have numeric values first using the available helper function getAttrsWithMissing- Values(), then out of these attributes we recovered those that contained missing values with another helper func- tion. In order to handle this issue, missing values were replaced with the mean for each attribute!
The next step in this process was to add some new attributes to our predictors list as part of our feature- engineering efforts. Using some of the square footage at- tributes we first created a new attribute named ’Prop- ertySF’ which adds both the square footage of the first floor and the Garage Living area. We then created an age attribute for each property by subtracting the ’Year- Built’ attribute by the current year, this new attribute was named ’YearsOld’. Another trivial attribute was the to- tal square footage of the property which adds values of square footage from the garage area, the total basement area, the first and second floor footage as well as the garage living area!
After adding new attributes to our predictors, we imple- mented functions to handle conversions from non-numeric values to numeric. As previously mentioned, attributes with ordinal values were converted to numerical by map- ping numbers in ascending order to each corresponding value of such attribute. For the attributes ’Condition 1’ and ’Condition 2’, we used a helper function that uses a dictionary to map certain values to their respective nu- meric representations and we followed the same proce- dure for Exterior attributes and Location attributes. Fig- ure 1 shows a snippet of the code used in this process for the qualitative attributes.
After converting our values to numerical values, we still needed to fill in missing values for this converted at- tributes, so we wrote a function to map the mode of the attribute for each missing value. In this case, we used this helper function for both location attributes and Exterior attributes.
Figure 1: This method, maps numerical values to each value of the ’qualitativeAtts’
Following this conversion from categorical to numer- ical values, we did one hot encoding for the ordinal at- tributes we converted and used the built-in getdummies function in pandas. This part of the pre-processing was especially intricate as some attributes when one-hot en- coded and included in our predictors, significantly re- duced our average cross-validation score or didn’t im- prove it at all, so we made sure to select only the attributes that improved said average.
2.2 Algorithms and Hyperparameteriza- tion
This research used numerous estimators and built off of a Linear Regression model to fit the data and compute an avergae score in the early stages. In order to come up with the best possible accuracy, five more experiments were im- plemented where various models were built and fit to the data set, we used these models in this respective order: an Elastic Net Regressor, a Bayesian Ridge Regressor, a Gra- dient Boosting Regressor, a Ridge Regressor and a Lasso Regressor. All of these models yielded different results at the end.
Elastic net is a type of regularized linear regression and it combines two popular penalties, specifically the L1 and L2 penalty functions.[1]. The objective function for this type of Regressor is:
min 1 ∥Xw−y∥2 +αρ∥w∥ + α(1−ρ)∥w∥2 w 2nsamples 2 1 2 2
   2

  Figure 2: Lasso and Elastic-Net Paths
Bayesian Ridge Regression is more robust to ill-posed problems, and we used it to generate an average CV score to compare to other models like Lasso which is ”a linear model that estimates sparse coefficients”, it is used over regression methods for a more accurate prediction by us- ing shrinkage.[3].Figure 2, shows Lasso and Elastic Net Paths![3]
We have also used a Linear Ridge regressor which yielded results close to those of the linear regression model. The most significant model that was used in this research is the Gradient Boosting Regressor, it yielded the highest CV score average because it uses an ensemble al- gorithm that produces multiple groups of models to con- verge to one best fitting model.
Our attempt at tuning our hyper parameters for the gra- dient boosting regression model didn’t yield increased re- sults, we believe partly due to a misimplementation when using hyper parameters like the learning rate and n esti- mators for the model. The ”tuned” gradient boosting re- gressor reported a CV average score lower than that of the regular one.
2.3 Results
After pre-processing the data and building various mod- els, the resulting predictions were very interesting and had a lot to say about the tuning of the models and how our pre-processing worked. On one hand the cross val-
Model
Gradient Boosting Bayesian Ridge Ridge
Linear Regressor Lasso
Elastic-Net
2.4 Analysis
CV Average Score
0.8982 0.8108 0.8075 0.8066 0.8066 0.8003
3
idation score obtained from our linear regression model was about 0.807 approximately, other models like Lasso, Ridge, and Elastic-Net had similar results that vary a lit- tle. The Bayesian Ridge slightly out performed these mod- els and yielded a score of 0.810. However, the Gradient Boosting Regression model had the best results out of all the other models used in this research having an average cross validation score of 0.898. Table 2, shows the aver- age cross validation scores for all six models we used in descending order.
Table 2: Table of Attributes used in this research
The results of this research vary greatly and give us an in- sight on how different models perform using the same data set. While other models like the linear regression one or the Lasso model had decent average scores, other mod- els like the Gradient Boosting regressor had impressive results. To better illustrate this difference, let’s compare how Lasso and Elastic-Net operate and compare them with how our highest scoring model woks. Lasso is usu- ally used for simple, sparse models with fewer parameters which is the opposite in our case with the The Ames Hous- ing data set. Elastic-Net is used for the same purpose, however it adds one more level of intricacy by adding regularization penalties to the loss function during train- ing. In addition, Elastic-net is useful when there are multi- ple attributes that are correlated with one another, in this case, Lasso is more likely to pick one of these attributes at random, while elastic-net is likely to pick both.[3]
In comparison to the other models, gradient boosting has a comparative advantage because boosting is a strong process that utilizes many models to improve earlier mod- els’ mistakes. This is the main reason why the Gradient

Boosting model greatly outperformed the rest of the mod- els, because it relies on ”the intuition that the best pos- sible next model, when combined with previous models, minimizes the overall prediction error.” [2]
3 Conclusion
The results of this project showed that data pre-processing is one of the main factors that produce an accurate model. When we used only numerical predictors at first, namely the square footage attributes, the model was not very accurate and was falling short of a 0.5 CV average score. But, processing categorical attributes like roofing attributes, and qualitative attributes improved the accu- racy of our models and helped us achieve greater results especially with the gradient boosting regression model. In conclusion, in order to produce a high accuracy model, each data attribute has to be closely scrutinized and pro- cessed in the right way, whether it be converted to a differ- ent type or one-hot encoded, paired with an adequate high scoring model for the data set. Hyper parameter optimiza- tion is also an excellent way to improve on a model’s ac- curacy.
References
[1] Jason Brownlee. How to develop elastic net regres- sion models in python. Machine Learning Mastery With Python Discover The Fastest Growing Platform For Professional Machine Learning With Step-By- Step Tutorials and End-To-End Projects, 2016.
[2] Jake Hoare. Gradient boosting explained – the coolest kid on the machine learning block.
[3] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duch-
esnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825– 2830, 2011.
4
