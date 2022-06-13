import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV 


# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")

    #demonstrateHelpers(trainDF)

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)

    testGBRegressor(trainInput, trainOutput, predictors)
    testBayesianRegressor(trainInput, trainOutput, predictors)
    testLinearRidge(trainInput, trainOutput, predictors)
    doExperiment(trainInput, trainOutput, predictors)
    testLasso(trainInput, trainOutput, predictors)
    testElasticNetRegressor(trainInput, trainOutput, predictors)

    print("============= Tuning:")
    
    tuningGBR(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw06 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score (Linear Regressor):", cvMeanScore)
    
def testElasticNetRegressor(trainInput, trainOutput, predictors):
    #BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    #EXPLANATION: We tried to use a different type of model seeing that Elastic-net is useful
    #             when there are multiple features that are correlated with one another.
    
    regr = ElasticNet(random_state=0)
    cvMeanScore = model_selection.cross_val_score(regr, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score (Elastic-Net):", cvMeanScore)
    #END
    
def testBayesianRegressor(trainInput, trainOutput, predictors):
    #BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
    #EXPLANATION: We tried to use a different type of model seeing that 
    #               Bayesian Ridge Regression is more robust to ill-posed problems.
    
    clf = linear_model.BayesianRidge(compute_score=True)
    cvMeanScore = model_selection.cross_val_score(clf, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score (Bayesian Ridge):", cvMeanScore)
    #END

def testGBRegressor(trainInput, trainOutput, predictors):
    #BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    #EXPLANATION: Including a Gradient Boosting Regressor to implement hyperparameter tuning
    
    gbrt=GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1) 
    cvMeanScore = model_selection.cross_val_score(gbrt, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score (GradientBoostingRegressor):", cvMeanScore)
    #END
    
    
def testLinearRidge(trainInput, trainOutput, predictors):
    #BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    #EXPLANATION: Adding Ridge regression model to compare results with other models
    
    rdg = linear_model.Ridge(alpha=1.0)
    cvMeanScore = model_selection.cross_val_score(rdg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score (Ridge):", cvMeanScore)
    #END=

def testLasso(trainInput, trainOutput, predictors):
    #BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    #EXPLANATION: Adding Lasso regressor to compare results with other models
    
    lso = linear_model.Lasso(alpha=1.0)
    cvMeanScore = model_selection.cross_val_score(lso, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score (Lasso):", cvMeanScore)
    #END

'''
============================================
Tuning our Gradient Boosting Regressor 
============================================
'''

def tuningGBR(trainInput, trainOutput, predictors):
    
    tunedGBR = GradientBoostingRegressor(learning_rate=0.5, min_samples_split=500
                                         ,min_samples_leaf=50,max_depth=10,max_features='auto',
                                          subsample=0.9,random_state=10)
    param_test1 = {'n_estimators':range(20,81,10)}
    gsearch1 = GridSearchCV(estimator = tunedGBR,param_grid = param_test1,scoring='r2',n_jobs=-1, cv=10)
    gsearch1.fit(trainInput.loc[:, predictors],trainOutput)
    print("Tuned GBR Score:", gsearch1.best_score_)

    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    #Replacing NaN values in Attributes with missing values with Mean
    missingVals = getAttrsWithMissingValues(testDF)
    numericVals = getNumericAttrs(testDF)
    
    #getting attributes that have Numeric Values and missing Values
    missingNumericVals = missingVals.intersection(numericVals)
    
    for i in missingNumericVals:
        fillMissingWMean(trainDF,testDF,i)
    
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    sqfAtts = ['1stFlrSF', '2ndFlrSF','LowQualFinSF','GrLivArea','WoodDeckSF',
               'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                'LotArea','MasVnrArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF',
                'TotalBsmtSF', 'LowQualFinSF','EnclosedPorch','TotalSQF',
                '3SsnPorch', 'ScreenPorch','PoolArea','PropertySF']
    yearAtts = ['YearBuilt','YearRemodAdd', 'GarageYrBlt','YearsOld']
    qualitativeAtts = ['OverallQual','OverallCond','KitchenQual','HeatingQC',
                       'BsmtCond','BsmtQual','ExterCond','ExterQual','FireplaceQu',
                       'GarageQual','GarageCond','PoolQC']
    locationAtts = ['Street','Alley']
    conditionAtts = ['Condition1','Condition2']
    ExtAtts = ['Exterior1st','Exterior2nd']
    roomAtts = ['TotRmsAbvGrd','BsmtFullBath','BsmtHalfBath','TotRmsAbvGrd',
                'KitchenAbvGr','BedroomAbvGr',]
    RoofAtts = ['RoofStyle','RoofMatl']
        
    
    #=====================================================
    #Creating more Atrributes for feature engineering
    
    #Creating total square footage of a house
    trainDF['PropertySF'] = trainDF['1stFlrSF']+trainDF['GrLivArea']
    testDF['PropertySF'] = testDF['1stFlrSF']+testDF['GrLivArea']
    
    #Creating years old attribute
    trainDF['YearsOld'] = 2021-trainDF['YearBuilt']
    testDF['YearsOld'] = 2021-trainDF['YearBuilt']
    
    #Creating total Square footage of inside area
    trainDF['TotalSQF']= +trainDF['GarageArea']+ trainDF['TotalBsmtSF'] + trainDF['1stFlrSF'] + trainDF['2ndFlrSF'] + trainDF['GrLivArea'] 
    testDF['TotalSQF']= +testDF['GarageArea']+ testDF['TotalBsmtSF'] + testDF['1stFlrSF'] + testDF['2ndFlrSF'] + testDF['GrLivArea'] 
    
    
    predictors = sqfAtts + yearAtts + qualitativeAtts + conditionAtts + locationAtts + ExtAtts+ roomAtts + RoofAtts
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''

    #=====================================================
    #Converting Ordinal values to numerical values using helper function ordToNumForQualitative
    #getting attributes that have qualitative values
    listofOrdinalAtts = ['KitchenQual','HeatingQC','BsmtCond','BsmtQual',
                         'ExterCond','ExterQual','FireplaceQu','GarageQual',
                         'GarageCond','PoolQC']
    
    for i in listofOrdinalAtts:
        ordToNumForQualitative(trainInput, testInput, i)
        
    
    #Converting Ordinal Values to Numerical values for conditionAtts using helper function ordToNumForConditions 
    nomToOrdForConditions(trainInput, testInput)
    
    #Converting Nominal Values to Numerical values for locationAtts using helper function ordToNumForCLocationAtts
    nomToOrdForLocationAtts(trainInput,testInput)
    
    #Encoding Roofing attributes
    encodeRoofingAtts(trainInput,testInput)
    
    
    #filling NA values for locationAtts
    for i in locationAtts:
        fillMissing(trainInput,testInput,i)
    
    nomToOrdForExtAtts(trainInput, testInput)
    for i in ExtAtts:
        fillMissing(trainInput,testInput,i)
    
    #=====================================================
    #One Hot Encoding more Attributes with Nominal and Ordinal Values(Categorical Attributes)
    #getting attributes that have qualitative values
    categoricalAtts = ['MSSubClass','OverallQual','OverallCond']+listofOrdinalAtts
    
    for i in categoricalAtts:
        trainDF = pd.get_dummies(trainDF,columns=[i])
        testDF = pd.get_dummies(testDF,columns=[i])
    
    '''
    #Standardizing on Square footage attribute
    
    standardize(trainDF, testDF, '1stFlrSF')
    '''

    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    
    return trainInput, testInput, trainOutput, testIDs, predictors

#Converts ordinal values into numerical values for the attribute col
def ordToNumForQualitative(trainDF,testDF,col):	
    
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  4 if v=="Ex" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  3 if v=="Gd" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2 if v=="TA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1 if v=="Fa" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  0 if v=="Po" else v)

    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="None" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])

    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 0 if v=="Po" else v)
    
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=="None" else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(testDF.loc[:, col].mode().loc[0])

def ordToNumForMSZoning(trainDF,testDF,col):	
    
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  7 if v=="RM" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  6 if v=="RP" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  5 if v=="RL" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  4 if v=="RH" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  3 if v=="I" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2 if v=="FV" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1 if v=="C" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  0 if v=="A" else v)

    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="None" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])

    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  7 if v=="RM" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  6 if v=="RP" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  5 if v=="RL" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  4 if v=="RH" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  3 if v=="I" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2 if v=="FV" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1 if v=="C" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  0 if v=="A" else v)
    
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=="None" else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(testDF.loc[:, col].mode().loc[0])
    
def nomToOrdForLocationAtts(trainDF,testDF):	
    #BEGIN: from https://stackoverflow.com/questions/67527277/how-to-convert-nominal-data-to-numeric-in-python
    #EXPLANATION: Changes ordinal character data to numeric data with Pandas by creating a new 
    #             dictionary with the new values and uses the built-in apply function to 
    #             switch the values. We ended up using the replace function since we want to
    #             keep our attributes' names 
    
    
    streetDict = {'Street':{'Grvl':1,'Pave':0}}
    trainDF.replace(streetDict,inplace=True)
    testDF.replace(streetDict,inplace=True)
    
    alleyDict = {'Alley':{'Grvl':2,'Pave':1,'NA':0}}
    trainDF.replace(alleyDict,inplace=True)
    testDF.replace(alleyDict,inplace=True)
    
    #END
    
#Helper function to fill NA values left from nominal to ordinal conversion
def fillMissing(trainDF,testDF,col):
    
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="None" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=="None" else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(testDF.loc[:, col].mode().loc[0])

#Helper function to convert nominal values to numerical values for Condition 1 and 2 attributes
def nomToOrdForConditions(trainDF,testDF):	

    conditionOneDict = {'Condition1':{'Norm':8, 'Feedr':7, 'PosN':6,
                                      'Artery':5, 'RRAe':4, 'RRNn':3,
                                      'RRAn':2, 'PosA':1,'RRNe':0}}
    trainDF.replace(conditionOneDict,inplace=True)
    testDF.replace(conditionOneDict,inplace=True)
    
    conditionTwoDict = {'Condition2':{'Norm':7, 'Artery':6, 'RRNn':5,
                                      'Feedr':4, 'PosN':3, 'PosA':2, 
                                      'RRAn':1, 'RRAe':0}}
    trainDF.replace(conditionTwoDict,inplace=True)
    testDF.replace(conditionTwoDict,inplace=True)

def nomToOrdForExtAtts(trainDF,testDF):
    
    ExtOneDict = {'Exterior1st':{'VinylSd':14, 'MetalSd':13, 'Wd Sdng':12, 
                                 'HdBoard':11, 'BrkFace':10, 'WdShing':9,
                                 'CemntBd':8, 'Plywood':7, 'AsbShng':6, 
                                 'Stucco':5, 'BrkComm':4, 'AsphShn':3,
                                 'Stone':2, 'ImStucc':1, 'CBlock':0}}
    trainDF.replace(ExtOneDict,inplace=True)
    testDF.replace(ExtOneDict,inplace=True)
    
    ExtTwoDict = {'Exterior2nd':{'VinylSd':15, 'MetalSd':14, 'Wd Shng':13,
                                 'HdBoard':12, 'Plywood':11, 'Wd Sdng':10,
                                 'CmentBd':9, 'BrkFace':8, 'Stucco':7,
                                 'AsbShng':6, 'Brk Cmn':5, 'ImStucc':4,
                                 'AsphShn':3, 'Stone':2, 'Other':1, 'CBlock':0}}
    trainDF.replace(ExtTwoDict,inplace=True)
    testDF.replace(ExtTwoDict,inplace=True)
    
def encodeRoofingAtts(trainDF,testDF):
    
    roofStyleDict = {'RoofStyle':{'Shed':5, 'Mansard':4, 'Hip':3,
                                  'Gambrel':2, 'Gable':1,'Flat':0}}
    trainDF.replace(roofStyleDict,inplace=True)
    testDF.replace(roofStyleDict,inplace=True)
    
    roofMatDict = {'RoofMatl':{'WdShngl':8, 'WdShake':7, 'Tar&Grv':6,
                                 'Roll':4, 'Metal':3,'Membran':2,
                                 'CompShg':1, 'ClyTile':0}}
    trainDF.replace(roofMatDict,inplace=True)
    testDF.replace(roofMatDict,inplace=True)
    
    
#standardize function standardizes values across our dataset given a columns parameter
def standardize(trainDF,testDF, cols):
     trainDF.loc[:,cols] = trainDF.loc[:,cols].apply(lambda row: (row-trainDF.loc[:,cols].mean())/(trainDF.loc[:,cols].std()), axis=1)
     testDF.loc[:,cols] = testDF.loc[:,cols].apply(lambda row: (row-testDF.loc[:,cols].mean())/(testDF.loc[:,cols].std()), axis=1)
     
#normalize function that normalizes values across our dataset given a columns parameter
def normalize(trainDF, testDF, cols):
    trainDF.loc[:,cols] = trainDF.loc[:,cols].apply(lambda row:(row- trainDF.loc[:,cols].min())/((trainDF.loc[:, cols].max())- (trainDF.loc[:,cols].min())), axis =1)
    testDF.loc[:,cols] = testDF.loc[:,cols].apply(lambda row:(row- testDF.loc[:,cols].min())/((testDF.loc[:, cols].max())- (testDF.loc[:,cols].min())), axis =1)

#Helper function to fill missing values with the mean of all the values in attributes with numeric values
def fillMissingWMean(trainDF,testDF,col):
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(round(trainDF.loc[:, col].mean()))
    testDF.loc[:, col] = testDF.loc[:, col].fillna(round(testDF.loc[:, col].mean()))

# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

