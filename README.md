# Heart-Disease-Prediction
![](https://github.com/KoreJosh/Heart-Disease-Prediction/blob/main/82f5faf38685839bcb87647afe6db756.jpg)

## Project Overview
This is an Heart diease predictive model project aimed at knowing using features selection to select useful features to determine wheather a patient  with various medical conditions (BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, Sex, AgeCategory, Race, Diabetic),  will have Heart diease or not.

## Data Source
The dataset contains 319,834 rows( each rows represent a patient), and 18 columns of various medical condtions. [See Here](https://github.com/KoreJosh/Heart-Disease-Prediction/blob/main/heart_2020_b.csv)

## Tools
-Python's Jupter nootbook

## Data Cleaning / Preparation

- Data Loading and Inspection
  - Using the .shape(), it shows that the dataset has 319834 rows and 18 columns
- Handling missing values and duplicates
  - the percentage of missing values is less 5% so the null values were dropped and the duplicates were also dropped
- Data Cleaning and Formating
  - The upper and lower quatiles  were calculated and outlier were dropped.
    
## Exploratory Data Analysis

### Spliting of categorical columns to oridinal and norminal categories:
  
   ```
ordCat = df1[['AgeCategory','GenHealth']]
normCat = df1[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking',
       'Sex','Race', 'Diabetic', 'PhysicalActivity','Asthma', 'KidneyDisease', 'SkinCancer']]
```

### Encoding the ordinals

```
import category_encoders as ce
#ordCat.GenHealth.unique()
gHCoder = ce.OrdinalEncoder(cols=['GenHealth'], return_df =True, 
                            mapping = [{'col':'GenHealth',
                                        'mapping':{'Poor':0,'Fair':1,'Good':2,'Very good':3,'Excellent':4}}])
ordCat['GH_encoded']= gHCoder.fit_transform(ordCat['GenHealth'])

ordCat.AgeCategory.unique()

acCoder =ce.OrdinalEncoder(cols=['AgeCategory'],return_df=True, 
                           mapping = [{'col':'AgeCategory',
                                       'mapping':{'18-24':0,'25-29':1,
                                                  '30-34':2,'35-39':3,
                                                  '40-44':4,'45-49':5,
                                                  '50-54':6,'55-59':7,
                                                  '60-64':8,'65-69':9,
                                                  '70-74':10,'75-79':11, 
                                                  '80 or older':12}}])
ordCat['AC_encoded'] = acCoder.fit_transform(ordCat['AgeCategory'])
```

### Using OneEncoder To Encode the Norminal Feature

```
#Encoding
normCols =list(normCat.columns) #obtaining names of the norminal columns and converting it to a list
normEncoder = ce.OneHotEncoder(cols=normCols, handle_unknown ='return_nan', return_df = True, use_cat_names = True)

normCat_encoded = normEncoder.fit_transform(normCat)
```

### Merging normCat_encoded and dfNew[numCols] to New DataFrame

```
normCat_encoded['GH_encoded'] = ordCat['GH_encoded']
normCat_encoded['AC_encoded'] = ordCat['AC_encoded']

newDf = pd.concat([normCat_encoded, dfNew[numCols]], axis = 1)
```

### Feature Selection

```
#recurrent feature elimination

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

#convert dataframe into an array

dfArr = newDf.values

X= dfArr[0:292779,0:34] #input feature
y= dfArr[0:292779, 34] #outcome or target feature

model = LogisticRegression(max_iter = 2000) #a call to the constructor of the logistic regression class
rfe = RFE(model, n_features_to_select = 8)
fit = rfe.fit(X,y)

fit.support_

fit.ranking_
```

### Using Train_Test_Split model on the Whole Features For Heart Disease Prediction

```
#Illustrating with a trained model

dfArr = newDf.values #converting dataframe into an array

X= dfArr[:, :-1]
y =dfArr[:, 34]

testSize =0.2 #specifying the size of our data to reserve for testing

seed =9 #set a value for randomized data splitting that will be reproducible

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=seed)

model = LogisticRegression() #instatiate the model

model.fit(X_train, y_train)

result = model.score(X_test,y_test)
```

## Result / Findings
Calling the "result" function of the model we had a prediction of 91%, also got the same percentage of prediction using just 5 features gotten from our recurrent feature elimination.
The K-Fold Cross Validation model also gave 91% prediction.


## Reconmendation

Using this few Features ['Stroke_No', 'Race_Asian', 'Diabetic_No','Diabetic_Yes (during pregnancy)', 'KidneyDisease_No'] we can easily predict wheather a patient would have Heart Disease or not.
