# Supervised Machine Learning Project: Predicting Heart Disease

# Summary

Cardiovascular diseases (CVDs) or heart diseases are the most prevalent disease globally. According to the World Health Organization (WHO), heart disease is the primary cause of death around the world and is estimated to cause about 17.9 million deaths a year, with four out of five deaths due to heart attacks and strokes

This is notebook was created investigate heart disease using a heart disease dataset from UC Irvine Machine Learning Repository. I will be using the processed Cleveland dataset since it has already been recduced from the original 76 attributes to 14. The goal is to predict if a patient has heart disease or not based on 13 features by creating multiple supervised machine learning models. Before this is done the data will be cleaned to find any missing values and potentially scaled or transformed to improved the ML model.

# Objective

Given the prevalence of heart disease and its impact on global health, this report intends to examine data collected as part of the Cleveland dataset and explored the following:

* What are the distributions of the variables such as sex, age, blood pressure, etc. - also explore how meaningful these factors are by applying and interpreting the results of statistical tests.  
* Is there a difference in presence of heart disease in men and women? The null hypothesis is that there is no difference between sexes.  
* What are the relationships between the variables and the prevalence of heart disease? Through our analysis, we intend to discuss these variables and draw conclusions on their impact on the presence of heart disease.  

Finally to develop various machine learning models to predict what patient has heart disease, and then compare different models to one another while checking individual models scoring metrics to comfirm the quality of the model created.

# Import all Python Libraries


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy import stats

from matplotlib import rcParams
rcParams['figure.figsize'] = 10,8
sns.set_theme()
```

## Read csv file into Pandas Dataframe


```python
columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
df = pd.read_csv('data.csv', names=columns)
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>145.0</td>
      <td>233.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>150.0</td>
      <td>0.0</td>
      <td>2.3</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>160.0</td>
      <td>286.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>108.0</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>120.0</td>
      <td>229.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>129.0</td>
      <td>1.0</td>
      <td>2.6</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>130.0</td>
      <td>250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>187.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>130.0</td>
      <td>204.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Data Preperation

A key rationale for the choice of the dataset (Cleveland rather than the other three) was the cleanliness of the original dataset as it would better lend to initial exploratory analysis and the development of a potential model.
The data preparation process was as follows:

* Acquire data: Reading data from the aforementioned sources;

* Set up the environment: In Pandas, libraries were imported with a comprehensive setup to run data frames and visualisation as well as statistics analysis;

* Prepare data for analysis:  
    -Imported the data  
    -Identified variables  
    -Cleaned undesired columns, checked for nulls  
    -Changed datatypes to numeric  
    
    
* Develop more variables for analysis: Created new data frame or changed existed ones as necessary for more in-depth analysis if needed:  
-Mapped multiple categories of disease to a old feature, 'num', which is either healthy (0) or has signs of heart disease (1). Previously this has a value from zero to four, though we are changing the problem to a binary classification problem   


* Perform analysis: Performed quantitative data analysis (Discrete) or Exploratory Data Analysis (EDA) for the data that covers heart disease.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       303 non-null    float64
     1   sex       303 non-null    float64
     2   cp        303 non-null    float64
     3   trestbps  303 non-null    float64
     4   chol      303 non-null    float64
     5   fbs       303 non-null    float64
     6   restecg   303 non-null    float64
     7   thalach   303 non-null    float64
     8   exang     303 non-null    float64
     9   oldpeak   303 non-null    float64
     10  slope     303 non-null    float64
     11  ca        303 non-null    object 
     12  thal      303 non-null    object 
     13  num       303 non-null    int64  
    dtypes: float64(11), int64(1), object(2)
    memory usage: 33.3+ KB


---

Columns 'ca' and 'thal' have dtype object but appear to be numerical values from the third cell above.

We notice below that there are rows with values filled in with '?', specifically 'ca' and 'thal' columns.


```python
df[df['thal'] == '?']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87</th>
      <td>53.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>128.0</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>115.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>?</td>
      <td>0</td>
    </tr>
    <tr>
      <th>266</th>
      <td>52.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>128.0</td>
      <td>204.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>?</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['ca'] == '?']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>52.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>138.0</td>
      <td>223.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>169.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>?</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>192</th>
      <td>43.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>132.0</td>
      <td>247.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>143.0</td>
      <td>1.0</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>?</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>287</th>
      <td>58.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>125.0</td>
      <td>220.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>144.0</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>2.0</td>
      <td>?</td>
      <td>7.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>38.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>138.0</td>
      <td>175.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>?</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We have two rows with unknown 'thal' values and four rows with unknown 'ca' values. These rows will be dropped for a cleaner dataset.


```python
drop_rows = [*df[df['thal'] == '?'].index.values, *df[df['ca'] == '?'].index.values]
df.drop(drop_rows, axis=0, inplace=True)
```

We also change the dtype of columns 'thal' and 'ca' to 'float64'.


```python
df[['thal','ca']] = df[['thal','ca']].astype('float64')
```

Since we are only trying to predict if a patient has heart disease we will change the values in the 'num' column to a binary value of 0 or 1. Since this column represents the diagnosis of heart disease (angiographic disease status) where a value 0: < 50% diameter narrowing and a value greater than 1: > 50% diameter narrowing then any number great than zero will be changed to a one.

We also create a copy of the Dataframe below so we can test to see if our transformations improve our models prediction ability.


```python
df.loc[(df.num > 0),'num']=1
df_T = df.copy()
```

# Exploratory Data Analysis

The 14 attributes in this dataset are:
1. age - numerical 
2. sex - cateforical 0-female, 1-male
3. cp - chest pain, categorical 1-typical angina, 2-atypical angina, 3-non-anginal pain, 4-asymptomatic 
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital), numerical
5. chol - serum cholestorel in mg/dl, numerical 
6. fbs - fasting blood sugar > 120 mg/dl, binary 0-false, 1-true 
7. restecg - resting electrocardiographic results, categorical 0-normal, 1-having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2-showing probable or definite left ventricular hypertrophy by Estes' criteria 
8. thalach - maximum heart rate achieved, numerical
9. exang - exercise induced angina, binary 0-false, 1-true
10. oldpeak - ST depression induced by exercise relative to rest, numerical
11. slope - the slope of the peak exercise ST segment, categorical 0-upsloping, 1-flat, 2-downsloping
12. ca - number of major vessels (0-3) colored by flourosopy, categorical 
13. thal - categorical  3-normal, 6-fixed defect, 7-reversable defect 
14. num - target variable, originally had value 0-4 but now only 0 or 1 as explained in section above.



```python
sns.displot(df, x="age", hue='sex', height=8, aspect=10/8)
plt.title('Figure 1. Distribution of Ages for each Sex');
```


![png](plot_imgs/output_25_0.png)



```python
num_m = len(df[df['sex'] == 1.0]) # number of males
num_f = len(df[df['sex'] == 0.0]) # number of females
print('Number of males in study are {} out of {}'.format(num_m,len(df)))
print('Number of females in study are {} out of {}'.format(num_f,len(df)))
```

    Number of males in study are 201 out of 297
    Number of females in study are 96 out of 297


We can see most of the patients in this dataset are males. It is also a small dataset with only 297 patients, which is too small by modern 'big data' standards.

In figure 2 we note there is a much higher chance of males having heart disease at 55.72%, and lower for females at 26.04%.


```python
sns.countplot(x='sex', hue="num", data=df)
plt.title('Figure 2. Ratio for each sex having signs of Heart Disease');
```


![png](plot_imgs/output_28_0.png)



```python
num_md = len(df[(df['num'] > 0) & (df['sex'] == 1.0)]) # males with heart disease
num_fd = len(df[(df['num'] > 0) & (df['sex'] == 0.0)]) # females with heart disease
print('Percentage of males in study with heart disease is {}%'.format(round(num_md/num_m*100,2)))
print('Percentage of females in study with heart disease is {}%'.format(round(num_fd/num_f*100,2)))
```

    Percentage of males in study with heart disease is 55.72%
    Percentage of females in study with heart disease is 26.04%



```python
df.corr()[['num']].sort_values(by='num',ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>num</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>thal</th>
      <td>0.526640</td>
    </tr>
    <tr>
      <th>ca</th>
      <td>0.463189</td>
    </tr>
    <tr>
      <th>oldpeak</th>
      <td>0.424052</td>
    </tr>
    <tr>
      <th>exang</th>
      <td>0.421355</td>
    </tr>
    <tr>
      <th>cp</th>
      <td>0.408945</td>
    </tr>
    <tr>
      <th>slope</th>
      <td>0.333049</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>0.278467</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.227075</td>
    </tr>
    <tr>
      <th>restecg</th>
      <td>0.166343</td>
    </tr>
    <tr>
      <th>trestbps</th>
      <td>0.153490</td>
    </tr>
    <tr>
      <th>chol</th>
      <td>0.080285</td>
    </tr>
    <tr>
      <th>fbs</th>
      <td>0.003167</td>
    </tr>
    <tr>
      <th>thalach</th>
      <td>-0.423817</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_palette("viridis")
sns.displot(x="trestbps", hue='sex', data=df, height=8, aspect=10/8)
plt.title('Figure 3. Distribution of Resting Blood Pressure for each sex');
```


![png](plot_imgs/output_31_0.png)



```python
sns.set_palette("cubehelix")
df[(df['trestbps'] > 0) & (df['sex'] == 1.0)].trestbps.plot(kind='hist')
plt.title('Figure 4. Distribution of Resting Blood Pressure for Men')
plt.xlabel('Resting Blood Pressure');
```


![png](plot_imgs/output_32_0.png)



```python
sns.set_palette("flare_r")
df[(df['trestbps'] > 0) & (df['sex'] == 0.0)].trestbps.plot(kind='hist')
plt.title('Figure 5. Distribution of Resting Blood Pressure for Women')
plt.xlabel('Resting Blood Pressure');
```


![png](plot_imgs/output_33_0.png)



```python
sns.set_palette("cubehelix")
df[(df['chol'] > 0) & (df['sex'] == 1.0)].trestbps.plot(kind='hist', bins=11)
plt.title('Figure 6. Distribution of Serum Cholesterol for Men')
plt.xlabel('Serum Cholesterol in mg/dl');
```


![png](plot_imgs/output_34_0.png)



```python
sns.set_palette("flare_r")
df[(df['chol'] > 0) & (df['sex'] == 0.0)].trestbps.plot(kind='hist', bins=11)
plt.title('Figure 7. Distribution of Serum Cholesterol for Women')
plt.xlabel('Serum Cholesterol in mg/dl');
```


![png](plot_imgs/output_35_0.png)



```python
sns.set_palette("cubehelix")
df[(df['thalach'] > 0) & (df['sex'] == 1.0)].trestbps.plot(kind='hist', bins=14)
plt.title('Figure 8. Distribution of Maximum Heart Rate Achieved for Men')
plt.xlabel('Maximum Heart Rate Achieved');
```


![png](plot_imgs/output_36_0.png)



```python
sns.set_palette("flare_r")
df[(df['thalach'] > 0) & (df['sex'] == 0.0)].trestbps.plot(kind='hist', bins=14)
plt.title('Figure 9. Distribution of Maximum Heart Rate Achieved for Women')
plt.xlabel('Maximum Heart Rate Achieved');
```


![png](plot_imgs/output_37_0.png)


# Statistics and Transformations


```python
stat = ols(formula='trestbps ~ age', data=df).fit()
```


```python
stat.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>trestbps</td>     <th>  R-squared:         </th> <td>   0.084</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.081</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   27.18</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 19 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>3.48e-07</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:51:06</td>     <th>  Log-Likelihood:    </th> <td> -1262.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   297</td>      <th>  AIC:               </th> <td>   2529.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   295</td>      <th>  BIC:               </th> <td>   2536.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  100.5966</td> <td>    6.046</td> <td>   16.640</td> <td> 0.000</td> <td>   88.699</td> <td>  112.494</td>
</tr>
<tr>
  <th>age</th>       <td>    0.5701</td> <td>    0.109</td> <td>    5.214</td> <td> 0.000</td> <td>    0.355</td> <td>    0.785</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>21.250</td> <th>  Durbin-Watson:     </th> <td>   1.894</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  26.106</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.569</td> <th>  Prob(JB):          </th> <td>2.14e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.902</td> <th>  Cond. No.          </th> <td>    338.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
tmp = stat.params[1]*df['age'] + stat.params[0]

sns.set_palette("crest")
plt.scatter(df['age'], df['trestbps'])
plt.plot(df['age'], tmp, c='r')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.title('Figure 10. Resting Blood Pressure vs. Age');
```


![png](plot_imgs/output_41_0.png)


Above a linear regression has been attempted on predicting one's resting blood pressure based on their age. However it is shown below that one of the assumptions for this model has been violated, 'Constant variance'. From Wikipedia's article on Linear Regression, https://en.wikipedia.org/wiki/Linear_regression, one of the assumptions is the residuals will be normal distributed. We see in the histogram, Figure 11, that there is a negative skew to this distribution and is not normally distributed, and the Kurtosis is approximately one when it should be closer to three. 

Also the R<sup>2</sup> value for this linear regression is only 0.084 which is a poor fit indead. One of the reasons machine learning has become so important is to come up with more sophiscated models to predict ones medical condition. If we are having a hard time predicting a person's blood pressure, than predicting if they have heart disease will be orders of magnitude more difficult!


```python
tmp_df = tmp - df['trestbps']
tmp_df.hist(bins=20)
plt.title('Figure 11. Residuals of Linear Regression in Figure 10')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
tmp_df.describe()
```




    count    2.970000e+02
    mean     4.406800e-14
    std      1.699691e+01
    min     -6.747517e+01
    25%     -1.032591e+01
    50%      8.143871e-01
    75%      1.194570e+01
    max      3.879644e+01
    dtype: float64




![png](plot_imgs/output_43_1.png)



```python
tmp_df.skew()
```




    -0.5722868823804681




```python
tmp_df.kurtosis()
```




    0.937352812271532



Scatter Plot below takes a while to run and can be skipped.


```python
pd.plotting.scatter_matrix(df, figsize=(20,16));
```


![png](plot_imgs/output_47_0.png)


## Addressing Skew and Scaling Input Variables

Many machine learning models benefit from having their input variables being having a normal distrubtion. We see along the diagonal of the scatter matrix plot above that some columns do not have a normal distribution, these are trestbps, thalch, chol, and oldpeak. We see their skew below


```python
df['trestbps'].skew()
#df['trestbps'].kurtosis()
```




    0.7000697177568133




```python
df['thalach'].skew()
```




    -0.5365400799355459




```python
df['chol'].skew()
```




    1.1180955225671279




```python
df['oldpeak'].skew()
```




    1.2471313241482946



We can use scipy stats module which will compute boxcox transformations (power transformation) on our skewed data (thalach, trestbps, and chol) and a yeojohnson transformation on our 'oldpeak' data. I would have used boxcox for all the data but 'oldpeak' has values that equal zero and boxcox can not handle values equal or less than zero.


```python
sci_tmp = stats.boxcox_normmax(df['trestbps'].values)
```

    /Users/DavidH/anaconda2/envs/py382/lib/python3.8/site-packages/scipy/stats/stats.py:3845: PearsonRConstantInputWarning: An input array is constant; the correlation coefficent is not defined.
      warnings.warn(PearsonRConstantInputWarning())



```python
sci_tt = stats.boxcox(df['trestbps'].values, sci_tmp)
```


```python
stats.skew(sci_tt)
```




    0.0037114292536530053




```python
plt.hist(sci_tt);
plt.title('Figure 12. Distribution of Scaled Resting Blood Pressure')
plt.xlabel('Scaled Resting Blood Pressure')
plt.ylabel('Frequency');
```


![png](plot_imgs/output_58_0.png)


Compared to original distribution shown below the transformation above is now closer to a Gaussian distribution.


```python
df['trestbps'].hist()
plt.title('Figure 13. Distribution of Original Resting Blood Pressure')
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Frequency');
```


![png](plot_imgs/output_60_0.png)



```python
df['oldpeak'].describe()
```




    count    297.000000
    mean       1.055556
    std        1.166123
    min        0.000000
    25%        0.000000
    50%        0.800000
    75%        1.600000
    max        6.200000
    Name: oldpeak, dtype: float64




```python
sns.set_palette("viridis")
box_peak = stats.yeojohnson_normmax(df['oldpeak'].values)
box_peak = stats.yeojohnson(df['oldpeak'].values, box_peak)
df_T['oldpeak'] = box_peak
plt.hist(box_peak);
```


![png](plot_imgs/output_62_0.png)



```python
stats.skew(box_peak)
```




    0.11474944210994208




```python
df['oldpeak'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12fb19490>




![png](plot_imgs/output_64_1.png)



```python
sns.set_palette("crest")
box_thal = stats.boxcox_normmax(df['thalach'].values)
box_thal = stats.boxcox(df['thalach'].values, box_thal)
df_T['thalach'] = box_thal
plt.hist(box_thal);
```


![png](plot_imgs/output_65_0.png)



```python
df['thalach'].hist();
```


![png](plot_imgs/output_66_0.png)



```python
stats.skew(box_thal)
```




    -0.02153700558620656




```python
sns.set_palette("viridis")
box_chol = stats.boxcox_normmax(df['chol'].values)
box_chol = stats.boxcox(df['chol'].values, box_chol)
df_T['chol'] = box_chol
plt.hist(box_chol);
```

    /Users/DavidH/anaconda2/envs/py382/lib/python3.8/site-packages/scipy/stats/stats.py:3845: PearsonRConstantInputWarning: An input array is constant; the correlation coefficent is not defined.
      warnings.warn(PearsonRConstantInputWarning())



![png](plot_imgs/output_68_1.png)



```python
plt.hist(df['chol']);
```


![png](plot_imgs/output_69_0.png)



```python
stats.skew(box_chol)
```




    -0.04094429306060899



Lastly we use RobustScaler function from sklean preprocessing module which is meant to good at dealing with outliers when scaling the data. It does this by removing the median and scales the data according to the quantile range


```python
scaler = RobustScaler()
collist = df_T.columns.tolist()
collist.remove('num')
#df_T[['thalach','oldpeak','chol','trestbps']] = scaler.fit_transform(df_T[['thalach','oldpeak','chol','trestbps']])
df_T[collist] = scaler.fit_transform(df_T[collist])
```

## Dummy Variables

Also need dummy variables in our dataframe for all the categorical variables in the dataset. This will help the machine learning models handle categorical data.


```python
cat_var = ['cp','restecg','slope','thal','ca']
data = pd.get_dummies(df, prefix=cat_var, columns=cat_var)
data_T = pd.get_dummies(df_T, prefix=cat_var, columns=cat_var)
```

# Models without Transformations


```python
X_train, X_test, y_train, y_test = train_test_split(data.drop('num',axis=1), data['num'], test_size=0.2, random_state=42)
```

## K Neighbors Classifier


```python
neigh = KNeighborsClassifier()
parameters = {
    'n_neighbors':[2,3,4,5,6,7,8,9],
    'weights':('uniform','distance'),
    'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'),
    'leaf_size':[15,20,25,30,35,40,45],
    'p':[1,2,3],
}
clf = GridSearchCV(neigh, parameters, cv=5, n_jobs=-1, verbose=5)
clf.fit(X_train, y_train)
```

    Fitting 5 folds for each of 1344 candidates, totalling 6720 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=-1)]: Done 176 tasks      | elapsed:    0.4s
    [Parallel(n_jobs=-1)]: Done 1552 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=-1)]: Done 3568 tasks      | elapsed:    5.7s
    [Parallel(n_jobs=-1)]: Done 6160 tasks      | elapsed:   11.3s
    [Parallel(n_jobs=-1)]: Done 6720 out of 6720 | elapsed:   12.4s finished





    GridSearchCV(cv=5, estimator=KNeighborsClassifier(), n_jobs=-1,
                 param_grid={'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                             'leaf_size': [15, 20, 25, 30, 35, 40, 45],
                             'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9],
                             'p': [1, 2, 3], 'weights': ('uniform', 'distance')},
                 verbose=5)




```python
clf.best_params_
```




    {'algorithm': 'auto',
     'leaf_size': 15,
     'n_neighbors': 9,
     'p': 1,
     'weights': 'uniform'}




```python
clf.best_score_
```




    0.6918439716312056




```python
clf.best_estimator_.score(X_test, y_test)
```




    0.65



## Logistic Regression Classifier


```python
lr = LogisticRegression()
estimators = {
    'penalty':('l1', 'l2', 'elasticnet'),
    'tol':[1e-6,1e-5,1e-4,1e-3,1e-2],
    'C':[0.01,0.05,0.1,0.5,1.0,2.0],
    'solver':('newton-cg','lbfgs','liblinear','sag','saga'),
    'max_iter':[10000]
}

clf_lr = GridSearchCV(lr, estimators, cv=5, n_jobs=-1, verbose=5)
clf_lr.fit(X_train, y_train)
```

    Fitting 5 folds for each of 450 candidates, totalling 2250 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=-1)]: Done 180 tasks      | elapsed:    0.7s
    [Parallel(n_jobs=-1)]: Done 1120 tasks      | elapsed:    9.3s
    [Parallel(n_jobs=-1)]: Done 1528 tasks      | elapsed:   19.1s
    [Parallel(n_jobs=-1)]: Done 2081 tasks      | elapsed:   32.6s
    [Parallel(n_jobs=-1)]: Done 2250 out of 2250 | elapsed:   38.7s finished





    GridSearchCV(cv=5, estimator=LogisticRegression(), n_jobs=-1,
                 param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                             'max_iter': [10000],
                             'penalty': ('l1', 'l2', 'elasticnet'),
                             'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag',
                                        'saga'),
                             'tol': [1e-06, 1e-05, 0.0001, 0.001, 0.01]},
                 verbose=5)




```python
clf_lr.best_params_
```




    {'C': 1.0,
     'max_iter': 10000,
     'penalty': 'l1',
     'solver': 'liblinear',
     'tol': 0.01}




```python
clf_lr.best_score_
```




    0.8311170212765957




```python
clf_lr.best_estimator_.score(X_test, y_test)
```




    0.8833333333333333



## SVC Classifier


```python
svc = SVC()
est_svc = {
    'kernel':('linear', 'sigmoid'),
    'degree':[3,4,5,6],
    'C':[0.01,0.05,0.1,0.5,1.0,2.0],
    'tol':[1e-4,1e-3,1e-2],
    'gamma':('scale','auto')
}

clf_svc = GridSearchCV(svc, est_svc, cv=5, n_jobs=-1, verbose=5)
clf_svc.fit(X_train, y_train)
```

    Fitting 5 folds for each of 288 candidates, totalling 1440 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    1.2s
    [Parallel(n_jobs=-1)]: Done  72 tasks      | elapsed:    1.4s
    [Parallel(n_jobs=-1)]: Done 1094 tasks      | elapsed:   11.6s
    [Parallel(n_jobs=-1)]: Done 1343 tasks      | elapsed:   22.3s
    [Parallel(n_jobs=-1)]: Done 1425 out of 1440 | elapsed:   28.5s remaining:    0.3s
    [Parallel(n_jobs=-1)]: Done 1440 out of 1440 | elapsed:   30.5s finished





    GridSearchCV(cv=5, estimator=SVC(), n_jobs=-1,
                 param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                             'degree': [3, 4, 5, 6], 'gamma': ('scale', 'auto'),
                             'kernel': ('linear', 'sigmoid'),
                             'tol': [0.0001, 0.001, 0.01]},
                 verbose=5)




```python
clf_svc.best_params_
```




    {'C': 2.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'tol': 0.0001}




```python
clf_svc.best_score_
```




    0.8355496453900709




```python
clf_svc.best_estimator_.score(X_test,y_test)
```




    0.85



## Random Forest Classifier


```python
rf = RandomForestClassifier()
est_rf = {
    'criterion':('gini', 'entropy'),
    'n_estimators':[50,80,100,120,150,180,200],
    'max_depth':[None,1,2,3,4,5],
    'min_samples_split':[2,3,4],
    'min_samples_leaf':[1,2]
}

clf_rf = GridSearchCV(rf, est_rf, cv=5, n_jobs=-1, verbose=5)
clf_rf.fit(X_train, y_train)
```

    Fitting 5 folds for each of 504 candidates, totalling 2520 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    1.5s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    3.8s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:    7.6s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:   13.1s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:   20.4s
    [Parallel(n_jobs=-1)]: Done 632 tasks      | elapsed:   30.0s
    [Parallel(n_jobs=-1)]: Done 866 tasks      | elapsed:   42.1s
    [Parallel(n_jobs=-1)]: Done 1136 tasks      | elapsed:   55.1s
    [Parallel(n_jobs=-1)]: Done 1442 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 1784 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=-1)]: Done 2162 tasks      | elapsed:  1.8min
    [Parallel(n_jobs=-1)]: Done 2520 out of 2520 | elapsed:  2.1min finished





    GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=-1,
                 param_grid={'criterion': ('gini', 'entropy'),
                             'max_depth': [None, 1, 2, 3, 4, 5],
                             'min_samples_leaf': [1, 2],
                             'min_samples_split': [2, 3, 4],
                             'n_estimators': [50, 80, 100, 120, 150, 180, 200]},
                 verbose=5)




```python
clf_rf.best_params_
```




    {'criterion': 'entropy',
     'max_depth': 1,
     'min_samples_leaf': 2,
     'min_samples_split': 4,
     'n_estimators': 100}




```python
clf_rf.best_score_
```




    0.835372340425532




```python
clf_rf.best_estimator_.score(X_test,y_test)
```




    0.8833333333333333



# Model with Transformations


```python
X_tr, X_t, y_tr, y_t = train_test_split(data_T.drop('num',axis=1), data_T['num'], test_size=0.2, random_state=42)
```

## K Neighbors Classifier


```python
neigh_T = KNeighborsClassifier()
parameters_T = {
    'n_neighbors':[2,3,4,5,6,7,8,9],
    'weights':('uniform','distance'),
    'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'),
    'leaf_size':[15,20,25,30,35,40,45],
    'p':[1,2,3],
}
clf_T = GridSearchCV(neigh_T, parameters_T, cv=5, n_jobs=-1, verbose=5)
clf_T.fit(X_tr, y_tr)
```

    Fitting 5 folds for each of 1344 candidates, totalling 6720 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    1.2s
    [Parallel(n_jobs=-1)]: Done  64 tasks      | elapsed:    1.4s
    [Parallel(n_jobs=-1)]: Done 1072 tasks      | elapsed:    3.1s
    [Parallel(n_jobs=-1)]: Done 3088 tasks      | elapsed:    6.3s
    [Parallel(n_jobs=-1)]: Done 5680 tasks      | elapsed:   11.1s
    [Parallel(n_jobs=-1)]: Done 6720 out of 6720 | elapsed:   13.2s finished





    GridSearchCV(cv=5, estimator=KNeighborsClassifier(), n_jobs=-1,
                 param_grid={'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                             'leaf_size': [15, 20, 25, 30, 35, 40, 45],
                             'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9],
                             'p': [1, 2, 3], 'weights': ('uniform', 'distance')},
                 verbose=5)




```python
clf_T.best_params_
```




    {'algorithm': 'auto',
     'leaf_size': 15,
     'n_neighbors': 7,
     'p': 1,
     'weights': 'distance'}




```python
clf_T.best_score_
```




    0.8312943262411349




```python
clf_T.best_estimator_.score(X_t, y_t)
```




    0.9



## Logisitic Regression Classifier


```python
lr_T = LogisticRegression()
estimators_T = {
    'penalty':('l1', 'l2', 'elasticnet'),
    'tol':[1e-6,1e-5,1e-4,1e-3,1e-2],
    'C':[0.01,0.05,0.1,0.5,1.0,2.0],
    'solver':('newton-cg','lbfgs','liblinear','sag','saga'),
    'max_iter':[10000]
}

clf_lr_T = GridSearchCV(lr_T, estimators_T, cv=5, n_jobs=-1, verbose=5)
clf_lr_T.fit(X_tr, y_tr)
```

    Fitting 5 folds for each of 450 candidates, totalling 2250 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=-1)]: Done 176 tasks      | elapsed:    0.3s
    [Parallel(n_jobs=-1)]: Done 1914 tasks      | elapsed:    2.2s
    [Parallel(n_jobs=-1)]: Done 2250 out of 2250 | elapsed:    2.6s finished





    GridSearchCV(cv=5, estimator=LogisticRegression(), n_jobs=-1,
                 param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                             'max_iter': [10000],
                             'penalty': ('l1', 'l2', 'elasticnet'),
                             'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag',
                                        'saga'),
                             'tol': [1e-06, 1e-05, 0.0001, 0.001, 0.01]},
                 verbose=5)




```python
clf_lr_T.best_params_
```




    {'C': 0.01,
     'max_iter': 10000,
     'penalty': 'l2',
     'solver': 'newton-cg',
     'tol': 1e-06}




```python
clf_lr_T.best_score_
```




    0.8310283687943263




```python
clf_lr_T.best_estimator_.score(X_t, y_t)
```




    0.9



## SVC Classifier


```python
svc = SVC()
est_svc = {
    'kernel':('linear', 'sigmoid'),
    'degree':[3,4,5,6],
    'C':[0.01,0.05,0.1,0.5,1.0,2.0],
    'tol':[1e-4,1e-3,1e-2],
    'gamma':('scale','auto')
}

clf_svc = GridSearchCV(svc, est_svc, cv=5, n_jobs=-1, verbose=5)
clf_svc.fit(X_train, y_train)
```

    Fitting 5 folds for each of 288 candidates, totalling 1440 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    1.2s
    [Parallel(n_jobs=-1)]: Done  72 tasks      | elapsed:    1.4s
    [Parallel(n_jobs=-1)]: Done 1094 tasks      | elapsed:   11.6s
    [Parallel(n_jobs=-1)]: Done 1343 tasks      | elapsed:   22.3s
    [Parallel(n_jobs=-1)]: Done 1425 out of 1440 | elapsed:   28.5s remaining:    0.3s
    [Parallel(n_jobs=-1)]: Done 1440 out of 1440 | elapsed:   30.5s finished





    GridSearchCV(cv=5, estimator=SVC(), n_jobs=-1,
                 param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                             'degree': [3, 4, 5, 6], 'gamma': ('scale', 'auto'),
                             'kernel': ('linear', 'sigmoid'),
                             'tol': [0.0001, 0.001, 0.01]},
                 verbose=5)




```python
clf_svc.best_params_
```




    {'C': 2.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'tol': 0.0001}




```python
clf_svc.best_score_
```




    0.8355496453900709




```python
clf_svc.best_estimator_.score(X_test,y_test)
```




    0.85



## Random Forest Classifier


```python
rf_T = RandomForestClassifier()
est_rf_T = {
    'criterion':('gini', 'entropy'),
    'n_estimators':[50,80,100,120,150,180,200],
    'max_depth':[None,1,2,3,4,5],
    'min_samples_split':[2,3,4],
    'min_samples_leaf':[1,2]
}

clf_rf_T = GridSearchCV(rf_T, est_rf_T, cv=5, n_jobs=-1, verbose=5)
clf_rf_T.fit(X_tr, y_tr)
```

    Fitting 5 folds for each of 504 candidates, totalling 2520 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=-1)]: Done  96 tasks      | elapsed:    4.6s
    [Parallel(n_jobs=-1)]: Done 276 tasks      | elapsed:   12.3s
    [Parallel(n_jobs=-1)]: Done 528 tasks      | elapsed:   24.0s
    [Parallel(n_jobs=-1)]: Done 852 tasks      | elapsed:   40.4s
    [Parallel(n_jobs=-1)]: Done 1176 tasks      | elapsed:   57.6s
    [Parallel(n_jobs=-1)]: Done 1410 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 1680 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 1986 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=-1)]: Done 2328 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=-1)]: Done 2520 out of 2520 | elapsed:  2.1min finished





    GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=-1,
                 param_grid={'criterion': ('gini', 'entropy'),
                             'max_depth': [None, 1, 2, 3, 4, 5],
                             'min_samples_leaf': [1, 2],
                             'min_samples_split': [2, 3, 4],
                             'n_estimators': [50, 80, 100, 120, 150, 180, 200]},
                 verbose=5)




```python
clf_rf_T.best_params_
```




    {'criterion': 'entropy',
     'max_depth': 2,
     'min_samples_leaf': 2,
     'min_samples_split': 2,
     'n_estimators': 50}




```python
clf_rf_T.best_score_
```




    0.8312943262411346




```python
clf_rf_T.best_estimator_.score(X_t,y_t)
```




    0.8833333333333333



# Conclusion

The datasets on heart disease has been used by many research groups around the world, in particular the Cleveland dataset. This project has investigated this datasetand have found some very interesting trends.
The data suggests old age is a significant factor to heart disease. Heart disease typically affects men more than women and this trend was also observed in men at a younger age than women, on average. Individuals with heart disease have a lower maximum heart rate, whereas healthy patients are able to achieve a higher maximum heart rate. It is also important to note that high cholesterol is dangerous and one will have a higher likelihood of heart disease.

Some limitations encountered in this dataset is how small it is. There are only 303 rows of data and we only used 297 after the cleaning stage of this analysis. In the future one could acquire more data since this is typically is a better approach than even the most sophisticated models in machine learning, as stated by Peter Norvig et al. (2009), “But invariably simple models and a lot data trump more elaborate models based on less data.”(2009)<sup>6</sup>.


6. https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf
