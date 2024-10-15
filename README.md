# [Travel Insurance Dataset](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data)
## Can I predict if customers will buy travel insurance? 

These data were acquired from [Kaggle](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data)
***
### Project Description
This notebook uses the 
[Travel Insurance Dataset](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data) to generate 
hypotheses tests and supervised machine learning models (i.e. classification models) in order to predict whether a 
customer will purchase travel insurance and to better understand customers who do and do not purchase travel insurance. 

Goals:
- Perform data wrangling and exploratory data analysis (EDA)
- Plot the data informatively
- Create various statistical and machine learning models to better understand the dataset 
- optimize the classification model 
- Test the optimized classification model

Altogether, these aims should allow us to predict whether someone is likely to purchase travel insurance given our 
model and to better understand the relationships between the features in these data and purchasing travel insurance. 
***
### Conclusions

- Large families buy more travel insurance than smaller families, but at a rate that below chance, indicating a potentially untapped market.
- People who buy travel insurance do appear to be slightly older than those who do not.
- Wealthier customers purchase travel insurance.
- Whether someone has ever traveled abroad or not appears to be the most important factor determining whether they will buy travel insurance or not. This increases the odds of purchasing travel insurance by over 400% according to logistic regression modeling.
- Other important positive predictors include Frequent Flyers and Chronic Disease.
- according to logistic regression and linear discriminant analysis, important features that predict TravelInsurance include: EverTravelledAbroad, FrequentFlyer, Family Members, and Chronic Diseases 

A variety of non-linear models outperformed linear models. All models generalized well, as did an ensemble Voting Model and included both linear and non-linear models <br><br> (see __TravelInsurancePrediction.ipynb__ for all details about model performance)

***

### Installation Instructions
- Clone or download repository <br>essential items:<br> 
  - dataset from Kaggle (TravelInsurancePrediction.csv)<br>
  - utils module  
  - ensure all requirements are installed, best practice to do so is to create a virtual environment as described below
  ```bash
  $ python3 -m venv travel_ins/
  $ source travel_ins/bin/activate
  $ pip install -r requirements.txt
  ```
  - Then run cells in notebook (TravelInsurancePrediction.ipynb)

***
### Requirements
- ipython
- matplotlib
- numpy
- pandas
- scipy
- seaborn
- scikit-learn
- statsmodels
(see requirements.txt for version details and more dependencies used in the development environment)

***
License 
[The MIT License](https://opensource.org/license/mit)