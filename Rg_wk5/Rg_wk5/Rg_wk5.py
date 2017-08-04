import graphlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn
except Exception:
    pass

sales = graphlab.SFrame('C:\\Machine_Learning\\Rg_wk5\\kc_house_data.gl\\')
from math import log,sqrt

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors'] = sales['floors'].astype(float)
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = graphlab.linear_regression.create(sales,'price',all_features,l2_penalty=0.0,l1_penalty=1e10,validation=None)
model_all.coefficients.print_rows(num_rows=18)

(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate
l1_penalty = np.logspace(1, 7, num=13)
print l1_penalty
err_lst = []
for l1 in l1_penalty:
    model_l1 = graphlab.linear_regression.create(training,'price',all_features,l2_penalty=0.0,l1_penalty=l1,validation_set=None,verbose=False)
    pred = model_l1.predict(validation)
    err = pred - validation['price']
    err = err * err
    err_lst.append(err.sum())
err_lst_ar = np.asarray(err_lst)
print err_lst
print min(err_lst_ar)

l1_best = l1_penalty[0]
model_best = graphlab.linear_regression.create(training,'price',all_features,l2_penalty=0.0,l1_penalty=l1_best,validation_set=None,verbose=False)
model_best.coefficients.print_rows(num_rows=18)

max_nonzeros = 7
l1_penalty_values = np.logspace(8, 10, num=20)
l1_max = 1e8
l1_min = 1e10
for l1 in l1_penalty_values:
    model_l1 = graphlab.linear_regression.create(training,'price',all_features,l2_penalty=0.0,l1_penalty=l1,validation_set=None,verbose=False)
    nozero = model_l1['coefficients']['value'].nnz()
    print nozero
    if nozero > max_nonzeros:
        if (l1>l1_max):
            l1_max = l1
    if nozero < max_nonzeros:
        if(l1<l1_min):
            l1_min = l1
print l1_max
print l1_min

err_lst = []
l1_lst = []
l1_penalty_values = np.linspace(l1_min,l1_max,20)
for l1 in l1_penalty_values:
    model_l1 = graphlab.linear_regression.create(training,'price',all_features,l2_penalty=0.0,l1_penalty=l1,validation_set=None,verbose=False)
    nozero = model_l1['coefficients']['value'].nnz()
    print nozero
    if nozero == max_nonzeros:
        pred = model_l1.predict(validation)
        err = pred - validation['price']
        err = err * err
        err_lst.append(err.sum())
        l1_lst.append(l1)
err_lst_ar = np.asarray(err_lst)
print err_lst
print min(err_lst_ar)
print l1_lst

model_l1 = graphlab.linear_regression.create(training,'price',all_features,l2_penalty=0.0,l1_penalty=l1_lst[0],validation_set=None,verbose=False)
model_l1.coefficients.print_rows(num_rows=18)   