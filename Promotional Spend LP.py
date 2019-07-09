# -*- coding: utf-8 -*-
"""
@author: mayank.singh
"""
import pulp
import numpy as np
import pandas as  pd
import re
import os

## Read the brand level data

#os.chdir("")

brand_mapping = pd.read_csv("brand_file.csv")
brand_mapping = brand_mapping.drop_duplicates()

base_data = pd.read_csv('product_file.csv',encoding = 'utf-8')
base_data['Month_Year'] = base_data['Calendar Month'].apply(str).str.split(".").apply(lambda x: x[0]) + "/" + base_data['Calendar Month'].apply(str).str.split(".").apply(lambda x: x[1])


## Map brands

final_data = pd.merge(base_data, brand_mapping, how = 'left', on = 'Product Code')
final_data['TPR'] = -final_data['TPR']

# Map the Channel
original_mapping = pd.read_csv("Original mapping.csv")

promo_subset = final_data.merge(original_mapping, how = 'left', left_on = ['Outlet Code', 'Calendar Month'], right_on = ['Outlet Code', 'Calendar Month.Year'])


# Read in the promo file
promo_info = pd.read_csv("Promo file.csv")
promo_info = promo_info[['Product.Code', 'Promo Categorization', 'Master.Brand']]

promo_subset = promo_subset.merge(promo_info, how = 'left', left_on = 'Product Code', right_on = 'Product.Code')


### Sum up at Outlet level
promo_subset_outlet = promo_subset.groupby(by = ['Outlet Code',
                                                 'Month_Year',
                                                 'Master Product Category',
                                                 'Master Brand desc',
                                                 'Promo Categorization']).aggregate({
                                                                                    'GSV': [np.sum],
                                                                                    'TPR':[np.sum],
                                                                                    'Sales PCS':[np.sum]
                                                                                    }).reset_index()
promo_subset_outlet.columns = promo_subset_outlet.columns.droplevel(1)


### Average for each outlet
promo_subset3 = promo_subset_outlet.groupby(by = ['Month_Year',
                                                  'Master Product Category',
                                                  'Master Brand desc',
                                                  'Promo Categorization']).aggregate({
                                                                                     'GSV': [np.mean],
                                                                                     'TPR':np.mean,
                                                                                     'Sales PCS': [np.mean]
                                                                                     }).reset_index()
promo_subset3.columns = promo_subset3.columns.droplevel(1)
promo_subset3['Date'] = '01/' + promo_subset3['Month_Year'].astype(str)


## Calculate TPR%

promo_subset3['TPR perc'] = promo_subset3['TPR']/promo_subset3['GSV']
promo_subset3['Brand_Promo'] = promo_subset3['Master Brand desc'] + '_' + promo_subset3['Promo Categorization']

#del final_data, original_mapping, promo_subset, quarter_map, promo_info, brand_mapping


## Linear regression to identify slope - Value

brand_promo_list = promo_subset3['Brand_Promo'].unique()

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

brandpromowise_slopes = pd.DataFrame(columns = ['Brand_Promo', 'Coef', 'Intercept'])

for i in range(0, len(brand_promo_list)):
    
    print(brand_promo_list[i])
    
    X = promo_subset3.loc[promo_subset3['Brand_Promo'] == brand_promo_list[i], 'TPR perc'].values.reshape(-1, 1)
    y = promo_subset3.loc[promo_subset3['Brand_Promo'] == brand_promo_list[i], 'GSV']

    linreg.fit(X, y)
    
    coef = linreg.coef_
    intercept = linreg.intercept_

    df = pd.DataFrame(np.reshape([brand_promo_list[i], np.float(coef), intercept], [1,3]), 
                      columns = ['Brand_Promo', 'Coef', 'Intercept'])

    brandpromowise_slopes = brandpromowise_slopes.append(df)
    print(brand_promo_list[i])


# removing negative slope brands
brandpromowise_slopes = brandpromowise_slopes.loc[brandpromowise_slopes['Coef'].apply(float) >= 0, :]


## Linear regression to identify slope - Volume
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

brandpromowise_slopes_vol = pd.DataFrame(columns = ['Brand_Promo', 'Vol_Coef', 'Vol_Intercept'])

for i in range(0, len(brand_promo_list)):
    
    print(brand_promo_list[i])
    
    X = promo_subset3.loc[promo_subset3['Brand_Promo'] == brand_promo_list[i], 'TPR perc'].values.reshape(-1, 1)
    y = promo_subset3.loc[promo_subset3['Brand_Promo'] == brand_promo_list[i], 'Sales PCS']

    linreg.fit(X, y)
    
    coef = linreg.coef_
    intercept = linreg.intercept_

    df = pd.DataFrame(np.reshape([brand_promo_list[i], np.float(coef), intercept], [1,3]), 
                      columns = ['Brand_Promo', 'Vol_Coef', 'Vol_Intercept'])

    brandpromowise_slopes_vol = brandpromowise_slopes_vol.append(df)
    print(brand_promo_list[i])


# Export Volume slopes

#brandpromowise_slopes_vol.to_csv('Brand Promo Vol Slopes Outlet level mean.csv', index = False)


# Current TPR% for constraint
tpr_perc_values = promo_subset3.loc[promo_subset3['Month_Year'] == '9/2017', ['Master Product Category', 'Master Brand desc', 'Brand_Promo', 'GSV', 'TPR perc']]
tpr_perc_values['lbound'] = 0.5 * tpr_perc_values['TPR perc'] # lower bound 50% of current TPR
tpr_perc_values['ubound'] = 1.5 * tpr_perc_values['TPR perc'] # upper bound 150% of current TPR


data = tpr_perc_values.merge(brandpromowise_slopes, how = 'inner', left_on = 'Brand_Promo', right_on = 'Brand_Promo')
data['Coef'] = data['Coef'].apply(float)
data.reset_index(inplace = True)


# LINEAR PROGRAMMING

## Define the problem

# Maximize GSV/TPR%

prob = pulp.LpProblem('OptimumTPR', pulp.LpMaximize)

# Declare variables (w/ TPR constraints)

decision_variables = []

for rownum, row in data.iterrows():
    
    variable = str('x' + str(row['index']))
    variable = pulp.LpVariable(str(variable), lowBound = row['lbound'], upBound = row['ubound'], cat = 'Continuous') # low/up bound
    decision_variables.append(variable)
    
print('Total number of decision variables: ' + str(len(decision_variables)))

## Create optimization function

GSVbyTPR = ''
for rownum, row in data.iterrows():
    for i, tpr in enumerate(decision_variables):
        if rownum == i:
            formula = row['Coef']*tpr
            GSVbyTPR += formula

print('Optimization function: ' + str(GSVbyTPR))

prob += GSVbyTPR


## Define Constraints

budget = sum(data['GSV'] * data['TPR perc'])

budget_constraints = ''
for rownum, row in data.iterrows():
    for i, tpr in enumerate(decision_variables):
        if rownum == i:
            formula = row['GSV']*tpr
            budget_constraints += formula

prob += budget_constraints <= budget

print('Budget Constraints: ' + str(budget_constraints))

print(prob)

#prob.writeLP('Trade Spends Brand_Promo_mean.lp')


## Solve for results

from pulp import *

optimization_result = prob.solve()

assert optimization_result == pulp.LpStatusOptimal
print('Status:', LpStatus[prob.status])
print('Optimal Solution to the problem: ', value(prob.objective))
print('Individual decision variables: ')
for v in prob.variables():
    print(v.name, '=', v.varValue)


variable_name = []
variable_value = []

for v in prob.variables():
    variable_name.append(v.name)
    variable_value.append(v.varValue)
    
df = pd.DataFrame({'index': variable_name, 'value': variable_value})
for rownum, row in df.iterrows():
    value = re.findall(r'(\d+)', row['index'])
    df.loc[rownum, 'index'] = int(value[0])
    

df = df.sort_values(by = 'index')
result = pd.merge(data, df, on = 'index')

# Hypothesis testing for Product level
promo_subset_allbrands = promo_subset.groupby(by = ['Month_Year', 'Master Product Category', 'Master Brand desc', 'Master Sub-Brand Code', 'Master Sub-Brand', 'Promo Categorization', 'Product Code', 'Product', 'Pack Size']).aggregate({'GSV': [np.mean], 'TPR':np.mean, 'Sales PCS': [np.mean]}).reset_index()
promo_subset_allbrands.columns = promo_subset_allbrands.columns.droplevel(1)

#promo_subset_allbrands.to_csv("All Brands Subset.csv", index = False)
# %reset
