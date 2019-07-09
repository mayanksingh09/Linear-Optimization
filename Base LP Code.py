# -*- coding: utf-8 -*-
"""
@author: mayank.singh
"""
import pulp
import numpy as np
import pandas as  pd
import re
import os

## Read file
os.chdir("") # Change this to reflect current working directory where code is located

promo_subset = pd.read_csv("./Data/File.csv")

## Calculate TPR%
### TPR: Promotional Spend
### GSV: Gross Sales Value

promo_subset['TPR perc'] = promo_subset['TPR']/promo_subset['GSV']

#del final_data, original_mapping, promo_subset, quarter_map, promo_info, brand_mapping


promo_subset['Brand_Promo'] = promo_subset['Master Brand desc'] + '_' + promo_subset['Promo Categorization']


## Linear regression to identify slope

brand_promo_list = promo_subset['Brand_Promo'].unique()

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

brandpromowise_slopes = pd.DataFrame(columns = ['Brand_Promo', 'Coef', 'Intercept'])

for i in range(0, len(brand_promo_list)):
    
    print(brand_promo_list[i])
    
    X = promo_subset.loc[promo_subset['Brand_Promo'] == brand_promo_list[i], 'TPR perc'].values.reshape(-1, 1)
    y = promo_subset.loc[promo_subset['Brand_Promo'] == brand_promo_list[i], 'GSV']

    linreg.fit(X, y)
    
    coef = linreg.coef_
    intercept = linreg.intercept_

    df = pd.DataFrame(np.reshape([brand_promo_list[i], np.float(coef), intercept], [1,3]), 
                      columns = ['Brand_Promo', 'Coef', 'Intercept'])

    brandpromowise_slopes = brandpromowise_slopes.append(df)

    print(brand_promo_list[i])


# removing negative slope brand promos

brandpromowise_slopes = brandpromowise_slopes.loc[brandpromowise_slopes['Coef'].apply(float) >= 0, :]

# Current TPR% for constraints

tpr_perc_values = promo_subset.loc[promo_subset['Month_Year'] == '9/1999', ['Master Product Category', 'Master Brand desc', 'Brand_Promo', 'GSV', 'TPR perc']]

tpr_perc_values['lbound'] = 0.5 * tpr_perc_values['TPR perc'] # lower bound 50% of current TPR
tpr_perc_values['ubound'] = 1.5 * tpr_perc_values['TPR perc'] # upper bound 150% of current TPR


data = tpr_perc_values.merge(brandpromowise_slopes, how = 'inner', left_on = 'Brand_Promo', right_on = 'Brand_Promo')

data['Coef'] = data['Coef'].apply(float)
data.reset_index(inplace = True)


# LINEAR PROGRAMMING

## Define the problem

### Maximize GSV/TPR%

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
result.columns = ['Master Product Category', 'Master Brand desc', 'Brand_Promo', 'GSV', 'TPR perc', 'lbound', 'ubound', 'Coef', 'Intercept', 'Optimized TPR perc']
