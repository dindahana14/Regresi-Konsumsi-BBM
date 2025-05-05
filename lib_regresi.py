#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle
from collections import defaultdict


# In[34]:


df = pd.read_csv('data/prediction_data.csv')
df


# In[35]:


print(df.columns)


# In[36]:


print(df.dtypes)


# In[37]:


df['GDP'] = df['GDP'].str.replace(',', '').astype(float)
df['Konsumsi BBM'] = df['Konsumsi BBM'].str.replace(',', '').astype(float)


# In[38]:


gdp = df['GDP']
konsumsi_bbm = df['Konsumsi BBM']
gdp_per_capita = df['GDP']/df['C']

plt.scatter(gdp_per_capita, konsumsi_bbm)
plt.xlabel('GDP per capita')
plt.ylabel('Konsumsi BBM')
plt.title('GDP per capita vs Konsumsi BBM')
plt.show()


# In[39]:


ln_gdp_per_capita = np.log(gdp_per_capita)
ln_konsumsi_bbm = np.log(konsumsi_bbm)
ln_gdp = np.log(gdp)


# In[40]:


# Split the data
X = ln_gdp_per_capita.values.reshape(-1,1)
y = ln_konsumsi_bbm.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

slope = model.coef_[0]
intercept = model.intercept_
print('Slope:', slope)
print('Intercept:', intercept)

def plot_linear_regression(df, slope, intercept):
    X = ln_gdp_per_capita
    y_actual = ln_konsumsi_bbm
    
    # Predicted values
    y_pred = slope * X + intercept  

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y_actual, color='blue', alpha=0.5, label='Actual Data')
    plt.plot(X, y_pred, color='red', label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')
    
    plt.title('GDP per Capita vs. BBM Consumption')
    plt.xlabel('GDP per Capita')
    plt.ylabel('BBM Consumption')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot results
plot_linear_regression(df, slope, intercept)


# In[41]:


model_params = {
    'slope': slope,
    'intercept': intercept
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_params, f)


# ### 📈 GDP Per Capita
#    $$
#    \text{GDP}[y] = \text{GDP}[y-1] \times \left(1 + \frac{\text{Angka Pertumbuhan Ekonomi}[y]}{100}\right)
#    $$
# 
#    $$
#    \text{GDP Per Capita}[y] = \frac{\text{GDP}[y]}{C[y]}
#    $$
# 

# In[42]:


def gdpc(gdp_before, start_year, angka_pertumbuhan_list, capita_list):
    gdp_list = []
    gdp_per_capita_list = []

    for idx in range(len(angka_pertumbuhan_list)):
        angka_pertumbuhan = angka_pertumbuhan_list[idx]

        if idx == 0:
            gdp = gdp_before * (1 + angka_pertumbuhan / 100)
        else:
            gdp = gdp_list[idx - 1] * (1 + angka_pertumbuhan / 100)
        gdp_list.append(gdp)

        capita = capita_list[idx]
        gdp_per_capita = gdp / capita if capita != 0 else 0
        gdp_per_capita_list.append(gdp_per_capita)

    return gdp_list, gdp_per_capita_list


# ### 📈 Total Konsumsi Per Tahun
# 
#    $$
#    \ln(\text{Konsumsi}) = \text{Intercept} + \text{Slope} \times \ln(\text{GDP})
#    $$
#    
#    $$
#    \text{Konsumsi} = \exp(\ln(\text{Konsumsi}))
#    $$
# 

# In[43]:


def konsumsi_per_tahun(start_year, end_year, gdp_per_capita_list, intercept, slope):
    konsumsi_list = []

    for i in range(start_year, end_year + 1):
        idx = i - start_year
        gdp = gdp_per_capita_list[idx]

        if gdp == 0:
            konsumsi = 0
        else:
            ln_gdp = np.log(gdp)
            ln_konsumsi = intercept + slope * ln_gdp
            konsumsi = np.exp(ln_konsumsi)

        konsumsi_list.append(konsumsi)

    return konsumsi_list


# ### 📈 Total Konsumsi BBM Per Jenis, Per Kebijakan JBT/JBKP, Per Kebijakan JBU
# 
# 1. **Total Konsumsi Per Jenis**:
# 
#    $$ 
#    \text{Percentage} = \frac{\text{proporsi}[y-1]}{\text{Total}[y-1]} 
#    $$
# 
#    $$ 
#    \text{Total proporsi} = \text{Percentage} \times \text{Total}[y] 
#    $$
# 
# 2. **Total Konsumsi JBT/JBKP**:
# 
#    $$ 
#    \text{Aggregate} = \frac{\text{Realisasi Kuota}}{\text{Total proporsi}} 
#    $$
# 
#    $$ 
#    \text{JBT/JBKP} = \text{Aggregate} \times \text{Total proporsi[y]} 
#    $$
# 
# 3. **Total Konsumsi JBU**:
# 
#    $$ 
#    \text{JBU} = \text{Total proporsi} - \text{JBT/JBKP} 
#    $$

# In[ ]:


def total_proporsi(total_year_before, total_year_after_list, grouped_proporsi, tahun_list):
    all_proporsi_result = []

    for idx, total_year_after in enumerate(total_year_after_list):
        proporsi_result = {}
        for jenis, total_proporsi in grouped_proporsi.items():
            rate = total_proporsi / total_year_before if total_year_before != 0 else 0
            proporsi_result[jenis] = total_year_after * rate

        all_proporsi_result.append({"Tahun": tahun_list[idx], **proporsi_result})

    return all_proporsi_result


# In[ ]:


def proporsi_jbkp_jbt(grouped_proporsi, filtered_proporsi, total_proporsi, jenis_bbm_list, tahun_list):
    all_jbkp_result = []

    for idx, total_year_after in enumerate(total_proporsi):
        jbkp_result = {}

        for _, row in filtered_proporsi.iterrows():
            jenis = row["JENIS BBM"]
            if jenis in jenis_bbm_list:
                kons_bbm = row["KONSUMSI BBM"]
                grouped_value = grouped_proporsi.get(jenis, None)
                total_value = total_year_after.get(jenis, 0)

                if grouped_value and grouped_value != 0:
                    agg = kons_bbm / grouped_value
                    jbkp_result[jenis] = agg * total_value
                else:
                    jbkp_result[jenis] = 0
            else:
                jbkp_result[jenis] = 0

        all_jbkp_result.append({"Tahun": tahun_list[idx], **jbkp_result})

    return all_jbkp_result


# In[ ]:


def proporsi_jbu(total_proporsi, jbkp_jbt_proporsi, jenis_bbm_list, tahun_list):
    all_jbu_result = []

    for idx, total_year_after in enumerate(total_proporsi):
        jbu_result = {}

        for jenis in jenis_bbm_list:
            total_value = total_year_after.get(jenis, 0)
            jbkp_value = jbkp_jbt_proporsi[idx].get(jenis, 0)

            if jbkp_value is not None:
                jbu_value = total_value - jbkp_value
                jbu_result[jenis] = jbu_value
            else:
                jbu_value = total_value
                jbu_result[jenis] = jbu_value

        all_jbu_result.append({"Tahun": tahun_list[idx], **jbu_result})

    return all_jbu_result

