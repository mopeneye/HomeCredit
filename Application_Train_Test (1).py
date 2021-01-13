#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Functions import*
import pandas as pd
import numpy as np


# In[2]:


train = read(r"E:\PROJECTS\Home_Credit_new\Data\application_train.csv")
test = read(r"E:\PROJECTS\Home_Credit_new\Data\application_test.csv")


# In[3]:


outliers_update(train, test)
days_features(train,test)
generate_binary_from_numerical(train, test, "DAYS_LAST_PHONE_CHANGE")
#train, test = new_features_from_EXT_features(train, test)
new_features_domain_knowledge(train, test)
class_synchronization(train, test)
collect_features(train, test, "FLAG_DOCUMENT")
drop_group_of_features(train, test, "REG_")
collect_features_and_binary(train, test, "CREDIT_BUREAU")
row_cross_replace(train, test,"NAME_INCOME_TYPE","OCCUPATION_TYPE","Pensioner","Pensioner")
White_collar,Blue_collar, Laborers = occupation_rare()
row_rename(train,test,"OCCUPATION_TYPE", "White_collar", White_collar)
row_rename(train,test,"OCCUPATION_TYPE", "Blue_collar", Blue_collar)
row_rename(train,test,"OCCUPATION_TYPE", "Laborers", Laborers)
personal_asset(train,test,"FLAG_OWN_CAR", "FLAG_OWN_REALTY","Personal_Assets")
row_cross_replace(train,test,"NAME_EDUCATION_TYPE","NAME_EDUCATION_TYPE","Academic degree","Higher education")
education_years(train, test) 
row_rename(train, test,"NAME_INCOME_TYPE", "Working", ["Student", "Unemployed", "Businessman"])
row_rename(train, test,"NAME_HOUSING_TYPE", "Other", ["Co-op apartment", "Municipal apartment","Office apartment", "Rented apartment", "With parents"])
row_rename(train, test,"NAME_FAMILY_STATUS", "Married", ["Civil marriage"])
row_rename(train, test,"NAME_FAMILY_STATUS", "Single",  ["Separated", "Single / not married", "Widow"])
log_transformation(train, test)
drop_features(train,test)
categorical_feats, numerical_feats = columns_dtypes(train)
train, test ,new_ohe = one_hot_encoder(train,test, categorical_feats, nan_as_category=True)
implement_robust_function(train, test)


# In[5]:


X = train.drop(["TARGET","SK_ID_CURR"], axis=1)
y = train["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123456)
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
print('LGBM accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm.predict(X_test))), '\n')
print(classification_report(y_test, lgbm.predict(X_test)))
lgbm_y_pred = lgbm.predict(X_test)
lgbm_cm = metrics.confusion_matrix( y_test,lgbm_y_pred, [1,0])
sns.heatmap(lgbm_cm, annot=True, fmt='.2f',xticklabels = ["Risk", "No Risk"] , yticklabels = ["Risk", "No Risk"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('LightGBM Model')


# In[6]:


Importance = pd.DataFrame({'Importance': lgbm.feature_importances_ * 100,
                           'Feature': X_train.columns}).sort_values(by="Importance", ascending =False).head(25)
fig=plt.figure(figsize=(10,8))
sns.barplot(x=Importance.Importance,y='Feature',data=Importance,color='blue')
plt.xticks(rotation=60)
plt.show()


# In[9]:


X_80,X_20 = train_test_split(train.drop(["SK_ID_CURR"], axis =1))


# In[10]:


# Veri setini shuffle ediyoruz.
shuffled_df = X_80.sample(frac=1,random_state=4)

# Kampanyaya evet yanıtıvermiş tüm müşteriler için ayrı bir data frame oluşturuyoruz.
yes_df = shuffled_df.loc[shuffled_df['TARGET'] == 1]

# Veri setimizde hayır yanıtını verenler çoğunluk sınıfı oluşturuyor.
# Çoğunluk sınıfına ait verilerden evet diyenlerin belirlediğimiz oranda olacak rastgele gözlemleri seçiyoruz.
no_df = shuffled_df.loc[shuffled_df['TARGET'] == 0].sample(n=int(X_80.TARGET.value_counts()[1]*1.8),random_state=42)

# Heriki datasetini birleştiriyoryz.
normalized_df = pd.concat([yes_df, no_df])


# In[11]:


print((normalized_df.TARGET.value_counts()/normalized_df.TARGET.count()))

plt.bar(['No', 'Yes'], normalized_df.TARGET.value_counts().values, facecolor = 'brown', edgecolor='brown', linewidth=0.5, ls='dashed')
sns.set(font_scale=1)
plt.title('Target Variable', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Amount')
plt.show()


# In[12]:


X = normalized_df.drop(["TARGET"], axis=1)
y = normalized_df["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123456)
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
print('LGBM accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm.predict(X_test))), '\n')
print(classification_report(y_test, lgbm.predict(X_test)))
lgbm_y_pred = lgbm.predict(X_test)
lgbm_cm = metrics.confusion_matrix( y_test,lgbm_y_pred, [1,0])
sns.heatmap(lgbm_cm, annot=True, fmt='.2f',xticklabels = ["Risk", "No Risk"] , yticklabels = ["Risk", "No Risk"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('LightGBM Model')


# In[13]:


Importance = pd.DataFrame({'Importance': lgbm.feature_importances_ * 100,
                           'Feature': X_train.columns}).sort_values(by="Importance", ascending =False).head(25)
fig=plt.figure(figsize=(10,8))
sns.barplot(x=Importance.Importance,y='Feature',data=Importance,color='blue')
plt.xticks(rotation=60)
plt.show()


# In[14]:


# Daha önce %20 test verisi ayırmıştım.
X_test = X_20.drop(["TARGET"], axis=1)
y_test = X_20["TARGET"]


# In[15]:


print('LGBM accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm.predict(X_test))), '\n')
print(classification_report(y_test, lgbm.predict(X_test)))
lgbm_y_pred = lgbm.predict(X_test)
lgbm_cm = metrics.confusion_matrix( y_test,lgbm_y_pred, [1,0])
sns.heatmap(lgbm_cm, annot=True, fmt='.2f',xticklabels = ["Risk", "No Risk"] , yticklabels = ["Risk", "No Risk"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('LightGBM Model')

