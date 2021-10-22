# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/dmitry/Downloads/titanic.csv')
original = df
model = LogisticRegression()
le = LabelEncoder()
df_survived = df['Survived']
df = df.drop('Survived', axis=1)
xnoise, ynoise = np.random.random(len(df)) / 2, np.random.random(len(df)) / 2

# let's check data completeness
# print(df.isnull().sum())
# looks like the data is almost complete except of age(177), cabin(687) and embarked(2)
# lack of age data is a problem but we actually don't need cabin and embarked
# we can get information about the cabin from the class and ticket price
# and the landing city does not affect the person's unsinkability)))

# For the starting point, we take the division by sex
df_sex = le.fit_transform(df['Sex'])
# sns.relplot(x=df_sex + xnoise, y=ynoise, hue=df_survived)
# plt.show()
# as we see, chances small if you are a man
# suppose all men are have to die and women are saved
x_train, x_test, y_train, y_test = train_test_split(df_sex, df_survived, test_size=0.75, shuffle=False)
model.fit(x_train.reshape(-1, 1), y_train)
# print(model.predict_proba(x_test.reshape(-1, 1)))
print(str(model.score(x_train.reshape(-1, 1), y_train)) + "/" + str(model.score(x_test.reshape(-1, 1), y_test)))
# accuracy in the region of 78 percent
# bad news for men((

# next step is to add polynomial features
df_sex_poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(df_sex.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(df_sex_poly, df_survived, test_size=0.75, shuffle=False)
model.fit(x_train, y_train)
print(str(model.score(x_train, y_train)) + "/" + str(model.score(x_test, y_test)))
# no expected effect: 77 percent

# let's look on the dependence of age and mortality on the Titanic
sns.relplot(x=df['Age'] + xnoise, y=ynoise, hue=df_survived)
# plt.show()
# there are more survivors among children than dead, but vice versa for old men
# ageism detected
# for the working-age population, age does not matter

# I concat all family members, because it should be no difference between parents, children and brothers/sisters
df['SibSpParCh'] = df.Parch + df.SibSp
# sns.relplot(x=df['SibSpParCh'] + xnoise, y=ynoise, hue=df_survived)
# plt.show()
# interesting: the chances of surviving are increased if you have 1 to 3 relatives on board
# but if there are more than 7 you are destined to die

# trace the correlation between ticket price and survival
sns.relplot(x=df['Fare'] + xnoise, y=ynoise, hue=df_survived)
# plt.show()
# capitalism: the more you pay, the better you are saved

# trace the correlation between ticket name and survival
sns.relplot(x=le.fit_transform(df['Name']) + xnoise, y=ynoise, hue=df_survived)
# plt.show()
# equal distribution: we don't need name

sns.relplot(x=le.fit_transform(df['PassengerId']) + xnoise, y=ynoise, hue=df_survived)
sns.relplot(x=le.fit_transform(df['Cabin']) + xnoise, y=ynoise, hue=df_survived)
sns.relplot(x=le.fit_transform(df['Embarked']) + xnoise, y=ynoise, hue=df_survived)
sns.relplot(x=le.fit_transform(df['Ticket']) + xnoise, y=ynoise, hue=df_survived)
# plt.show()
# looks like we don't need PassengerId, Cabin, Embarked and Ticket (just as Name)
# so I'll drop it down

df = df.drop(['PassengerId', 'Cabin', 'Embarked', 'Ticket'], axis=1)

# lets replace all NuN ages with median age (a kind of admission)
grp = df.groupby(['Sex', 'Pclass'])
grp['Age'].apply(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df["Age"].median())

# Also I'll add another features to train model better
features = ["Pclass", "Sex", "SibSpParCh", "Age"]
df_featured = pd.get_dummies(df[features])
x_train, x_test, y_train, y_test = train_test_split(df_featured, df_survived, test_size=0.75, shuffle=False)
model.fit(x_train, y_train)
print(str(model.score(x_train, y_train)) + "/" + str(model.score(x_test, y_test)))
# model a little bit better

# I decided to add special categories for age based on the assumptions from the graph
df['Age_Category'] = pd.cut(df['Age'], bins=[-1, 16, 60, 100], labels=['Child', 'Adult', 'Old'])
# and special category for family size
df['Family_Category'] = pd.cut(df['SibSpParCh'], bins=[-1, 0.9, 3, 12], labels=['Alone', 'Small family', 'Big family'])
# I check if we can use Fare as feature or it's better to use Pclass
print(df.loc[df[df['Pclass'] == 3]['Fare'].idxmax()][['Fare', 'Family_Category']])
# Class 3, Fare: 69.55, Family_Category: Big family
print(df.loc[df.loc[(df['Pclass'] == 2) & (df['Fare'] != 0.0)]['Fare'].idxmin()][['Fare', 'Family_Category']])
# Class 2, Fare: 10.5, Family_Category: Alone
# so 3 class passenger paid much more because of family then alone 2 class passenger
# I think Pclass is better metrics

# also let check the survival rate depending on social status (Mr, miss, mrs, etc)
df['Salutation'] = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
# print(np.unique(df['Salutation'].values))
sns.relplot(x=le.fit_transform(df['Salutation']) + xnoise, y=ynoise, hue=df_survived)
# plt.show()
# Salutation have real affect
# let split salutation on 4 groups (men, priests, married women, unmarried women)
# (spoiler: initially I divided into men and military officers, but unlike of them all priests died)
# the idea is that young women are more likely to survive and and almost impossible for priests
df['Salutation_Category'] = df['Salutation'].replace({'Capt': "Man",
                                                      'Col': "Man",
                                                      "Don": "Man",
                                                      "Dr": "Man",
                                                      "Jonkheer": "Man",
                                                      "Lady": "Married women",
                                                      "Major": "Man",
                                                      "Master": "Man",
                                                      "Miss": "Unmarried women",
                                                      "Mlle": "Unmarried women",
                                                      "Mme": "Married women",
                                                      "Mr": "Man",
                                                      "Mrs": "Married women",
                                                      "Ms": "Unmarried women",
                                                      "Rev": "Priest",
                                                      "Sir": "Man",
                                                      'the Countess': "Married women"})

sns.relplot(x=le.fit_transform(df['Salutation_Category']) + xnoise, y=ynoise, hue=df_survived)
# plt.show()

# and also I'll drop already unnecessary features
df = df.drop(['Name', 'Sex', 'SibSp', 'Parch', 'Fare', 'SibSpParCh', 'Salutation'], axis=1)

# let's use our features
features = ["Pclass", "Salutation_Category", "Family_Category", "Age_Category"]
df_featured = pd.get_dummies(df[features])
x_train, x_test, y_train, y_test = train_test_split(df_featured, df_survived, test_size=0.75, shuffle=False)
model.fit(x_train, y_train)
print(str(model.score(x_train, y_train)) + "/" + str(model.score(x_test, y_test)))
# average score are become higher!!!

original = original.drop(['PassengerId', 'Cabin', 'Embarked', 'Ticket', 'Name'], axis=1)
original['Sex'] = le.fit_transform(original['Sex'])
original['Sex'] = le.fit_transform(original['Survived'])
corr = original.corr()
print(corr)
plt.matshow(corr)
# plt.show()
# the correlation matrix suggests that sex and survival have the greatest correlation,
# the less obvious relationship between fare and passenger class.
# Even weaker relationships between age and passenger class,
# age and onboarded siblings, sex and passenger class, as well as passenger class and survival

# try to build confusion matrix
print(confusion_matrix(y_train.values.reshape(-1, 1), model.predict(x_train)))
# and to get precision and recall (I'll cheat and use classification_report)
print(classification_report(y_train.values.reshape(-1, 1), model.predict(x_train)))
# the report says that we more accurately determine that the passenger died (84%) than that he survived (78%)
# at the same time, we identify the dead with an accuracy of 90% and the survivors with an accuracy of 68%
# conclusion: it is easier for the algorithm to write everyone to the dead
# it seems that it is easier for certain classes of passengers to die than other classes to survive
# well, life is the path on which death lies in wait for us everywhere

# lets try to use another algorithm different regularization cost functions (such as L1/L2)
# and smaller inverse of regularization strength

# L1 regularization tries to estimate the median of the data
# liblinear solver could use L1 and it is recommended for solving large-scale classification problem
model1_liblinear = LogisticRegression(penalty='l1', C=0.5, solver='liblinear')
model1_liblinear.fit(x_train, y_train)
print(str(model1_liblinear.score(x_train, y_train)) + "/" + str(model1_liblinear.score(x_test, y_test)))

# L2 regularization tries to estimate the mean of the data
# saga solver could use L1 too, but the best solutions are obtained with l2
model2_saga = LogisticRegression(penalty='l2', C=0.5, solver='saga')
model2_saga.fit(x_train, y_train)
print(str(model2_saga.score(x_train, y_train)) + "/" + str(model2_saga.score(x_test, y_test)))

# Newton's method, more classic way to minimize error
model3_newton = LogisticRegression(penalty='l2', C=0.5, solver='newton-cg')
model3_newton.fit(x_train, y_train)
print(str(model3_newton.score(x_train, y_train)) + "/" + str(model3_newton.score(x_test, y_test)))

# all three multi params LogisticRegression objects have improved accuracy
# (actually looks like the main impact is С param make regularization stronger)
