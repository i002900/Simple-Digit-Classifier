## Simple Digit Recognition using Naive Bayes
## The image dataset is imported from Scvikit Learn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
#   

X = digits.data
y = digits.target

# The dataset is split into a Training and Test arrays
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

##
from sklearn.naive_bayes import GaussianNB  #1. model class
model = GaussianNB()                        #2. instantiate model


#

model.fit(X_train , y_train)                  #3. fit model to Training data set
# Use the Fitted(Trained model) to predict using the Test data set
y_model = model.predict(X_test)

# Determine model accuracy using inbuilt accuracy_score function
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test , y_model)
print(f"The accuracy score of this simple model is: {accuracy}")

#Print model predictions from y_model and compare with the Test labels
n = 0
 
for i in range(0,len(y_test)):
    if y_test[i] != y_model[i]:
        print(f"Actual:  , {y_test[i]},   Predicted: , {y_model[i]}")
        n += 1
print(f"There were {n} incorrect predictions from a total of {len(y_test)} labels")

# Develop a Visualisation of results using Confusion Matrix function
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_model)

# This matrix (mat) gives the results per label i.e classifications per label
import seaborn as sns
sns.heatmap(mat,square=True,annot=True, cbar=False)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Confusion Matrix')
plt.show()





