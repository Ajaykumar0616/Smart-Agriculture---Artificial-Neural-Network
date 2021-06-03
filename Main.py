'''# code for dataset generation
import pandas as pd
import numpy as np

#generate a dataframe with random number for range values
df = pd.DataFrame(np.random.randint(low=600, high=1200, size=(49,)))
for i in range(0, 49):
    #print(np.around(df[0][i], decimals = 1))
    print(df[0][i])'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

dict1 = {0:"Bajra", 1:"Corn", 2:"Cotton", 3:"Groundnut", 4:"Jowar", 5:"Rice", 6:"Soyabean", 7:"Sugarcane", 8:"Wheat"}
dict2 = {"Cotton":[110, 45, 50], "Sugarcane":[175, 100, 100], "Jowar":[85, 35, 45], "Bajra":[50, 25, 20], "Soyabean":[25,70,20], "Corn":[90, 25, 10], "Rice":[100, 50, 50], "Wheat":[110, 50, 50], "Groundnut":[30, 50, 50]}
dict3 = {1:"Urea, Ammonium Sulphate, Sodium Nitrate", 2:"Calcium Hydrogen Phosphate or Superphosphate, Ammonium Hydrogen Phosphate or ammophos, Ammonium Phosphate", 3:"Potassium Nitrate or Potassium Sulphate, Potassium Chloride, Potassium Sulphate"}

# Importing the dataset
dataset = pd.read_csv('shuffle.csv')
dataset = dataset.sample(frac=1).reset_index(drop=True)
X = dataset.iloc[:, 0:7].astype(float)
y = dataset.iloc[:, 7].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

# Adding the second hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


'''# for sharing my model following pickling is done.
import pickle

with open('model_pickle', 'wb') as f:
    pickle.dump(classifier, f)


from sklearn.externals import joblib

joblib.dump(classifier, 'model_joblib')'''

#for getting input and output tensor name.
#[node.op.name for node in model.outputs]

#file implementation
'''f=open("input.txt", "r")
if f.mode == 'r':
    contents =f.readline()
array = [contents]

while contents:
    contents = f.readline()
    array.append(contents)
    
print(array)

#for i in range(0, len(array)-1):
uii = [[ float(j) for j in array[i].split(',')] for i in range(0, len(array)-1)]
print(uii)
'''
# Part 3 - Making predictions and evaluating the model
    
    #print(max(a))

        
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# user input
ui = [[8.0, 95, 48, 45, 35, 30, 800], [6.8, 70, 50, 30, 25, 25, 1000], [7.6, 90, 25, 10, 22, 19, 557], [6.5, 95, 45, 55, 20, 23, 200], 
      [7.3, 45, 25, 25, 10, 31, 800], [6.0, 120, 45, 45, 50, 23, 900], [7.0, 25, 70, 10, 20, 32, 800], [7.2, 75, 45, 35, 45, 29, 900]
      , [6.9, 150, 80, 80, 50, 30, 850]]

for i in ui:
    #print(i)
    new_pred = classifier.predict(sc.transform(np.array([i])))
    new_pred = (new_pred > 0.5)
    for j in new_pred:
        for k in range(0, 9):
            if(j[k] == True):
                print(dict1[k])
    #print(new_pred)
# single prediction
new_pred = classifier.predict(sc.transform(np.array([[7.7, 90, 25, 10, 44, 20, 432]])))
new_pred = (new_pred > 0.5)
print(new_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

cm = confusion_matrix(
    y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)


'''0 = Bajara, 1 = corn, 2 - cotton, 3 - groundnut, 4 - jowar, 5 - rice, 6 - soyabeans, 7 - sugercane, 8 - wheat.'''
''' correct answers 2, 3, 4, 5, 0, 8, 6, 4, 7'''
#bs = 5 & e = 5 = 3
#bs = 5 & e = 8 = 5
#bs = 4 & e = 8 = 7
#bs = 40 & e = 80 = 7
#bs = 3 & e = 8 = 7
#bs = 3 & e = 9 = 7
#bs = 10 & e = 16 = 7
"6.0, 120, 45, 45, 50, 23, 900"

# GUI
def crop(ph, n, p, k, d, t, r):
    new_pred = classifier.predict(sc.transform(np.array([[ph, n, p, k, d, t, r]])))
    new_pred = (new_pred > 0.5)
    print(new_pred)
    for j in new_pred:
        for k in range(0, 8):
            if(j[k] == True):
                #print(dict1[k])
                #lbl.configure(text=dict1[k])
                return(dict1[k])
    #lbl.configure(text=dict1[k])
    
def fertilizer(pc, n, p, k):
    for key, value in dict2.items():
        if(pc == key):
            if(n < value[0] and p < value[1] and k < value[2]):
                return("DEFICIENCY of N, P, K FOUND:"+"For N-"+dict3[1]+","+"For P-"+dict3[2]+","+"For K-"+dict3[3])
            elif(n < value[0] and p < value[1]):
                return("DEFICIENCY of N and P FOUND:"+"For N-"+dict3[1]+","+"For P-"+dict3[2])
            elif(p < value[1] and k < value[2]):
                return("DEFICIENCY of P and K FOUND:"+"For P-"+dict3[2]+","+"For K-"+dict3[3])
            elif(n < value[0] and k < value[2]):
                return("DEFICIENCY of N and K FOUND:"+"For N-"+dict3[1]+","+"For K-"+dict3[3])
            elif(n < value[0]):
                return("DEFICIENCY of N FOUND:"+dict3[1])
            elif(p < value[1]):
                return("DEFICIENCY of P FOUND:"+dict3[2])
            elif(k < value[2]):
                return("DEFICIENCY of K FOUND:"+dict3[3])
            else:
                return("NO DEFICIENCY")
                
    
def reset_values():    
    e1.set(0)
    e2.set(0)
    e3.set(0)
    e4.set(0)
    e5.set(0)
    e6.set(0)
    e7.set(0)
    v.set("Predicted Crop")
    w.set("Fertilizer Suggestion")
    
from tkinter import *

def predict():
    pc = crop(e1.get(), e2.get(), e3.get(), e4.get(), e5.get(), e6.get(), e7.get())
    fert = fertilizer(pc, e2.get(), e3.get(), e4.get())
    print(fert)
    v.set(pc)
    w.set(fert)

root = Tk()
root.resizable(0, 0)
root.geometry('550x600')
root.title("Agriculture")

#Label(root, justify=LEFT, compound = LEFT, padx = 10, text=fert).pack(side="right")

label_0 = Label(root, text="Crop Prediction System",width=20,font=("bold", 30))
label_0.place(x=60,y=40)


label_1 = Label(root, text="pH",width=20,font=("bold", 10))
label_1.place(x=80,y=130)

e1 = IntVar()
entry_1 = Entry(root, textvariable=e1)
entry_1.place(x=240,y=130)

label_2 = Label(root, text="N",width=20,font=("bold", 10))
label_2.place(x=80,y=160)

e2 = IntVar()
entry_2 = Entry(root, textvariable=e2)
entry_2.place(x=240,y=160)

label_3 = Label(root, text="P",width=20,font=("bold", 10))
label_3.place(x=80,y=190)

e3 = IntVar()
entry_3 = Entry(root, textvariable=e3)
entry_3.place(x=240,y=190)

label_4 = Label(root, text="K",width=20,font=("bold", 10))
label_4.place(x=80,y=220)

e4 = IntVar()
entry_4 = Entry(root, textvariable=e4)
entry_4.place(x=240,y=220)

label_5 = Label(root, text="Depth",width=20,font=("bold", 10))
label_5.place(x=68,y=250)

e5 = IntVar()
entry_5 = Entry(root, textvariable=e5)
entry_5.place(x=240,y=250)

label_6 = Label(root, text="T",width=20,font=("bold", 10))
label_6.place(x=80,y=280)

e6 = IntVar()
entry_6 = Entry(root, textvariable=e6)
entry_6.place(x=240,y=280)

label_7 = Label(root, text="Rainfall",width=20,font=("bold", 10))
label_7.place(x=60,y=310)

e7 = IntVar()
entry_7 = Entry(root, textvariable=e7)
entry_7.place(x=240,y=310)

Button(root, text='Reset',width=20,bg='brown',fg='white', command=reset_values).place(x=30,y=380)
Button(root, text='Submit',width=20,bg='brown',fg='white', command=predict).place(x=200,y=380)
Button(root, text='Quit',width=20,bg='brown',fg='white', command=root.destroy).place(x=370,y=380)

#label_8 = Label(root, text="Predicted Crop",width=20,font=("bold", 10))
#label_8.place(x=60,y=310)

v = StringVar()
e = Entry(root, textvariable=v, width=30, font=("bold", 20))
e.place(x=50,y=420)
w = StringVar()
e = Entry(root, textvariable=w, width=30, font=("bold", 20))
e.grid(row = 40, column = 0)
e.place(x=50,y=480)

v.set("Predicted Crop")
s = v.get()

w.set("Fertilizer Suggestion")


root.mainloop()
 



# protobuf
##[node.op.name for node in model.outputs]
'''
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# Exporting The Network In a .pb (Protobuf) File (Names Of Input and Output Layers may be Different)
def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
                         'protobuf6_graph.pbtxt')
    saver.save(K.get_session(), 'out/protobuf6.chkp')
    freeze_graph.freeze_graph('out/protobuf6_graph.pbtxt', None, \
                              False, 'out/protobuf6.chkp', output_node_name, \
                              "save/restore_all", "save/Const:0", \
                              'out/frozen_protobuf6.pb', True, "")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_protobuf6.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)
    
    with tf.gfile.FastGFile('out/opt_protobuf6.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    
    print("Saved")
    
for name in K.get_session().graph_def.node:
    print(name.name + "/n")

export_model(tf.train.Saver(), classifier, ["dense_1_input"], "dense_3/Softmax")

'''

# without using neural network, using various classifier algorithms i.e. inbuild in sklearn
# Plot of a ROC curve for a specific class
def multiclassplot(fpr, tpr, roc_auc, n_classes, name): 
    for i in range(n_classes):
        plt.figure()
        label = '%s: auc=%f' % (name, roc_auc[i])
        plt.plot(fpr[i], tpr[i], linewidth=5, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def delete_dict(fpr, tpr, roc_auc, n_classes):
    for i in range(n_classes):
        del fpr[i]
    for i in range(n_classes):
        del tpr[i]
    for i in range(n_classes):
        del roc_auc[i]
    return fpr, tpr, roc_auc

from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import label_binarize

CLASS_MAP = {'LogisticRegression': ('-', LogisticRegression()), 'Naive Bayes':('--', GaussianNB()), 'Decision Tree':('.-', DecisionTreeClassifier(max_depth=5)), 'Random Forest':(':', RandomForestClassifier(max_depth=5, n_estimators = 10, max_features=1)),}

dataset = pd.read_csv('shuffle.csv')
dataset = dataset.sample(frac=1).reset_index(drop=True)
#dataset.columns = ['pH', 'N', 'P', 'K', 'Depth', 'T', 'Rainfall', 'Crop'] 
#X = dataset.iloc[:, 0:7].astype(float)
#y = dataset.iloc[:, 7].values

'''factor = pd.factorize(dataset['Crop'])
dataset.species = factor[0]
definitions = factor[1]'''

X = dataset.iloc[:, 0:7].astype(float)
y = dataset.iloc[:, 7].values

y = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8])
n_classes = 9
# encode class values as integers
'''encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

for name, (line_fmt, model) in CLASS_MAP.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    pred = pd.Series(preds[:, 1])
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    auc_score = auc(fpr, tpr)
    label = '%s: auc=%f' % (name, auc_score)
    plt.plot(fpr, tpr, line_fmt, linewidth=5, label=label)
    
plt.legend(loc="lower right")
plt.title('Cpmparing Classifiers')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# solving roc_curve error,
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for name, (line_fmt, model) in CLASS_MAP.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    pred = pd.Series(preds[:, 1])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    multiclassplot(fpr, tpr, roc_auc, n_classes, name)
    fpr, tpr, roc_auc = delete_dict(fpr, tpr, roc_auc)

    












