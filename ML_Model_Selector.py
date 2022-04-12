import pandas as pd
import numpy as np

class Model_Selector():
    
    def __init__(self,dataset):
        self.dataset = pd.read_csv(dataset)
        self.name = []
        self.accuracy = []
        self.final = []
    
    def run(self):
        
        X = self.dataset.iloc[:,:-1].values
        y = self.dataset.iloc[:,-1].values
        y = y.reshape(len(y),1)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        from sklearn.linear_model import LogisticRegression
        logi = LogisticRegression(random_state=0)    
        logi.fit(X_train,y_train.ravel())
        y_pred = logi.predict(X_test)
        from sklearn.metrics import accuracy_score
        rec = ['Logistic Regression',accuracy_score(y_test,y_pred)]
        self.accuracy.append([accuracy_score(y_test,y_pred)])
        self.name.append(rec)
        
        from sklearn.svm import SVC
        svc = SVC(kernel = 'linear', random_state=0)
        svc.fit(X_train,y_train.ravel())
        y_pred = svc.predict(X_test)
        from sklearn.metrics import accuracy_score
        rec = ['SVM Classifier',accuracy_score(y_test,y_pred)]
        self.accuracy.append([accuracy_score(y_test,y_pred)])
        self.name.append(rec)
        
        
        from sklearn.svm import SVC
        ksvm = SVC(kernel='rbf',random_state=0)
        ksvm.fit(X_train,y_train.ravel())
        y_pred = ksvm.predict(X_test)
        from sklearn.metrics import accuracy_score
        rec = ['Kernel SVM Classifier',accuracy_score(y_test,y_pred)]
        self.accuracy.append([accuracy_score(y_test,y_pred)])
        self.name.append(rec)
        
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        knn.fit(X_train, y_train.ravel())
        y_pred = knn.predict(X_test)
        from sklearn.metrics import accuracy_score
        rec = ['K-Nearest Neighbour Classifier',accuracy_score(y_test,y_pred)]
        self.accuracy.append([accuracy_score(y_test,y_pred)])
        self.name.append(rec)
        
        from sklearn.naive_bayes import GaussianNB
        naive = GaussianNB()
        naive.fit(X_train,y_train.ravel())
        y_pred = naive.predict(X_test)
        from sklearn.metrics import accuracy_score
        rec = ['Naives Bayes Classifier',accuracy_score(y_test,y_pred)]
        self.accuracy.append([accuracy_score(y_test,y_pred)])
        self.name.append(rec)
        
        
        from sklearn.tree import DecisionTreeClassifier
        deci = DecisionTreeClassifier(criterion='entropy',random_state=0)
        deci.fit(X_train,y_train.ravel())
        y_pred = deci.predict(X_test)
        from sklearn.metrics import accuracy_score
        rec = ['Decision Tree Classifier',accuracy_score(y_test,y_pred)]
        self.accuracy.append([accuracy_score(y_test,y_pred)])
        self.name.append(rec)
        
        from sklearn.ensemble import RandomForestClassifier
        rand = RandomForestClassifier(criterion='entropy',n_estimators=40,random_state=0)
        rand.fit(X_train,y_train.ravel())
        y_pred = rand.predict(X_test)
        from sklearn.metrics import accuracy_score
        rec = ['Random Trees Classifier',accuracy_score(y_test,y_pred)]
        self.accuracy.append([accuracy_score(y_test,y_pred)])
        self.name.append(rec)
        
        maxi = max(self.accuracy)
        for i in self.name:
            if i[1] == maxi:
                r = [i[0],i[1]]
                self.final.append(r)
            else:
                pass
        
        return self.final
    
def main():
    model = Model_Selector("hands_data.csv")
    ans = model.run()
    print(ans)  
    
    
if __name__ == '__main__':
    main()       
        
        
        
        
        
        
        
        
        
        
