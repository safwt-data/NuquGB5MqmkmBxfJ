import logging
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from project_1.dataset import load_data


def split_data(df):
    y = df['Y']
    X = df.drop('Y',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, random_state=42, stratify= y )
    logging.info("Dataset split successfully")
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    logging.info("Trained successfully")
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    knn = KNeighborsClassifier(n_neighbors=20)
    trained_knn = train_model(knn, X_train, y_train)

    logreg = LogisticRegression(random_state =42)
    trained_logreg = train_model(logreg, X_train, y_train)

    svm = SVC(random_state =42)
    trained_svm = train_model(svm, X_train, y_train)

    normal_tree = DecisionTreeClassifier(random_state=42,
                                     class_weight='balanced')
    trained_normal_tree = train_model(normal_tree, X_train, y_train)

    xgboost = XGBClassifier(random_state =42)
    trained_xgboost = train_model(xgboost,X_train,y_train)

    rf = RandomForestClassifier(random_state=42,  
                            class_weight='balanced')
    trained_rf = train_model(rf,X_train,y_train)

    tuned_rf = RandomForestClassifier(random_state=42,  
                            class_weight='balanced',
                            max_depth = 17,
                            min_samples_leaf = 5,
                            min_samples_split = 6,
                            n_estimators = 675
                           )
    trained_tuned_rf = train_model(tuned_rf, X_train, y_train)



    



