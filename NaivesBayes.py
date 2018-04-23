from sklearn import datasets
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
import pandas as pd


def gaussian(output, results):
    dummy_variable = []
    
    def dummy(features):
        dum = pd.get_dummies(results, dummy_na=True)
        df = pd.DataFrame(dum)
        return df

    df = pd.DataFrame(dummy(results))
    output_var = pd.DataFrame(output).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(df, output_var, test_size=0.3)    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    xtest_df = pd.DataFrame(X_test)
    ytest_df = pd.DataFrame(y_test)
    pred_df = pd.DataFrame(y_pred)
    
    frames = [xtest_df, ytest_df, pred_df]
    New_Grid = pd.concat(frames, axis = 1)
    
    print("Mean squared error: %.2f"
       % mean_squared_error(y_test, y_pred))
    
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
                      
    writer = pd.ExcelWriter('Results.xlsx', engine = 'xlsxwriter')
    New_Grid.to_excel(writer, sheet_name = 'Share Of Voice')
    writer.save()