from sklearn import preprocessing
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/iris.csv')
    headers = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
    df.columns = headers

    label_encoder = preprocessing.LabelEncoder()
    df['Species'] = label_encoder.fit_transform(df['Species'])
    # Split into X and y
    X = df.drop(['Species'], axis=1)
    y = df['Species']

    # Apply standardScaler for X
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # splitting data into training and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    pd.DataFrame(X_train).to_csv('data/prepared/X_train.csv', header=None, index=None)
    pd.DataFrame(X_test).to_csv('data/prepared/X_test.csv', header=None, index=None)
    pd.DataFrame(y_train).to_csv('data/prepared/y_train.csv', header=None, index=None)
    pd.DataFrame(y_test).to_csv('data/prepared/y_test.csv', header=None, index=None)

