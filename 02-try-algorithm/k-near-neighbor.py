from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


def KNN():
    '''
    step :
        # read data --- pandas.read
        # process data --- simplelize ,modification,standar
        # extract feature and expected
        # split train and test --- split
        # use algorithm --- k near neighbor
    :return: None
    '''
    # read data
    file_path = r'D:\chrome-download\facebook-v-predicting-check-ins\train.csv'
    data = pd.read_csv(file_path)
    #  process data
    #   modification
    print(data.shape)
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")

    time = pd.to_datetime(data['time'],unit='s')
    time = pd.DatetimeIndex(time)

    data['day'] = time.day
    data['week'] = time.week
    data['weekday'] = time.weekday
    data['hour'] = time.hour

    data = data.drop(['time'],axis=1)

    # extract feature and expected
    place_count = data.groupby('place_id').count()

    tf = place_count[place_count.row_id > 3].reset_index()

    data = data[data['place_id'].isin(tf.place_id)]
    print(data.shape)
    # data = data.iloc[::2]
    print(data.shape)
    print(data.head())

    # 0.4732860520094563
    print('start drop...')
    data = data.drop(['row_id'],axis=1)


    print('start split x y...')
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    # split train and test
    print('start spilt train test...')
    x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25)

    std = StandardScaler()

    print('start standar...')
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    print('start knn...')
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(x_train,y_train)

    print('end ...')
    print('score :',knn.score(x_test,y_test))

    print(data.head())







if __name__ == '__main__':

    KNN()