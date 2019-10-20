import numpy as np
from sklearn.preprocessing import Imputer

def im():
    '''
    :return:
    '''
    need_process_data = [[1,5,7],[2,np.nan,34],[34,86,23]]

    Im = Imputer(missing_values='NaN',strategy='mean',axis=0)

    data = Im.fit_transform(need_process_data)

    print(data)


if __name__ == '__main__':
    im()