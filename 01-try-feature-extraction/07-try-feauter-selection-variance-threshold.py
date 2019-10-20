from sklearn.feature_selection import VarianceThreshold

def var():
    '''
    variance threshold
    :return: None
    '''

    var = VarianceThreshold(threshold=0.0)

    data = var.fit_transform([[22, 23, 24], [23, 84, 12], [22, 74, 54]])

    print(data)

    var = VarianceThreshold(threshold=0.1)

    data = var.fit_transform([[22,23,24],[23,84,12],[22,74,54],
                              [22,23,24],[22,84,12],[22,74,54],
                              [22,23,24],[22,84,12],[22,74,54]])

    print(data)


if __name__ == '__main__':
    var()

