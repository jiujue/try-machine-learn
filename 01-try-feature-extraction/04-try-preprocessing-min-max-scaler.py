from sklearn.preprocessing import MinMaxScaler

def mmScaler():
    '''
    to one-zero
    :return: None
    '''

    mmS = MinMaxScaler()

    data = mmS.fit_transform([[223,23,24],[352,84,12],[732,74,54]])

    print(data)


if __name__ == '__main__':

    mmScaler()

