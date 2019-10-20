from sklearn.preprocessing import StandardScaler


def standar():
    """
    StandarScaler data
    :return: None
    """
    sc = StandardScaler()

    data = sc.fit_transform([[223,23,24],[352,84,12],[732,74,54]])

    print(sc.mean_)
    print(sc.var_)

    print(data)

if __name__ == '__main__':
    standar()
