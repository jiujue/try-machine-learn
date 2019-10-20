from sklearn.decomposition import PCA

def pca():
    """
    reducing data dimensions ,use principal component analysis
    :return: None
    """
    pca = PCA()

    data = pca.fit_transform([[22,23,24],[23,84,12],[22,74,54],[22,23,24],[22,84,12],[22,74,54],[22,23,24],[22,84,12],[22,74,54]])

    print(data)

if __name__ == '__main__':
    pca()
