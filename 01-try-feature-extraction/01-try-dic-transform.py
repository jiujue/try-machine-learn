from sklearn.feature_extraction import dict_vectorizer

def vectorizer():

    dict_vec = dict_vectorizer.DictVectorizer(sparse=False)

    data = dict_vec.fit_transform([{'name':'jiujue','age':10},{'name':'mmp','age':11},{'name':'sam','age':12}])

    print(dict_vec.get_feature_names())

    print(data)
    print(type(data))



    pass


if __name__ == '__main__':

    vectorizer()