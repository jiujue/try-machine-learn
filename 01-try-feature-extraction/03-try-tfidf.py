from sklearn.feature_extraction.text import TfidfVectorizer
import jieba



def get_divided_word():

    con1 = jieba.cut('人生苦短，我用python，人生漫长，我用c++')
    con2 = jieba.cut('php是世界上最好的语言，没有之一')
    con3 = jieba.cut('学c头发掉的快，掉的早，嘿嘿')

    c1 = str(list(con1))
    c2 = str(list(con2))
    c3 = str(list(con3))

    return c1,c2,c3

def tfidf_vec():

    tf = TfidfVectorizer()

    c1,c2,c3 = get_divided_word()
    print(c1,c2,c3)

    data = tf.fit_transform([c1,c2,c3])

    print(tf.get_feature_names())  
    
    print(data.toarray())


    pass

if __name__ == '__main__':
    tfidf_vec()

