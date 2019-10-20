from sklearn.datasets import fetch_20newsgroups


news = fetch_20newsgroups(subset='all')


print(news.target)
print('')
# print(news.data)