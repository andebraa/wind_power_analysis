import pandas as pds
import numpy as num
import matplotlib.pyplot as plot
import seaborn as sb 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load
from scipy.sparse import save_npz, load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as num
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()
data=pd.read_csv("Test.csv")
data.head()

fig=plt.figure(figsize=(5,5))
clrs=["green",'red']
positive=data[data['label']==1]
negative=data[data['label']==0]
cv=[positive['label'].count(),negative['label'].count()]
piechart=plot.pie(cv,labels=["Positive","Negative"],
                 autopct ='%1.1f%%',
                 shadow = True,
                 colors = clrs,
                 startangle = 45,
                 explode=(0, 0.1))
dataframe=["Hello master, Dont carry the world upon your shoulders for well you know that its a fool who plays it cool by making his world a little colder Na-na-na,a, na"]
bagview=count.fit_transform(dataframe)
print(count.get_feature_names())
print(bagview.toarray())


system('wget "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"')
system('tar -xzf "aclImdb_v1.tar.gz"')

def create_data_frame(folder: str) -> pd.DataFrame:
  
    positive_folder = f'{folder}/pos' 
    negative_folder = f'{folder}/neg' 
    
    def get_files(fld: str) -> list:
        
        return [join(fld, f) for f in listdir(fld) if isfile(join(fld, f))]
    
    def append_files_data(data_list: list, files: list, label: int) -> None:
        for file_path in files:
            with open(file_path, 'r') as f:
                text = f.read()
                data_list.append((text, label))
    
    positive_files = get_files(positive_folder)
    negative_files = get_files(negative_folder)
    
    data_list = []
    append_files_data(data_list, positive_files, 1)
    append_files_data(data_list, negative_files, 0)
    shuffle(data_list)
    
    text, label = tuple(zip(*data_list))
    textfile = list(map(lambda txt: re.sub('(<br\s*/?>)+', ' ', txt), text))
    
    return pd.DataFrame({'text': text, 'label': label})



imdb_train = create_data_frame('aclImdb/train')
imdb_test = create_data_frame('aclImdb/test')

system("mkdir 'csv'")
imdb_train.to_csv('csv/imdb_train.csv', index=False)
imdb_test.to_csv('csv/imdb_test.csv', index=False)


system("mkdir 'data_preprocessors'")
system("mkdir 'vectorized_data'")

unigram_v
unigram_vectorizers.fit(imdb_train['text'].values)

dump(unigram_vectorizers, 'data_preprocessors/unigram_vectorizer.joblib')

x_train_unigrams = unigram_vectorizers.transform(imdb_train['text'].values)

save_npz('vectorized_data/X_train_unigram.npz', x_train_unigrams)

# Unigram Tf-Idf

unigram_tf_idf_transformer = TfidfTransformer()
unigram_tf_idf_transformer.fit(x_train_unigrams)

dump(unigram_tf_idf_transformer, 'data_preprocessors/unigram_tf_idf_transformer.joblib')
x_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(x_train_unigrams)

save_npz('vectorized_data/x_train_unigram_tf_idf.npz', x_train_unigram_tf_idf) load_npz('vectorized_data/X_train_unigram_tf_idf.npz')


# Bigram Counts

bigram_vectorizers = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizers.fit(imdb_train['text'].values)

dump(bigram_vectorizers, 'data_preprocessors/bigram_vectorizers.joblib')

x_train_bigram = bigram_vectorizers.transform(imdb_train['text'].values)

save_npz('vectorized_data/x_train_bigram.npz', x_train_bigram)


# Bigram Tf-Idf

bigram_tf_idf_transformer = TfidfTransformer()
bigram_tf_idf_transformer.fit(X_train_bigram)

dump(bigram_tf_idf_transformer, 'data_preprocessors/bigram_tf_idf_transformer.joblib')


x_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(x_train_bigram)

save_npz('vectorized_data/x_train_bigram_tf_idf.npz', x_train_bigram_tf_idf)

###############################################################################################################################


def train_and_show_scores(x: csr_matrix, y: np.array, title: str) -> None:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, train_size=0.75, stratify=y
    )

    classifier = SGDClassifier()
    classifier.fit(x_train, y_train)
    train_score = classifier.score(x_train, y_train)
    valid_score = classifier.score(x_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')

y_train = imdb_train['label'].values

train_and_show_scores(x_train_unigrams, y_train, 'Unigram Counts')
train_and_show_scores(x_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(x_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(x_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')


#################################################################################################################################

data = []
datalabels = []
with open("positive_tweets.txt") as f:
    for i in f:
        data.append(i)
        datalabels.append('positive')

with open("negative_tweets.txt") as f:
    for i in f:
        data.append(i)
        datalabels.append('negative')
vectorizers = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizers.fit_transform(
    data
)
features_nd = features.toarray()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test  = train_test_split(
        features_nd,
        datalabels,
        train_size=0.80,
        random_state=1234)
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()
logreg_model = logreg_model.fit(X=x_train, y=y_train)
y_pred = logreg_model.predict(x_test)
import random as rand
j = rand.randint(0,len(x_test)-7)
for i in range(j,j+7):
    print(y_pred[0])
    index = features_nd.tolist().index(x_test[i].tolist())
    print(data[index].strip())
