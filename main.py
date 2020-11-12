from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input,Dense,Dropout,Concatenate

from keras import backend as K

def custom_activation(x):
    return  1 + 4/(1+K.exp(-x))
    # return (K.sigmoid(x) * 5) - 1

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true,y_pred)


def get_embed_matrix(word_embed_file):
    embeddings_index = {}
    f = open(word_embed_file,encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index



def read_data(text):
    examples = [s.strip() for s in text]
    # Split by words
    x_text = [s.split(" ") for s in examples]
    max_len = max(len(x) for x in x_text)
    return np.array(x_text), max_len


def sent2vec(s,wef):
    embeddings_index = get_embed_matrix(wef)
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


def like2vec(s,like_emb_mat):

    M = []
    for w in s:
        try:
            M.append(like_emb_mat[:,int(w) - 1])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(100)
    return v / np.sqrt((v ** 2).sum())
    # return v / M.shape[0]

def load_data(data_file,word_embedding_file,like_embedding_file):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    rawdata = pd.read_table(data_file, encoding='utf-8', sep="\t")
    status = np.array(rawdata.iloc[:, 7])
    likeset = np.array(rawdata.iloc[:,6])

    ope_lable = np.array(rawdata.iloc[:,1])
    con_lable = np.array(rawdata.iloc[:,2])
    ext_lable = np.array(rawdata.iloc[:,3])
    agr_lable = np.array(rawdata.iloc[:,4])
    neu_lable = np.array(rawdata.iloc[:,5])
    userid = np.array(rawdata.iloc[:,0])

    statusVec = np.array([sent2vec(statu,word_embedding_file) for statu in status])

    like_emb_mat = np.array(pd.read_table(like_embedding_file, header=None, encoding='utf-8',sep='\t'))

    likes, maxlike = read_data(likeset)

    likesVec = np.array([like2vec(statu,like_emb_mat) for statu in likes])


    userid_train,userid_test,text_train,text_test,like_train,like_test,ope_lable_train,ope_lable_test, con_lable_train, con_lable_test,\
    ext_lable_train,ext_lable_test, agr_lable_train,agr_lable_test, neu_lable_train,neu_lable_test = train_test_split(
        userid,statusVec, likesVec,ope_lable, con_lable, ext_lable, agr_lable, neu_lable,test_size=0.2,random_state=42)

    return  userid_train,userid_test,text_train,text_test,like_train,like_test,ope_lable_train,ope_lable_test, con_lable_train, con_lable_test,\
    ext_lable_train,ext_lable_test, agr_lable_train,agr_lable_test, neu_lable_train,neu_lable_test


def DNNmodel():
    input = Input(shape=(300,))
    input2 = Input(shape=(100,))
    merged_tensor = Concatenate(axis=1)([input, input2])
    h1 = Dense(units=500, activation='tanh')(merged_tensor)
    d1 = Dropout(rate=0.2)(h1)

    h2 = Dense(units=300, activation='tanh')(d1)
    d2 = Dropout(rate=0.2)(h2)
    h3 = Dense(units=200, activation='tanh')(d2)
    d3 = Dropout(rate=0.2)(h3)
    h4 = Dense(units=100, activation='tanh')(d3)
    d4 = Dropout(rate=0.2)(h4)

    ope_out = Dense(units=1,activation=custom_activation,name='ope')(d4)
    con_out = Dense(units=1,activation=custom_activation,name='con')(d4)
    ext_out = Dense(units=1,activation=custom_activation,name='ext')(d4)
    agr_out = Dense(units=1,activation=custom_activation,name='agr')(d4)
    neu_out = Dense(units=1,activation=custom_activation,name='neu')(d4)
    model =Model(inputs=[input,input2],outputs=[ope_out,con_out,ext_out,agr_out,neu_out])
    return model

def resTran(x):
    c = []
    for i in x:
        for j in i:
            c.append(j)
    return c

if __name__=='__main__':
    data_file = "data/raw_data"
    word_embed_file = "data/glove.6B.300d.txt"
    like_embed_file = "data/item.100d.txt"



    userid_train, userid_test,text_train, text_test, like_train,like_test,ope_lable_train, ope_lable_test, con_lable_train, con_lable_test,\
    ext_lable_train, ext_lable_test, agr_lable_train, agr_lable_test, neu_lable_train, neu_lable_test = load_data(data_file,word_embed_file,like_embed_file)


    model = DNNmodel()
    model.compile(optimizer='adam',loss=['mse','mse','mse','mse','mse'],loss_weights=[0.2,0.2,0.2,0.2,0.2])

    history=model.fit([text_train,like_train],[ope_lable_train,con_lable_train, ext_lable_train,agr_lable_train,neu_lable_train],
                      epochs=50, batch_size=256, validation_split=0.1)
    # model evaluate  result
    ope_pred, con_pred, ext_pred, agr_pred, neu_pred = model.predict([text_test,like_test])
    print("ope MAE:", MAE(ope_lable_test,ope_pred))
    print("con MAE:", MAE(con_lable_test,con_pred))
    print("ext MAE:", MAE(ext_lable_test,ext_pred))
    print("agr MAE:", MAE(agr_lable_test,agr_pred))
    print("neu MAE:", MAE(neu_lable_test,neu_pred))
