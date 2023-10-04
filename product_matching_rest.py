from flask import Flask
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection, 
DropHighPSIFeatures
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
import faiss
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)


global clf, pipe, idx_l2


# функция генерации датафреймов для кандадатов к данной записи
def make_df(target, el, train_id):
    dfo = pd.DataFrame()
    for r in el:  # iterate ove candidates
        dftmp = pd.DataFrame(df_base2[[r]], index=[train_id])
        dftmp['tgt'] = int(target==base_index[r])
        dfo = pd.concat((dfo, dftmp), axis=0)
    return dfo


@app.route("/match/<query>")
def match(query):
    
    query_df = pd.DataFrame(np.array(query).reshape(1,-1), columns=range(72))
    query_arr = pipe.transform(req_df)
    
    vecs, idx_cnd = idx_l2.search(np.ascontiguousarray(query_arr).astype('float32'), 20)        
    
    #  датафрейм кандидатов
    df_clf = pd.DataFrame(vecs, index=[0]*vecs.shape[0])
    targets = np.array([base_index[x] for x in idx_cnd])
    # датафрейм запросов
    df_clf0 = pd.DataFrame(query_arr, index=[0])
    # объединенный датафрейм
    df_clf = df_clf.join(df_clf0, lsuffix='c', rsuffix='q')
    
    preds = clf.predict_proba(df_clf)[:, 1]                  
    
    # возвращаем идентификаторы топ-5 кандидатов, начиная с конца, поскольку argsort делает прямую сортировку
    return {'match':targets[np.argsort(preds)[::-1][:5]]}


if __name__ == '__main__':
    
    with open('./models/pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
        
    with open('./models/clf2.pkl', 'rb') as f:
        clf = pickle.load(f)
                                
    with open('./models/base_index.pkl', 'rb') as f:
        base_index = pickle.load(f)
        
    idx_l2 = faiss.read_index("./models/trained_block.index")
    idx_l2.nprobe = 192
    
    app.run()
    