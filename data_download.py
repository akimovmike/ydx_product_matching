import zipfile
import requests
from urllib.parse import urlencode
import os
import pickle
import faiss
import pandas as pd


if __name__ == '__main__':
    
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/BBEphK0EHSJ5Jw'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    if not os.path.exists('./content/data.zip'):
        download_response = requests.get(download_url)
        with open('./content/data.zip', 'wb') as f:
            f.write(download_response.content)

        zip_path = './content/data.zip'

        # Распаковка zip-архива
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()

    with open('./models/pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
        
    df_base = pd.read_csv("./content/base.csv", index_col=0, dtype=dict_base)
    df_base2 = pipe.transform(df_base)
    
    # изготавливаем индекс товаров
    base_index = {k: v for k, v in enumerate(df_base.index.to_list())}
    with open('./models/base_index.pkl', 'wb') as f:
        pickle.dump(base_index, f)
    
    # параметры FAISS
    dims = df_base2.shape[1]
    n_cells = 3416
    nprobe = 192

    res = faiss.StandardGpuResources()
    # return all gpu memory after use
    res.noTempMemory()

    # подготовка индекса FAISS
    quantizer2 = faiss.GpuIndexFlatL2(res, dims)
    idx_l2 = faiss.GpuIndexIVFFlat(res, quantizer2, dims, int(n_cells))
    idx_l2.train(np.ascontiguousarray(df_base2).astype('float32'))
    idx_l2.add(np.ascontiguousarray(df_base2).astype('float32'))
    idx_l2.nprobe = int(nprobe)

    # переносим индекс на cpu для сохранения
    idx_l2cpu = faiss.index_gpu_to_cpu(idx_l2)
    
    faiss.write_index(idx_l2cpu, "./models/trained_block.index") 
    
    