from csv import reader
import pickle
from tqdm import tqdm

CSV_PATH = "/home/cobalt/datasets/fma/fma_metadata/tracks.csv"
PICKLE_PATH = "./all_metadata.p"

def read_dict(path):
    """
    reads the complete metadata dict for all files from a .csv-file
    """
    read = reader(open(path), delimiter=',')
    head = list(zip(tuple(next(read)), tuple(next(read))))
    next(read) #empty line

    res = {}
    for i, item in enumerate(tqdm(read)):
        tmp_d = make_dict(head, item)
        res[tmp_d['id']] = tmp_d
    
    return res

def make_dict(header, data):
    """
    converts metadata for a single file to a dict
    called by read_dict
    """
    d = {}
    for i, sub in enumerate(header):
        if sub == ('',''): 
            d['id'] = int(data[i])
            id = f"{d['id']:06d}"
            d['path'] = f"{id[:3]}/{id}.mp3"
            continue
        if sub[0] not in d:
            d[sub[0]] = {}
        d[sub[0]][sub[1]] = data[i]
    return d

if __name__ == "__main__":
    
    res = read_dict(CSV_PATH)

    for k1, v1 in res[2].items():
        if k1 == 'id':
            print(f"##### id: {v1} #####")
        elif k1 == 'path':
            continue
        else:
            for k2, v2 in v1.items():
                print(f"{k1}:{k2} => {v2}")

    pickle.dump(res, open(PICKLE_PATH, "wb"))