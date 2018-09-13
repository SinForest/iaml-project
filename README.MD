# This is a very cool project, you know?

## the dataset

The 'fma'¹ dataset used in this project is located in [this](https://github.com/mdeff/fma) repo.\
It is a dump of the [Free Music Archive](http://freemusicarchive.org/), containing 106'574 songs in 161 categories (of which we only use those listed as main genres)².

¹: neither Full Metal Alchemist, nor Finanzmarktaufsicht\
²: someone \**puts finger at nose*\* has to evaluate how many exactly

## dataset class

Use `SoundfileDataset` from `./dataset.py`.  
The constructor needs a metadata file (@param 'path'). Either use the `tracks.csv` from the dataset or run `./read_csv.py` and use the resulting `all_metadata.p`, which runs a helluvalot faster.  
Other parameters:  
+ `ipath`: path to dataset, default should work, if linked
+ `seg_size`: size (in seconds) of the random crops of the data
+ `hotvec`: set to `True`, if labels should be returned one-hot-encoded
+ `cut_data`: set to `True`, if data is from a trimmed dataset (each song is only 30s instead of its original duration)

## Importanté: link the dataset and the metadata

do this via:  
`cd <THIS REPO>`  
`ln -s <DATASET PATH> dataset.ln`  
`ln -s <METADATA PATH> metadata.ln`  
with:  
`<DATASET PATH>`  := absolute path to dataset, e.g. `<DATASET PATH>/000/000002.mp3` should exist  
`<METADATA PATH>` := absolute path to metadata, e.g. `<METADATA PATH>/tracks.csv` should exist