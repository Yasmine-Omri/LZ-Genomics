===TASK1:

HOW: 
- 1: download and unzip task1.zip into data/  (LINK: https://drive.google.com/drive/folders/1t3WJyGO4ewWWbWV7D2d1TL0YxDg8HsYX)

- 2:   cd run_scripts & ./task1.sh
- outputs: the report .txt in outputs/ and the .bin in best_spas/
NOTE:
- the max tree depth is set to 16
- the minimal sweep is used (including entropy for ensemble heuristic)
- The Ns are removed (however, note that they are more substantial here, over 800). Might want to think about deterministically replacing them?

===PRETRAIN (useful for tasks 2, 4, 5 + potentially reporting zero-shot for 1 + 3)
- 1: Download pretraining data (this should fetch both the 30GB train.txt and the small 100MB dev.txt):

pip install gdown
cd ../data/pretrain_data
gdown --id 1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f --no-cookies

- 2: update the path to the pretrain train.txt (30GB) in pretrain.sh. Review choices of max depth (chose 16), handle N (remove), and budget of pretrain data (chose 5GB). Feel free to update these flags depending on system and intuition.
Note: To attempt to improve memory (I/O and cachce) efficiency, the data I/O is streamed using 64MB chunks and passed into the tree using 4MB chunk. See these lines in the python_scripts/pretrain.py, feel free to update:
    '''
    read_chunk_bytes: int = 64 * 1024 * 1024,  # 64 MiB I/O
    train_block_len: int = 4 * 1024 * 1024,     # ~4 Mi symbols per train call
    '''

    cd run_scripts & ./pretrain.sh

- outputs: the report .txt in outputs/ and the .bin in best_spas/


====TASK3:
- 1: download and unzip task3.zip into data/  (LINK: https://drive.google.com/drive/folders/1t3WJyGO4ewWWbWV7D2d1TL0YxDg8HsYX)
- 2:   cd run_scripts & ./task3.sh
- outputs: the report .txt in outputs/ and the .bin in best_spas/
- same notes as task1 regarding max depth used and other hyperparams