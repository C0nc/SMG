for dataset in IMDB-MULTI # PROTEINS IMDB-BINARY 
do
    python main_graph.py --use_cfg --dataset=$dataset --activation='relu'  
done