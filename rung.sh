for dataset in IMDB-BINARY IMDB-MULTI PROTEINS MUTAG REDDIT-BINARY COLLAB NCI1
do
    python main_graph.py --use_cfg --dataset=$dataset --activation='relu'
done