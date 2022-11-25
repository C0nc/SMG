for ppi in 0 
do
    for inductive_ppi in 1 2 3 4 5
    do
        python /data/guest/GraphMAE/main_transductive.py  --ppi=$ppi --inductive_ppi=$inductive_ppi 
    done
done




