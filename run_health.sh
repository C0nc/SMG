for ppi in 0 1 2 3 4 5
do
    python /data/guest/GraphMAE/main_transductive.py  --ppi=$ppi --health --lr_f=0.001 --inductive_ppi=-1
done




