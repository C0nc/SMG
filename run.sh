for ppi in 0 1 2 3 4 5
do
    python /home/yancui/ppimae/main_transductive.py  --ppi=$ppi   --lr_f=0.01 --weight_decay_f=0.001 --residual --inductive_ppi=-1 --expression
done




