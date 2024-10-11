ca /home/xueying/anaconda3/envs/jaeger
cd src/jaeger/train
# python jaeger_train.py --csv_file /mnt/disk1/xueying/JAEGER/models/training_data/Novartis_GNF.csv  --assay_id Novartis_GNF --num_threads 24 --use_qualified True
python jaeger_validate.py --csv_file /mnt/disk1/xueying/mol-gen/JAEGER/models/training_data/Novartis_and_GDI_7817.csv  --assay_id Novartis_and_GDI_7817