ca /home/xueying/anaconda3/envs/jaeger
cd src/jaeger/train
python jaeger_train.py --csv_file /mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/JAEGER/models/training_data/Novartis_GNF_cleaned.csv  --assay_id test_1104 --num_threads 24 --use_qualified True
# python jaeger_validate.py --csv_file /mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/JAEGER/models/training_data/Novartis_GNF_cleaned.csv  --assay_id test_1104