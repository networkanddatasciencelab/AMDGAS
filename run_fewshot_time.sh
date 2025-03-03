cd /home/imss/syn/SearchNet

python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.7' --ratio 0.1 > experiment_fewshot/time/sp07_ratio01.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.7' --ratio 0.25 > experiment_fewshot/time/sp07_ratio025.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.7' --ratio 1.0 > experiment_fewshot/time/sp07_ratio1.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.8' --ratio 0.1 > experiment_fewshot/time/sp08_ratio01.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.8' --ratio 0.25 > experiment_fewshot/time/sp08_ratio025.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.8' --ratio 1.0 > experiment_fewshot/time/sp08_ratio1.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.9' --ratio 0.1 > experiment_fewshot/time/sp09_ratio01.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.9' --ratio 0.25 > experiment_fewshot/time/sp09_ratio025.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sp' --sp_b '0.9' --ratio 1.0 > experiment_fewshot/time/sp09_ratio1.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'hiv' --sp_b '0.7' --ratio 0.1 > experiment_fewshot/time/hiv_ratio01.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'hiv' --sp_b '0.7' --ratio 0.25 > experiment_fewshot/time/hiv_ratio025.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'hiv' --sp_b '0.7' --ratio 1.0 > experiment_fewshot/time/hiv_ratio1.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sider' --sp_b '0.7' --ratio 0.1 > experiment_fewshot/time/sider_ratio01.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sider' --sp_b '0.7' --ratio 0.25 > experiment_fewshot/time/sider_ratio025.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'sider' --sp_b '0.7' --ratio 1.0 > experiment_fewshot/time/sider_ratio1.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'bace' --sp_b '0.7' --ratio 0.1 > experiment_fewshot/time/bace_ratio01.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'bace' --sp_b '0.7' --ratio 0.25 > experiment_fewshot/time/bace_ratio025.txt
wait
python graph_search.py --seed 4211 --jieduan 4 --useac True --device '1' --dataset_str 'bace' --sp_b '0.7' --ratio 1.0 > experiment_fewshot/time/bace_ratio1.txt
wait

