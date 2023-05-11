# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 5 --n_runs 15 --n_impr 200 --data_sets MQ2008 --click_model perfect --algo tdmgd
# python ../graphs/makegraphs.py output1 ./average/MQ2008/algo/TDMGD/perfect/1000/10/0.1-0.99999977-1.out

# pythonrap run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 15 --n_runs 15 --n_impr 200 --data_sets MQ2008 --click_model perfect --algo tddbtr
# python ../ghs/makegraphs.py output1 ./average/MQ2008/algo/TDDBTR/perfect/200/10/0.1-0.99999977-1.out

# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 10 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model perfect --algo pdbtr
# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 10 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model info --algo pdbtr
# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 10 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model navi --algo pdbtr

# python ../graphs/makegraphs.py output1 ./average/MQ2008/algo/PDBTR/navi/1000/10/0.1-0.99999977-1.out

# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 15 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model perfect --algo dbgd
# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 15 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model info --algo dbgd
# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 15 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model navi --algo dbgd

# python ../graphs/makegraphs.py output1 ./average/MQ2008/algo/PDBTR/navi/1000/10/0.1-0.99999977-1.out
# python ../graphs/makegraphs.py output1 ./average/MQ2008/algo/PDBTR/navi/1000/10/0.1-0.99999977-1.out

# python run_script.py --output_folder ./output2 --average_folder ./average2 --log_folder ./logs --n_proc 5 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model perfect --algo pdbtr
# python run_script.py --output_folder ./output2 --average_folder ./average2 --log_folder ./logs --n_proc 5 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model info --algo pdbtr
# python run_script.py --output_folder ./output2 --average_folder ./average2 --log_folder ./logs --n_proc 5 --n_runs 15 --n_impr 1000 --data_sets MQ2008 --click_model navi --algo pdbtr
# python run_script.py --output_folder ./output --average_folder ./average --log_folder ./logs --n_proc 10 --n_runs 15 --n_impr 400 --data_sets MQ2008 --click_model navi --algo pdbtr

# python ../graphs/makegraphs.py output1 ./average/MQ2008/algo/PDBTRDR/navi/400/10/0.1-0.99999977-1.out
# python ../graphs/makegraphs.py output1 ./average/MQ2008/algo/PDBTR/navi/1000/10/0.1-0.99999977-1.out

# python ../graphs/makegraphs.py output1 ./average/MQ2008/algo/TDMGD/navi/1000/10/0.1-0.99999977-1.out
# python ../graphs/makegraphs.py output2 ./average2/MQ2008/algo/PDBTR/navi/1000/10/0.1-0.99999977-1.out
python ../graphs/makegraphs.py --plot_name navi --output_files ./average2/MQ2008/algo/PDBTR/navi/1000/10/0.1-0.99999977-1.out ./average/MQ2008/algo/DBGD/navi/1000/10/0.1-0.99999977-1.out ./average/MQ2008/algo/TDMGD/navi/1000/10/0.1-0.99999977-1.out 
python ../graphs/makegraphs.py --plot_name perfect --output_files ./average2/MQ2008/algo/PDBTR/perfect/1000/10/0.1-0.99999977-1.out ./average/MQ2008/algo/DBGD/perfect/1000/10/0.1-0.99999977-1.out ./average/MQ2008/algo/TDMGD/perfect/1000/10/0.1-0.99999977-1.out 
python ../graphs/makegraphs.py --plot_name info --output_files ./average2/MQ2008/algo/PDBTR/info/1000/10/0.1-0.99999977-1.out ./average/MQ2008/algo/DBGD/info/1000/10/0.1-0.99999977-1.out ./average/MQ2008/algo/TDMGD/info/1000/10/0.1-0.99999977-1.out 