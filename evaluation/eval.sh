
accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1
