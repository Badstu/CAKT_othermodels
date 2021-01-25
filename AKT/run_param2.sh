python -u main.py --dataset assist2009_updated --batch_size 128 --dropout 0 --d_model 256 --d_ff 128 --n_head 4
python -u main.py --dataset assist2015 --batch_size 128 --dropout 0 --d_model 256 --d_ff 128 --n_head 4
python -u main.py --dataset assist2017 --batch_size 32 --dropout 0 --d_model 256 --d_ff 128 --n_head 4
python -u main.py --dataset STATICS --batch_size 32 --dropout 0 --d_model 256 --d_ff 128 --n_head 4
python -u main.py --dataset synthetic --batch_size 128 --dropout 0 --d_model 256 --d_ff 128 --n_head 4
