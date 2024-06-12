
for i in $(seq 0 4); do
   python generate_universal_character.py --round=$i --gpus=2 --model_name='Qwen/Qwen-VL-Chat'
   ray stop
done