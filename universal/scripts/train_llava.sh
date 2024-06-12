for i in $(seq 0 4); do
   python generate_universal_character.py --round=$i --gpus=2 --model_name='llava-hf/llava-v1.6-mistral-7b-hf'
   ray stop
done