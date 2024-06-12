#vllm
#pillow
#diffusers
#pandas
import torch
from PIL import Image, ImageFont, ImageDraw
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import AutoModel, CLIPImageProcessor
from transformers import pipeline
import transformers

import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd



def main(args):
    # Open image folder

    save_path = args.save_path
    data_path = args.data_path
    dataset = args.dataset
    model_name = args.model_name
    data_file = args.data_file

    image_root = args.image_root

    os.makedirs(f"{save_path}/{model_name}/{dataset}/", exist_ok=True)

    
    
    query_df = pd.read_csv(f"{data_path}/{dataset}/{data_file}.csv")

    print("Generating "+f"{save_path}/{model_name}/{dataset}/{data_file}.csv")
    batch_query_text = query_df["jailbreak_query"]
    batch_image_path =  [ f"{image_root}/{image_file}" for image_file in query_df["jailbreak_query_image"]] 


    df_save = query_df
    batch_response = [None] * len(batch_image_path) 
    

    if model_name == 'llava-hf/llava-v1.6-mistral-7b-hf':
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        model = LlavaNextForConditionalGeneration.from_pretrained(f"llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        model.to("cuda")
        
    
        for index, (image_path, prompt) in enumerate(tzip(batch_image_path, batch_query_text)):
            prompt = f"[INST] <image>\n{prompt}[/INST]"
            image = Image.open(image_path)
            inputs = processor(prompt, image, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=512)
            response = processor.decode(output[0], skip_special_tokens=True)
            batch_response[index] = response
            query_df["response"] = batch_response
            print(response)
        query_df.to_csv(f"{save_path}/{model_name}/{dataset}/{data_file}.csv")
    elif model_name == "Qwen/Qwen-VL-Chat":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

        for index, (image_path, prompt) in enumerate(tzip(batch_image_path, batch_query_text)):
            response = model.chat(tokenizer, query=f'<img>{image_path}</img>{prompt}', history=None)
            batch_response[index] = response[0]
            query_df["response"] = batch_response
            print(response[0])
        query_df.to_csv(f"{save_path}/{model_name}/{dataset}/{data_file}.csv")
        
    else:
        print("Model name error!")
        return
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_file", type=str, default="vrp")
    parser.add_argument('--dataset', type=str, default="redteam2k_test")
    parser.add_argument("--data_path", type=str, default="./prompt")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--image_root", type=str, default="./image")
    args = parser.parse_args()


    main(args)
    