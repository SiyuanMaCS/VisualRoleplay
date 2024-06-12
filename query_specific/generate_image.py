from PIL import Image, ImageFont, ImageDraw

from diffusers import DiffusionPipeline
import torch
import os
import time
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np


font_size = 60
font = ImageFont.truetype('../assets/ARIAL.TTF', font_size)
max_width = 1024


def get_index(question, dataset):
    dataset_path = f"../datasets/{dataset}.csv"
    df = pd.read_csv(dataset_path,index_col=0)
    df = df.reset_index()
    df = df.set_index("redteam_query")
    index = df.index.get_loc(question)
    return index

def typo_format_text(text):
    font_size = 60
    font = ImageFont.truetype('../assets/ARIAL.TTF', font_size)
    max_width = 1024
    text = text.strip()
    img = Image.new('RGB', (max_width, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    word_list = text.split(" ")
    word_num = len(word_list)
    formated_text = word_list[0]
    cur_line_len = draw.textlength(formated_text, font=font)
    line_num = 1
    for i in range(1, word_num):
        cur_line_len += draw.textlength(" "+word_list[i], font=font)
        if cur_line_len < max_width:
            formated_text += " "+word_list[i]
        else:
            formated_text += "\n "+word_list[i]
            cur_line_len= draw.textlength(" "+word_list[i], font=font)
            line_num += 1
    return formated_text, line_num

def typo_draw_img(formated_text, line_num, img_path):
    font_size = 60
    max_width = 1024
    max_height = 1024
    max_height = font_size * (line_num + 1)
    img = Image.new('RGB', (max_width, max_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, font_size/2.0), formated_text, (0, 0, 0), font=font)
    img.save(img_path)


def vertical_concat(image1, image2, image3):
    if os.path.exists(image3):
        return
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    width1, height1 = img1.size
    width2, height2 = img2.size

    result_width = max(width1, width2)
    result_height = height1 + height2
    result = Image.new('RGB', (result_width, result_height))

    result.paste(img1, (0, 0))
    result.paste(img2, (0, height1))
    result.save(image3)
    print(image3)

def generate_character_sd_img(roles_path, image_root, dataset="redteam2k"):

      
    os.makedirs(image_root, exist_ok=True)
    roles_df = pd.read_csv(roles_path,index_col=0)


    SD_base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
    SD_base.to("cuda")

    SD_refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=SD_base.text_encoder_2,
        vae=SD_base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    SD_refiner.to("cuda")

    roles_df = pd.read_csv(roles_path, index_col=0)

    os.makedirs(f"{image_root}/{dataset}/", exist_ok=True)
          
    for _ , role in roles_df.iterrows():
        
        i = get_index(role['redteam_query'], dataset)
        if os.path.exists(f"{image_root}/{dataset}/character_sd_{i}.png"):
            continue
        prompt = role['diffusion prompt']

        image_base = SD_base(
            prompt=prompt,
            num_inference_steps=30,
            height=1024, width=1024,
            denoising_end=0.8,
            output_type="pil",
        ).images

        image = SD_refiner(
            prompt=prompt,
            num_inference_steps=30,
            height=1024, width=1024,
            denoising_start=0.8,
            image=image_base,
        ).images[0]

        image.save(f"{image_root}/{dataset}/character_sd_{i}.png")
        
        
def generate_question_img(roles_path, image_root, dataset="redteam2k"):

    os.makedirs(image_root, exist_ok=True)
    roles_df = pd.read_csv(roles_path, index_col=0)
    for _ , role in roles_df.iterrows():
        i = get_index(role['redteam_query'], dataset)
        if os.path.exists(f"{image_root}/{dataset}/question_img_{i}.png"):
            continue
        question_img_path = f"{image_root}/{dataset}/question_img_{i}.png"
        formated_text, line_num = typo_format_text(role['redteam_query'].replace("\n", " "))
        typo_draw_img(formated_text, line_num, question_img_path)

def generate_character_description_img(roles_path, image_root, dataset="redteam2k"):      
    os.makedirs(image_root, exist_ok=True)
    roles_df = pd.read_csv(roles_path,index_col=0)
    for _ , role in roles_df.iterrows():
        i = get_index(role['redteam_query'], dataset)
        typo_r_path = f"{image_root}/{dataset}/character_description_{i}.png"
        #print(item['brief description'])
        if os.path.exists(typo_r_path):
            continue
        formated_text, line_num = typo_format_text(role['brief description'].replace("\n", " "))
        typo_draw_img(formated_text, line_num, typo_r_path)


def generate_vrp_img(roles_path, image_root,dataset="redteam2k"):

   
    os.makedirs(image_root, exist_ok='True')
    roles_df = pd.read_csv(roles_path,index_col=0)
    
    for _ , role in tqdm(roles_df.iterrows(), total=len(roles_df)):
        i = get_index(role['redteam_query'], dataset)    
        
        character_sd_img = f"{image_root}/{dataset}/character_sd_{i}.png"    
        character_description_img = f"{image_root}/{dataset}/character_description_{i}.png"
        question_img = f"{image_root}/{dataset}/question_img_{i}.png"
        character_img = f"{image_root}/{dataset}/character_{i}.png"
        vrp_img = f"{image_root}/{dataset}/vrp_{i}.png"
        if os.path.exists(vrp_img):
            continue
        vertical_concat(character_description_img, character_sd_img, character_img)
        vertical_concat(character_img, question_img, vrp_img)
                
def parse_int_list(input_str):
    return [int(item) for item in input_str.split(',')]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='描述你的脚本的目的')

    # 添加命令行参数
    parser.add_argument('--dataset', type=str, default="redteam2k")
    parser.add_argument('--image_root', type=str, default="./image")
    parser.add_argument('--roles_root', type=str, default="./roles")

    args = parser.parse_args()
    dataset = args.dataset
    roles_root = args.roles_root
    image_root = args.image_root
    roles_path = f"{roles_root}/{dataset}/roles.csv"
    os.makedirs(f"{image_root}/{dataset}/", exist_ok=True)

    generate_character_description_img(roles_path, image_root, dataset)
    generate_character_sd_img(roles_path, image_root, dataset)
    generate_question_img(roles_path, image_root, dataset)
    
    generate_vrp_img(roles_path, image_root, dataset)


       
    