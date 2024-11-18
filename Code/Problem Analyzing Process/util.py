from openai import OpenAI
from pydantic import BaseModel
import base64
import os
import json
import matplotlib.pyplot as plt

client = OpenAI()
using_model = 'gpt-4o-mini'
descriptive_catalogs = []
idea_catalogs = []
statistical_dictionary = {}
train_data = []
descriptive_feature_stats = {}
output_format = {
    "Problem_title": "The title extracted from the first line of the problem image.",
    "Descriptive_feature": "The descriptive catalog this problem belongs to. Use an existing catalog from the provided list, or create a new concise and specific one if necessary.",
    "Problem_idea_catalog": "The algorithm or idea catalog this problem relates to. Use an existing catalog from the provided list, or create a new concise and specific one if necessary."
}
train_dataset = "./dataset/train.jsonl"
image_main_directory = "./data"
result_log_directory = "./logs"
statistical_image_directory = "./logs/image"
overall_stats_file_path = "./logs/overall_stats.json"
descriptive_based_stats_file_path = "./logs/descriptive_based_stats.json"

class Problem_analysis(BaseModel): #Define the response format for ChatGPT
    Problem_title: str
    Descriptive_feature: str
    Problem_idea_catalog: str

def read_image_sub_directory(): #Search and get the full directory list for the images
    return os.listdir(image_main_directory)

def encode_image(image_path): #Encode the image for ChatGPT to understand
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def set_default_setting_and_prompt(): #Generate the dynamic prompt with all the things I need
    default_setting = "You are a very helpful algorithm expert. And you are taking an International Collegiate Programming Contest."
    prompt = (
        f"Your task is to read the problem images I send and classify each problem into two types of catalogs: "
        f"1) *Descriptive Feature Catalog*: Use an existing catalog from this list {str(descriptive_catalogs)}, or create a new concise and specific catalog if none fit. "
        f"2) *Problem Idea Catalog*: Use an existing catalog from this list {str(idea_catalogs)}, or create a new concise and specific catalog if none fit. "
        f"Output your results in the following format: {output_format}. "
        f"Focus on providing clear, accurate classifications and creating new catalogs only when necessary."
    )
    return default_setting, prompt

def using_gpt_to_classify(encoded_image): #Chat Completion with ChatGPT and get the output in the supposed format
    default_setting, prompt = set_default_setting_and_prompt()
    completion = client.beta.chat.completions.parse(
        model=using_model,
        messages=[
            {"role": "user", "content": default_setting},
            {"role": "system", "content": [{"text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]}
        ],
        response_format=output_format
    )
    return completion.choices[0].message.parsed

def process_gpt_output(original_response, id): #Process and store the output as data and get some statistical output
    formatted_response = dict(original_response)
    data = {'Id' : id, 'Title': formatted_response['Problem_title'], 'Descriptive_feature' : formatted_response['Descriptive_feature'], 'Idea & Algorithm' : formatted_response['Problem_idea_catalog']}
    train_data.append(data)
    statistical_dictionary[(formatted_response['Problem_idea_catalog'], formatted_response['Descriptive_feature'])] = statistical_dictionary.get((formatted_response['Problem_idea_catalog'], formatted_response['Descriptive_feature']), 0) + 1

def write_train_data(): #Store the asked data as the right for further usage of training a model or just for checking [train, validation] is both applicable
    with open(train_dataset, 'w', encoding='utf-8') as train_dataset_file:
        for data in train_data:
            json_line = json.dumps(data)
            train_dataset_file.write(json_line + '\n')

def show_overall_statistical_results(): #Show the statistical output in a graph way
    plt.bar(list(statistical_dictionary.keys()), list(statistical_dictionary.values()), color='#7799CC')
    plt.title('Category Statistics')
    plt.xlabel('Categories')
    plt.ylabel('Statistical number')
    plt.xticks(rotation=45)
    plt.show()

def write_overall_statistical_result(): #Write the overall statistical data into the log directory
    with open(overall_stats_file_path, 'w', encoding='utf-8') as overall_stats_file:
        overall_stats_file.write(json.dumps(statistical_dictionary))

def show_statistical_results_by_descriptive_feature(): #Show the statistical images with different descriptive_features
    for data in train_data:
        descriptive_feature = data['Descriptive_feature']
        problem_idea_catalog = data['Idea & Algorithm']
        descriptive_feature_stats[descriptive_feature] = descriptive_feature_stats.get(descriptive_feature, {})
        descriptive_feature_stats[descriptive_feature][problem_idea_catalog] = descriptive_feature_stats[descriptive_feature].get(problem_idea_catalog, 0) + 1
    write_statistical_log_json()
    for descriptive_feature, idea_algorithm_stats in descriptive_feature_stats.items():
        plt.figure(figsize=(10, 6))
        plt.bar(list(idea_algorithm_stats.keys()), list(idea_algorithm_stats.values()), color='#3388BB')
        plt.title(f'{descriptive_feature} - Algorithm Categories Statistics')
        plt.xlabel('Problem Idea Catalog')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

def write_statistical_log_json(): #Write all the descriptive features with seperated statistical data into the log
    with open(descriptive_based_stats_file_path, 'w', encoding='utf-8') as descriptive_based_stats_file:
        descriptive_based_stats_file.write(json.dumps(descriptive_feature_stats))