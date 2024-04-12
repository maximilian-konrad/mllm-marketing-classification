#%% - Imports
from aleph_alpha_client import Client, Prompt, Image, CompletionRequest, CompletionResponse, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation
import numpy as np
import os
import pandas as pd
import re
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image as img
import requests
from datetime import datetime

#%% - Prepare Luminous
client = Client(token="TBD") # Insert Aleph Alpha API Key

#%% - Prepare InstructBLIP
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


#%% - Define run model functions
# Run Luminous
def run_Aleph_Alpha_Luminous(file_path, prompt):
    model_input = [Image.from_image_source(file_path), prompt]
    model_request = CompletionRequest(prompt=Prompt(model_input), maximum_tokens=50, temperature=0)
    model_response = client.complete(request=model_request, model="luminous-extended").completions[0].completion
    return model_response

# Run InstructBLIP
def run_Salesforce_InstructBLIP(file_path, prompt):
    image = img.open(file_path).convert("RGB")
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=50,
        min_length=1,
        #top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        #temperature=0,
    )
    outputs[outputs == 0] = 2
    model_response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return model_response

#%% - Helper functions
def df_to_excel(df: pd.DataFrame, file_name: str):
    now = datetime.now()
    short_format_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    path = r"TBD" + "\\" + file_name + "_" + short_format_date_time + ".xlsx" # Insert output path
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False)

def run_inference(df, luminous: bool, instructblip: bool, llava: bool):
    total_rows = len(df)
    
    # Iterating through all rows in the dataframe
    for index, row in df.iterrows():
        # Print progress
        print(f"Processing row {index+1} of {total_rows}...")
        
        # Saving values from columns "file_path" and "prompt" into variables
        file_path = row['file_path']
        prompt = row['prompt']
        
        # Running the models and saving their outputs in dedicated column
        if luminous:
            prompt_luminous = 'Q:' + prompt + ' A:'
            # Aleph Alpha Luminous 
            try:
                inference_output = run_Aleph_Alpha_Luminous(file_path, prompt_luminous)
            except Exception as e:
                print(f"An error occurred: {e}")
                inference_output = "n/a"
            df.at[index, 'output_Luminous'] = inference_output
        
        if instructblip:
            # Salesforce InstructBLIP
            try:
                inference_output = run_Salesforce_InstructBLIP(file_path, prompt)
            except Exception as e:
                print(f"An error occurred: {e}")
                inference_output = "n/a"
            df.at[index, 'output_InstructBLIP'] = inference_output
    
    # Saving the modified dataframe to an Excel file using your custom function
    df_to_excel(df, "output")


#%% - Main function
# Import XLS with prompts
df1 = pd.read_excel(r"TBD", sheet_name=0) # Insert path to input Excel - Excel needs to contain columns "file_path" and "prompt"
run_inference(df1, luminous = True, instructblip = True, llava = False)
# %%
