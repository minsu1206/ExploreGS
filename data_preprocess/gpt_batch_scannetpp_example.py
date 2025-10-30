"""
Example3 : Batch API processing ; Multiple batch
"""

import glob
import os
import cv2
import base64
import json
from openai import OpenAI
from tqdm import tqdm
import time

os.environ['OPENAI_API_KEY'] = '' # REPLACE WITH YOUR OPENAI API KEY

client = OpenAI()

SYSTEM_MESSAGE_DICT = {
    "role": "system",
    "content": "You are a helpful assistant that can caption videos. "
                "This caption will be used as input for CLIP text encoder. "
                "Please describe the video frames in detail while keeping the max tokens under 70."
}

def create_prompt_message(frames):
    return [
        SYSTEM_MESSAGE_DICT,
        {
            "role": "user",
            "content": [
                "Please caption the following video frames: ",
                *map(lambda x: {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', x)[1]).decode()}"
                    }
                }, frames)
            ]
        }
    ]

if __name__ == "__main__":
    # Prepare frames from the video directory
    # root_dir = '/workspace/dataset/DL3DV'
    root_dir = '/workspace/dataset/scannetpp_dslr_only'
    json_root_dir = '/workspace/dataset/scannetpp_dslr_only'
    os.makedirs(json_root_dir, exist_ok=True)
    
    scenes = ["45b0dac5e3", "99fa5c25e1", "825d228aec", "927aacd5d1"]
    
    for scene in scenes:
        
        step = 1
    
        requests = []
        
        scene_dir = os.path.join(root_dir, scene, "dslr")
        
        train_test_json = os.path.join(scene_dir, "train_test_lists.json")
        with open(train_test_json, "r") as f:
            train_test_lists = json.load(f)
        
        train_set_name_list = sorted(train_test_lists["train"])
        # test_list = train_test_lists["test"]
        
        stride = len(train_set_name_list) // 15
        
        train_frame_path_list = []
        for train_set_name in train_set_name_list[::stride]:
            
            train_frame_path = os.path.join(scene_dir, "resized_undistorted_images", train_set_name)
            train_frame_path_list.append(train_frame_path)
        
        frames = []
        for train_frame_path in train_frame_path_list:
            frame = cv2.imread(train_frame_path)
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5) # 0.5 scale. pass fx fy
            frames.append(frame)
        
        prompt_messages = create_prompt_message(frames)
        
        request_data = {
            "model": "gpt-4o",          # Make sure this model is accessible
            "messages": prompt_messages,
            "max_tokens": 70
        }
        
        request_data = {
            "custom_id": f"scannetpp_{scene}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_data
        }
        
        requests.append(request_data)
        
        print(f"[INFO] : {len(requests)} requests created ; batchinput_{scene}.jsonl creating ...")
        
        input_filename = os.path.join(json_root_dir, f"batchinput_{scene}.jsonl")
        with open(input_filename, "w", encoding="utf-8") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        
        # Upload the file for batch processing
        batch_input_file = client.files.create(
            file=open(input_filename, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        print(f"[INFO] : Created batch input file with ID: {batch_input_file_id}")
        
        # Create the batch
        created_batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "video caption batch job"
            }
        )
        
        print(f"[INFO] : Batch created successfully with ID: {created_batch.id}")
        print("[INFO] : Batch status:", created_batch.status)
        
        while True:
            batch_status = client.batches.retrieve(created_batch.id)
            print("[INFO] : Current batch status:", batch_status.status)
            if batch_status.status in ["completed", "failed", "expired"]:
                break
            time.sleep(10)  # wait a bit before checking again
        
        if batch_status.status == "completed":
            output_file_id = batch_status.output_file_id
            print(f"[INFO] : Batch completed. Output file ID: {output_file_id}")
            # Openai Docs : https://github.com/openai/openai-cookbook/blob/main/examples/batch_processing.ipynb
            result = client.files.content(output_file_id).content
            result_file_name = os.path.join(json_root_dir, f"batch_captions_result_{scene}.jsonl")
    
            print(f"[INFO] : Result file name: {result_file_name}")
            with open(result_file_name, "wb") as f:
                f.write(result)
            results = []
            with open(result_file_name, "r") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
        else:
            print(f"[ERROR] : Batch ended with status {batch_status.status}. Check `error_file_id` if available.")