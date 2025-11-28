import pandas as pd
from qwen_vl_utils.vision_process import fetch_image

df = pd.read_parquet("your/path/to/SFT+RL/data_case/train/test.parquet")
row = df.iloc[0]
mm = row["images"][0]

print(type(mm), mm)       
img = fetch_image(mm)      
print(img.size)            
