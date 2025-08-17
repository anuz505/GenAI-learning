import boto3
import json
from botocore.exceptions import ClientError
client = boto3.client(service_name="bedrock-runtime")

prompt = "Tell me a fun story about Radiohead."


model_id  = "meta.llama3-70b-instruct-v1:0"

formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

payload = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
}

request = json.dumps(payload)

try:
    response = client.invoke_model(modelId=model_id, body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

model_res = json.loads(response["body"].read())

response_text = model_res["generation"]
print(response_text)

