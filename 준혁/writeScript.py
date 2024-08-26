import base64
import boto3
import json
import os

# AWS 서비스 클라이언트 설정
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Claude 모델 ID
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# 사람의 이름
NAME = "신준혁"

# 결과 파일 이름
IMAGE_FILE = 'result.jpg'

def call_claude_haiku(base64_string):
    # 프롬프트 메시지에 NAME 변수를 삽입
    prompt = f"""당신은 이미지 속 인물과 사물을 분석하고, 가상의 인생 스토리를 만들어주세요.
    1. 특정 개인을 식별하는 것은 금지됩니다. 스토리의 주인공은 실존하지 않는 가상의 인물이어야 합니다.
    2. 스토리의 인물은 당신이라는 명칭으로 시작해야 합니다.
    3. 일대기 스토리의 글자는 150개로 제한해주세요.
    4. 이미지 속 인물, {NAME},의 직업을 판단하고 그 직업에 대한 재미있는 일화를 만들어주세요.
    """

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    response = bedrock_runtime.invoke_model(
        body=body, modelId=MODEL_ID, accept="application/json", contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results

def generate_text_from_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')

    text_result = call_claude_haiku(base64_image)
    return text_result

if __name__ == "__main__":
    try:
        generated_text = generate_text_from_image(IMAGE_FILE)
        print(f"Generated Text: {generated_text}")
    except Exception as e:
        print(f"Error: {e}")
