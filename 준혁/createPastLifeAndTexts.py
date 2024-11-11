import base64
import io
import json
import logging
import boto3
import random
from PIL import Image
from botocore.exceptions import ClientError

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 모델 및 클라이언트 설정
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
s3 = boto3.client('s3')
IMAGE_MODEL_ID = 'amazon.titan-image-generator-v2:0'
TEXT_MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

# 사용자 정의 예외 클래스
class ImageError(Exception):
    def __init__(self, message):
        self.message = message

# 이미지 생성 함수
def generate_image():
        professions = ["writer", "painter","scholar", "farmer", "musician", "traveler", "king", "queen", 
                   "emperor","Town Crier", "Rat Catcher", "Lamp Lighter", "Phrenologist", "Knocker-Up", 
                   "Cartographer", "Lamplighter", "Cooper", "Blacksmith", "Telegraph Operator", 
                   "Ice Cutter", "Leech Collector", "Scribe", "Cobbler", "Gong Farmer", "Alchemist", 
                   "Water Carrier", "Lector", "Tanner", "Miller", "Ragpicker", "Thatcher", "Haberdasher", 
                   "Perfumier", "Silversmith", "Scrivener", "Wainwright",
                   "Charcoal Burner", "Oyster Shucker",
                   "glassblower", "spindle whorl", "fishwife", "mudlark", "shoemaker", "dyer", 
                   "saddler", "Farrier", "Cordwainer", "Wheelwright", "Oyster Shucker", "Chimney Sweep"
                   "Lacemaker", "Drover", "Milkmaid", "Usher", "Basket Weaver", "Salt Boiler", "Tinker", "Ship Chandler", "Town Balladeer",
                   "Ploughman", "Scullery Maid", "Wattle-and-Dauber", "Fuller", "Ropemaker", "Crofter", "Quarryman", 
                   "Brewster", "Alewife", "Parish Beadle", "Match Girl", "Collier", "Ostler", "Pannier Man", "Pewterer"]
  
    eras = ["ancient times", "early modern period"]
    ethnicities = ["diverse", "cultural blend", "heritage-rich", "white", "yellow", "black"]

    random_profession = random.choice(professions)
    random_era = random.choice(eras)
    random_ethnicity = random.choice(ethnicities)

    logger.info("Selected profession: %s", random_profession)
    logger.info("Selected era: %s", random_era)
    logger.info("Selected ethnicity: %s", random_ethnicity)

    prompt = f"""
    A person from {random_era}, practicing as a {random_profession}.
    Appears with a {random_ethnicity} background.
    A whole face with a detailed background.
    Looking at viewer.
    Finely detailed eyes and face.
    High nose bridge.
    Reflecting the cultural style of {random_era}.
    """

    random_seed = random.randint(0, 214783647)
    
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 9.0,
            "seed": random_seed
        }
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=IMAGE_MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        image_bytes = base64.b64decode(base64_image.encode('ascii'))

        if response_body.get("error"):
            raise ImageError(f"Image generation error. Error is {response_body.get('error')}")

        logger.info("Image generation successful")
        return image_bytes

    except ClientError as err:
        logger.error("A client error occurred: %s", err.response["Error"]["Message"])
        raise

# 텍스트 생성 함수
def generate_text_from_image(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
        이미지를 보고 이 인물의 가상의 전생 이야기를 만들어주세요.
        1. 특정 개인을 식별하지 마세요. 주인공은 가상의 인물이어야 합니다.
        2. 이야기는 "당신은..."으로 시작하세요.
        3. 이야기의 길이는 150자 이내로 제한하세요.
        4. 마지막에 '*'를 쓰고 '헤어스타일', '성별', '피부색'을 상세히 적어주세요.
        5. 인물의 어린 시절 이야기는 포함하지 마세요.
        6. 응답은 반드시 한국어로 작성해주세요.
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
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    response = bedrock.invoke_model(
        body=body,
        modelId=TEXT_MODEL_ID,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results

# 메인 함수
def main():
    try:
        # 이미지 생성
        image_bytes = generate_image()
        image = Image.open(io.BytesIO(image_bytes))
        image.save('past_life_result.jpg', format='JPEG')
        logger.info("Image saved as past_life_result.jpg")
        
        # 텍스트 생성
        generated_text = generate_text_from_image(image_bytes)
        print("Generated Story:", generated_text)

        # 이미지 출력 (로컬에서만 가능)
        image.show()
        
    except ImageError as err:
        logger.error(err.message)
    except Exception as e:
        logger.error("An error occurred: %s", e)

if __name__ == "__main__":
    main()
