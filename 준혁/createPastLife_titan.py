import base64
import io
import json
import logging
import boto3
from PIL import Image
from botocore.exceptions import ClientError
import random

class ImageError(Exception):
    "Custom exception for errors returned by Amazon Titan Image Generator G1"

    def __init__(self, message):
        self.message = message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_image(model_id, body):
    logger.info("Generating image with Amazon Titan Image Generator G1 model %s", model_id)
    
    # Ensure that the correct region is set
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

    accept = "application/json"
    content_type = "application/json"

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        base64_bytes = base64_image.encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise ImageError(f"Image generation error. Error is {finish_reason}")

        logger.info("Successfully generated image with Amazon Titan Image Generator G1 model %s", model_id)
        return image_bytes

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        raise

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_id = 'amazon.titan-image-generator-v2:0'
    
    professions = ["writer", "painter","scholar", "farmer", "musician", "traveler", "king", "queen"]
    eras = ["ancient times",  "early modern period"]
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
        image_bytes = generate_image(model_id=model_id, body=body)
        image = Image.open(io.BytesIO(image_bytes))

        image.save('past_life_result.jpg', format='JPEG')
        logger.info("Image saved as past_life_result.jpg")
        image.show()
        
    except ClientError as err:
        print(f"A client error occurred: {err}")
    except ImageError as err:
        logger.error(err.message)
        print(err.message)
    else:
        print(f"Finished generating image with Amazon Titan Image Generator G1 model {model_id}.")

if __name__ == "__main__":
    main()
