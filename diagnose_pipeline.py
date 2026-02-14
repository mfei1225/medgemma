
import torch
from transformers import pipeline
import logging
from PIL import Image
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = "google/medgemma-1.5-4b-it"

def debug_pipeline():
    logger.info(f"Loading model for diagnosis: {model_id}")
    try:
        # Load pipeline
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"attn_implementation": "eager"}
        )
        
        logger.info(f"Pipeline type: {type(pipe)}")
        logger.info(f"Tokenizer type: {type(pipe.tokenizer)}")
        if hasattr(pipe, 'image_processor'):
            logger.info(f"Image processor type: {type(pipe.image_processor)}")
            
        # Create a dummy image
        img = Image.new('RGB', (224, 224), color='red')
        prompt = "Describe this image."
        
        # Test 1: Positionals
        logger.info("Attempting Test 1: (img, prompt)")
        try:
            res = pipe(img, prompt, max_new_tokens=10)
            logger.info(f"Test 1 Success: {res[0]['generated_text'][:20]}...")
        except Exception as e:
            logger.info(f"Test 1 Failed: {e}")
            
        # Test 2: (prompt, img)
        logger.info("Attempting Test 2: (prompt, img)")
        try:
            res = pipe(prompt, img, max_new_tokens=10)
            logger.info(f"Test 2 Success: {res[0]['generated_text'][:20]}...")
        except Exception as e:
            logger.info(f"Test 2 Failed: {e}")
            
        # Test 3: Keywords (image, prompt)
        logger.info("Attempting Test 3: (image=img, prompt=prompt)")
        try:
            res = pipe(image=img, prompt=prompt, max_new_tokens=10)
            logger.info(f"Test 3 Success: {res[0]['generated_text'][:20]}...")
        except Exception as e:
            logger.info(f"Test 3 Failed: {e}")

        # Test 4: Keywords (images, prompt)
        logger.info("Attempting Test 4: (images=img, prompt=prompt)")
        try:
            res = pipe(images=img, prompt=prompt, max_new_tokens=10)
            logger.info(f"Test 4 Success: {res[0]['generated_text'][:20]}...")
        except Exception as e:
            logger.info(f"Test 4 Failed: {e}")

        # Test 5: Keywords (images, text)
        logger.info("Attempting Test 5: (images=img, text=prompt)")
        try:
            res = pipe(images=img, text=prompt, max_new_tokens=10)
            logger.info(f"Test 5 Success: {res[0]['generated_text'][:20]}...")
        except Exception as e:
            logger.info(f"Test 5 Failed: {e}")

        # Test 6: Messages
        logger.info("Attempting Test 6: (messages format)")
        try:
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            res = pipe(messages, max_new_tokens=10)
            logger.info(f"Test 6 Success: {res[0]['generated_text'][:20]}...")
        except Exception as e:
            logger.info(f"Test 6 Failed: {e}")

    except Exception as e:
        logger.error(f"Failed to load or debug pipeline: {e}")

if __name__ == "__main__":
    debug_pipeline()
