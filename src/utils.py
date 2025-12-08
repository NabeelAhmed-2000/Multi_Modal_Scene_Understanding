import re
import nltk
from PIL import Image

def expand_box(box, image_size, padding_pct):
    """
    Expands a bounding box by a percentage.
    Source: Cell 6
    """
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    w_pad = width * padding_pct
    h_pad = height * padding_pct

    new_xmin = max(0, xmin - w_pad)
    new_ymin = max(0, ymin - h_pad)
    new_xmax = min(image_size[0], xmax + w_pad)
    new_ymax = min(image_size[1], ymax + h_pad)

    return [new_xmin, new_ymin, new_xmax, new_ymax]

def clean_llama_text(text):
    """
    Removes JSON artifacts from LLM output.
    Source: Cell 18
    """
    if "{" in text and "}" in text:
        try:
            text = re.sub(r'[\{\}\"\[\]]', '', text)
            text = text.replace("final_global_caption", "").replace("Object Observations", "")
            text = text.replace(":", "").strip()
            text = re.sub(r'\s+', ' ', text)
        except:
            pass
    return text

def setup_nltk():
    """Source: Cell 18"""
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng') 
        nltk.download('universal_tagset')
    except Exception as e:
        print(f"NLTK Warning: {e}")
