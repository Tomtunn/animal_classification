# gradio app
import gradio as gr
import numpy as np
from fastai.vision.all import *

# Load the model
predictor = load_learner("models/best_model.pkl")

def predict_image(img):
    # Convert the image to a tensor
    img = PILImage.create(img)
    # Make a prediction
    pred, pred_idx, probs = predictor.predict(img)
    probs =  np.round(probs[pred_idx].item(), 4)
    return {pred: probs}


# Create a Gradio interface\
def main(): 
    demo = gr.Interface(
        fn=predict_image, 
        inputs="image", 
        outputs="label", 
        title="Image Classifier", 
        description="Classify images using a pre-trained model.")
    demo.launch()
    
if __name__ == "__main__":
    main()

