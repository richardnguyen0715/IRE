from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_KEY"))
project = rf.workspace("food-recipe-ingredient-images-0gnku").project("food-ingredients-dataset")
version = project.version(4)
dataset = version.download("yolo26")
                