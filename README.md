# Heart rate detection from live video stream (or a video file)
Developed by Rasa Kundrotaite and Martynas Kucys, IFD-0

## Create a development (conda) environment
```
conda create --name vision python=3.10
conda activate vision
pip install -r requirements.txt
```
## Run the application 
From root folder ```visual_heart_rate_detection``` run
```
python main.py
```

## Additional information
If you install any new libraries, add them to requirements.txt
```
pip list --format=freeze > requirements.txt
```
