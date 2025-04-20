### Installation Process
Open Command Prompt in a new folder (for tidiness purposes) on a Windows 10 machine that already has Python 3.12.3 installed and functional. Then type the following commands.
> py -m venv virtEnv
* Creates a Python virtual environment.
* virtEnv is the name of the Python virtual environment. Changing the name will affect the next command.
* Cutting/Pasting/Copying the Python virtual environment into a different folder location will destroy its usefulness due to paths hardcoded at venv creation, and you will ***need*** to make a new one in the desired location, so do not modify this folder once created.
> virtEnv\Scripts\activate
* Enter the Python virtual environment. If you exit the Command Prompt, you will need to enter this command again, while Command Prompt is in the folder that contains the Python virtual environment, before running the model
> pip install torch torchvision  
> pip install easyocr  
* Install the software that supports text detection and recognition models
> pip install matplotlib  
> pip install gradio  
> py main.py  
* Runs the detection and recognition model. The first time you run the program, the specific model requested in the code will be downloaded from the internet. Switch "main.py" to the correct file you desire to run.

#### Metainfo
* A list of specific libraries that were included exists in the created requirements.txt file, made by `pip freeze > requirements.txt` while inside the activated Python virtual environment.
* If you are unsure of which Python version is active, you can use the commands `where python` and `python --version` in Windows Command Prompt. Ensure that the Python virtual environment is active, as that will change both the libraries and Python version available for use.

### User's Process
User inputs a picture from their phone, the pre-trained model is loaded according to the user-selected language, the inputs are fed to the pre-trained model to generate the characters in the same language as they are written in their environment (i.e., reading a Swedish text will output Swedish characters), then the characters are translated to the user's desired language, such as English, and are finally displayed to the user.
