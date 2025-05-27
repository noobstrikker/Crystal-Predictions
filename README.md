# Crystal-Predictions
to run our setup and models you'll need a couple of dependencies
these are as following
Python
    can be installed using pip with the command: pip install python

pytorch
    we recommend going to https://pytorch.org/get-started/locally/ and using their "start locally" configuration menu, simply chose the for your system appropriate fields and then simply copy and paste the output command in your command line (we recomend chosing a cuda option if possible to take advange of any gpu toy might have)

torch_geometric
   can be installed using pip with the command: pip install torch_geometric

the materials project api
    can be installed using pip with the command: pip install mp_api

scikit-learn
    can be installed using pip with the command: pip install mp_api

seaborn
    can be installed using pip with the command: pip install seaborn

optuna
    can be installed using pip with the command: pip install optuna

now assuming you followed the previous steps and got no errors or issues, you simply need to path to the directory and run the main file using the command py main.py or python main.py depending on your configuration of python (if one doesn't work simply try the other)

you should now find yourself with a couply of in program options such as "retreive new dataset", "train model" and a couple others 

from here you're free to play around with the options we have left a couple of pretrained models and dataset for your convience including the models detailed in our rapport and the dataset used.

note when using the optuna auto-tuner it's optimized paraters are placed as a json-file in the optuna_studies folder, if you wish to use these you will have to manually insert them.