To install the requirements, run the following:
```bash
bash setup_custom_venv.sh
```

To activate the installed python venv:
```bash
PEFT_VENV_DIR=./venv-lola-custom-task
source $PEFT_VENV_DIR/bin/activate
```

To convert the training data to json:
```bash
python convert_to_json.py
```

To start the training, run:
```bash
bash start_custom_train.sh false true
```

To run the model in inference, use the following script:
```bash
python inference_example.py
```