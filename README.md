# This is for PRIMA.

## Prerequisites:
The code is tested on Linux with a CPU cluster and it will be a further improvement for running with GPU. There are two ways provided to build the environment for running the code.
- All the packages (listed in "requirements.txt") are installed based on python3.6. After it is done, install [ray](https://github.com/ray-project/ray) with "sudo pip install ray". In addition, install [Jacinle](https://github.com/vacancy/Jacinle/tree/ed90c3a70a133eb9c6c2f4ea2cc3d907de7ffd57) if needed.
- Run the docker file ("Dockerfile") to setup the environment.

## Code
- "NLM_MBRL.py" is the entry of running this code.
- "NLM_MBRL_config.py" is the main file with the hyper-parameter settings.
- "models.py" is the main file for constructing the model.

## Usage
This repo contains 3 graph tasks and 5 family tree tasks under the multi-task reasoning setting.
- An example of the command for training is "jac-run NLM_MBRL.py". 
- "model.weights" in the folder of "saved_model" is the pre-trained model weights. An example of testing is "jac-run NLM_MBRL.py --task_mode=test".

Reference:
- [Neural Logic Machines](https://github.com/google/neural-logic-machines)
- [MuZero](https://github.com/werner-duvaud/muzero-general)
