# Below configures are examples, 
# you can modify them as you wish.
"""
It is not a good design to use .py as configurations. I rewrote config files

"""
### YOUR CODE HERE

import yaml, os

def load_config(config_name):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'configs', config_name)) as file:
        config = yaml.safe_load(file)
    return config


# model_configs = {
# 	"name": 'MyModel',
# 	"save_dir": '../saved_models/',
# 	"depth": 2,
# 	"levels":, channels, num_classes:10,
#                  block=BasicBlock
# 	# ...
# }

# training_configs = {
# 	"learning_rate": 0.01,
# 	# ...
# }

### END CODE HERE