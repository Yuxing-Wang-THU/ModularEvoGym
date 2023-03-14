im_size = 7

# 'EMPTY': 0, 'RIGID': 1, 'SOFT': 2, 'H_ACT': 3, 'V_ACT': 4
num_classes = 5

# Moore Neighbor
number_neighbors = 9

# Design steps
design_iterations = 10

class ncaconfig:
    def __init__(self) -> None:

        self.im_size = im_size

        self.num_classes = num_classes

        self.number_neighbors=number_neighbors

        self.design_iterations=design_iterations

class transformerconfig:
    def __init__(self) -> None:
        self.POS_EMBEDDING = "learnt"

        self.use_other_obs_encoder = False

        self.condition_decoder=True

        self.attention_embedding_size=64

        self.attention_heads=1

        self.attention_hidden_size=128

        self.attention_layers=1

        self.dropout_rate=0.0

        self.transformer_norm=False

class ppoconfig:
    def __init__(self) -> None:

        self.DESIGN_SIZE = int((im_size-2)*(im_size-2))

        self.CELLDIM = 1
        
        self.NUM_NEIGHBOR = number_neighbors

        self.ori_log_dir = '/tmp/modularevogym/'

        self.num_env_steps=10e6

        self.num_evals = 1

        self.lr = 2.5e-4

        self.log_interval=10

        self.eval_interval=10

        self.max_grad_norm = 0.5

        self.use_linear_lr_decay= True

        self.ACTION_STD_FIXED = True
        
        self.ACTION_STD = 0.2
       
        # Discount factor for rewards
        self.GAMMA = 0.99

        # GAE lambda parameter
        self.GAE_LAMBDA = 0.95

        # Hyperparameter which roughly says how far away the new policy is allowed to
        # go from the old
        self.CLIP_EPS = 0.1

        # Number of epochs (K in PPO paper) of sgd on rollouts in buffer
        self.EPOCHS = 8

        # Batch size for sgd (M in PPO paper)
        self.NUM_MINI_BATCH_SIZE = 8

        # Value (critic) loss term coefficient
        self.VALUE_COEF = 0.5

        # If KL divergence between old and new policy exceeds KL_TARGET_COEF * 0.01
        # stop updates. Default value is high so that it's not used by default.
        self.KL_TARGET_COEF = 20.0

        # Clip value function
        self.USE_CLIP_VALUE_FUNC = True

        # Entropy term coefficient
        self.ENTROPY_COEF = 0.01

        # Max timesteps per rollout
        self.TIMESTEPS = 2048
        
        # EPS for Adam/RMSProp
        self.EPS = 1e-5

