from modularevogym.envs.base import *
from modularevogym.envs.balance import *
from modularevogym.envs.manipulate import *
from modularevogym.envs.climb import *
from modularevogym.envs.flip import *
from modularevogym.envs.jump import *
from modularevogym.envs.multi_goal import *
from modularevogym.envs.change_shape import *
from modularevogym.envs.traverse import *
from modularevogym.envs.walk import *

from gym.envs.registration import register

## SIMPLE ##
register(
    id = 'Walker-v0',
    entry_point = 'modularevogym.envs.walk:WalkingFlat',
    max_episode_steps=500
)

register(
    id = 'BridgeWalker-v0',
    entry_point = 'modularevogym.envs.walk:SoftBridge',
    max_episode_steps=500
)

register(
    id = 'CaveCrawler-v0',
    entry_point = 'modularevogym.envs.walk:Duck',
    max_episode_steps=1000
)

register(
    id = 'Jumper-v0',
    entry_point = 'modularevogym.envs.jump:StationaryJump',
    max_episode_steps=500
)

register(
    id = 'Flipper-v0',
    entry_point = 'modularevogym.envs.flip:Flipping',
    max_episode_steps=600
)

register(
    id = 'Balancer-v0',
    entry_point = 'modularevogym.envs.balance:Balance',
    max_episode_steps=600
)

register(
    id = 'Balancer-v1',
    entry_point = 'modularevogym.envs.balance:BalanceJump',
    max_episode_steps=600
)

register(
    id = 'UpStepper-v0',
    entry_point = 'modularevogym.envs.traverse:StepsUp',
    max_episode_steps=600
)

register(
    id = 'DownStepper-v0',
    entry_point = 'modularevogym.envs.traverse:StepsDown',
    max_episode_steps=500
)

register(
    id = 'ObstacleTraverser-v0',
    entry_point = 'modularevogym.envs.traverse:WalkingBumpy',
    max_episode_steps=1000
)

register(
    id = 'ObstacleTraverser-v1',
    entry_point = 'modularevogym.envs.traverse:WalkingBumpy2',
    max_episode_steps=1000
)

register(
    id = 'Hurdler-v0',
    entry_point = 'modularevogym.envs.traverse:VerticalBarrier',
    max_episode_steps=1000
)

register(
    id = 'GapJumper-v0',
    entry_point = 'modularevogym.envs.traverse:Gaps',
    max_episode_steps=1000
)

register(
    id = 'PlatformJumper-v0',
    entry_point = 'modularevogym.envs.traverse:FloatingPlatform',
    max_episode_steps=1000
)

register(
    id = 'Traverser-v0',
    entry_point = 'modularevogym.envs.traverse:BlockSoup',
    max_episode_steps=600
)

## PACKAGE ##
register(
    id = 'Lifter-v0',
    entry_point = 'modularevogym.envs.manipulate:LiftSmallRect',
    max_episode_steps=300
)

register(
    id = 'Carrier-v0',
    entry_point = 'modularevogym.envs.manipulate:CarrySmallRect',
    max_episode_steps=500
)

register(
    id = 'Carrier-v1',
    entry_point = 'modularevogym.envs.manipulate:CarrySmallRectToTable',
    max_episode_steps=1000
)

register(
    id = 'Pusher-v0',
    entry_point = 'modularevogym.envs.manipulate:PushSmallRect',
    max_episode_steps=500
)

register(
    id = 'Pusher-v1',
    entry_point = 'modularevogym.envs.manipulate:PushSmallRectOnOppositeSide',
    max_episode_steps=600
)

register(
    id = 'BeamToppler-v0',
    entry_point = 'modularevogym.envs.manipulate:ToppleBeam',
    max_episode_steps=1000
)

register(
    id = 'BeamSlider-v0',
    entry_point = 'modularevogym.envs.manipulate:SlideBeam',
    max_episode_steps=1000
)

register(
    id = 'Thrower-v0',
    entry_point = 'modularevogym.envs.manipulate:ThrowSmallRect',
    max_episode_steps=300
)

register(
    id = 'Catcher-v0',
    entry_point = 'modularevogym.envs.manipulate:CatchSmallRect',
    max_episode_steps=400
)

### SHAPE ###
register(
    id = 'AreaMaximizer-v0',
    entry_point = 'modularevogym.envs.change_shape:MaximizeShape',
    max_episode_steps=600
)

register(
    id = 'AreaMinimizer-v0',
    entry_point = 'modularevogym.envs.change_shape:MinimizeShape',
    max_episode_steps=600
)

register(
    id = 'WingspanMazimizer-v0',
    entry_point = 'modularevogym.envs.change_shape:MaximizeXShape',
    max_episode_steps=600
)

register(
    id = 'HeightMaximizer-v0',
    entry_point = 'modularevogym.envs.change_shape:MaximizeYShape',
    max_episode_steps=500
)

### CLIMB ###
register(
    id = 'Climber-v0',
    entry_point = 'modularevogym.envs.climb:Climb0',
    max_episode_steps=400
)

register(
    id = 'Climber-v1',
    entry_point = 'modularevogym.envs.climb:Climb1',
    max_episode_steps=600
)

register(
    id = 'Climber-v2',
    entry_point = 'modularevogym.envs.climb:Climb2',
    max_episode_steps=1000
)

### MULTI GOAL ###
register(
    id = 'BidirectionalWalker-v0',
    entry_point = 'modularevogym.envs.multi_goal:BiWalk',
    max_episode_steps=1000
)