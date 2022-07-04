# Deep Reinforcement Learning based Autonomous Driving Agents


## About the Project
The main objective of this thesis is to provide a comprehensive analysis of current approaches for training autonomous vehicle agents using deep reinforcement learning (DRL). And our key contribution is to present two working implementations of DRL algorithms. First, we present a Deep Q-Network (DQN)-based agent capable of reliably learning to drive in the CARLA urban driving simulator. Second, Proximal Policy Optimization (PPO) to teach an agent to drive in a driving-like environment (CarRacing-v0)

## DRL deep-Q-Network in CARLA 

The implementation was written in Python 3.7 and TensorFlow 1.14 with Keras 2.2.4.
[CALRA](https://github.com/carla-simulator/carla/releases) version 0.9.4
The code configuration and setup Carla was done on my own PC using PyCharm the training process was carried out on a Linux which was running Ubuntu version 20.4 supported with GPU, the connection was done remotely by OpenVPN and PuTTY.
```ruby
IM_WIDTH = 640
IM_HEIGHT = 480
actor_list = []
collision_hist = []
try:
# connect to client
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
#filter a vehicle of type tesla model3
bp = blueprint_library.filter('model3')[0]
print(bp)
#pick a spawn point randomly
spawn_point = random.choice(world.get_map().get_spawn_points())
#spawn the car
vehicle = world.spawn_actor(bp, spawn_point)
#control the car
vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
#add the vehicle to our list of actors that we need to track
actor_list.append(vehicle)
# get the blueprint for this sensor, spawn location
blueprint = blueprint_library.find('sensor.camera.rgb')
# change the dimensions of the image
blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
blueprint.set_attribute('fov', '110')
# Adjust sensor relative to vehicle
spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
# spawn the sensor and attach to vehicle.
sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
# add sensor to list of actors
actor_list.append(sensor)
```