import logging
import os
from torch_dqn import *
from agent import *
from scaling_model import *
from config import config 

from datetime import datetime
import time
import torch.optim as optim
import torch
import requests

# <Important> parameters for Reinforcement Learning
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 2500            # Maximum Buffer size
batch_size    = 16              # Batch size for mini-batch sampling
num_neurons   = 128             # Number of neurons in each hidden layer
epsilon       = 0.10            # epsilon value of e-greedy algorithm
required_mem_size = 20          # Minimum number triggering sampling
copy_param_interval = 20        # Number of iteration to copy parameters to target Q-network
max_apisode   = 1000              # Number of episode

scaler_list = []
sfc_update_flag = True

OKGREEN = '\033[92m'
WARNING = '\033[93m'
ENDC = '\033[0m'
logging.basicConfig(level=logging.INFO, format=f"{OKGREEN}%(levelname)s{ENDC} %(message)s")

def get_containers_info() -> list:
    URL = f"{config['docker']['base_url']}/containers"
    try:
      request = requests.get(URL)
      if request.status_code != 200:
        raise ValueError(f"connection error on {URL}")
    except Exception as e:
      raise e
    return request.json()['containers']

def get_app_stat() -> dict:
    URL = f"{config['docker']['base_url']}/app/stat"
    try:
      request = requests.get(URL)
      if request.status_code != 200:
        raise ValueError(f"connection error on {URL}")
    except Exception as e:
      raise e
    return request.json()

def calculate_service_info(containers_info):
  cpu_percent = 0
  mem_percent = 0
  for container in containers_info:
    cpu_percent += float(container['cpu_percent'])
    mem_percent += float(container['mem_percent'])
  return {
    'instance_num': len(containers_info),
    'cpu_percent': cpu_percent/len(containers_info),
    'mem_percent': mem_percent/len(containers_info),
  }

def get_pre_processing_state(docker_info):
    state = []
    state.append(docker_info["cpu_percent"])
    state.append(docker_info["mem_percent"])
    return np.array(state)

class ScalingPod():
  In = 'in'
  Out = 'out'

def scale_pod(scale: ScalingPod):
  try:
    request = requests.post(
      url=f"{config['docker']['base_url']}/scale/{scale}",
      json='',
    )
    logging.info(f"scaling pod got: status code {request.status_code}")
    if request.status_code == 200:
      return request.json()
    else:
      logging.error(f"scaling pod got: status code {request.status_code}")
      logging.error(f"scaling pod got: {request.json()}")
      raise request.json()
  except Exception as e:
    raise e


# dqn-threshold(scaler): doing auto-scaling based on dqn
# Input: scaler
# Output: none
def dqn_scaling(scaler: AutoScaler):

  # Initial Processing
  start_time = dt.datetime.now() #+ dt.timedelta(hours = 24)
  epsilon_value = epsilon

  # Q-networks
  num_states = 2 # Number of states
  num_actions = 3 # Scale-out, Maintain, Scale-In

  q = Qnet(num_states, num_actions, num_neurons)
  q_target = Qnet(num_states, num_actions, num_neurons)
  q_target.load_state_dict(q.state_dict())
  
  
  is_file = os.path.isfile("./save_model/"+scaler.get_scaling_name())
  # if scaler.has_dataset == True:
  if is_file:
    q.load_state_dict(torch.load("./save_model/"+scaler.get_scaling_name()))
    q_target.load_state_dict(torch.load("./save_model/"+scaler.get_scaling_name()))
    logging.info("loaded a saved model")
  else:
    logging.info("learning from live data")

  optimizer = optim.Adam(q.parameters(), lr=learning_rate)
  n_epi = 0

  # If there is dataset, read it
  memory = ReplayBuffer(buffer_limit)

  # Start scaling
  scaler.set_active_flag(True)
  
  # Epsilon_value setting
  epsilon_value = 0.5
  if scaler.has_dataset == True:
    epsilon_value = 0.11
    logging.info("has dataset")

  while scaler.get_active_flag():
    logging.info(f"{WARNING} ---------- Episode {n_epi} ---------- {ENDC}")
    # Get state and select action
    containers_info = get_containers_info()
    service_info = calculate_service_info(containers_info)
    state = get_pre_processing_state(service_info)
    logging.info(f"state (s): {state}")
    decision = q.sample_action(torch.from_numpy(state).float(), epsilon_value)
    a = decision["action"]

    done = False

    # Check whether it is out or in or maintain
    if a == 0:
        logging.info("[%s] Scaling-out! by %s" % (scaler.get_scaling_name(), decision["by"]))
        scaling_flag = 1
    elif a == 2:
        logging.info("[%s] Scaling-in! by %s" % (scaler.get_scaling_name(), decision["by"]))
        scaling_flag = -1
    else:
        logging.info("[%s] Maintain! by %s" % (scaler.get_scaling_name(), decision["by"]))
        scaling_flag = 0


    # For testing
    # scaling_flag = 0
    # Scaling in or out
    logging.info(f"Epsilon value : {epsilon_value}")
    respond =  {
      'start_time': 0,
      'finish_time': 0,
      'time_spent_sec': 0.00
    }

    num_instances = service_info['instance_num']
    logging.info(f"current instance: {num_instances}")

    if scaling_flag != 0 and \
      (num_instances+scaling_flag < config["instance"]["min_number"] or \
      num_instances+scaling_flag > config["instance"]["max_number"]):
      logging.info(f"continue due to a number of instance is out of bound")
      continue

    if scaling_flag != 0 and \
      num_instances+scaling_flag >= config["instance"]["min_number"] and \
      num_instances+scaling_flag <= config["instance"]["max_number"]:

      # Scaling-out
      if scaling_flag > 0:
        respond = scale_pod(ScalingPod.Out)
        logging.info(f"scaling OUT succeeded with:{respond['time_spent_sec']}")
        done = True

      # Scaling-in
      elif scaling_flag < 0:
        respond = scale_pod(ScalingPod.In)
        logging.info(f"scaling IN succeeded with:{respond['time_spent_sec']}")
        done = True

    # Maintain
    else:
      scaling_flag = 0
      done = True

    # Prepare calculating rewards
    # Find s' (post state)
    app_stat = get_app_stat()
    post_service_info = calculate_service_info(app_stat['containers'])
    s_prime = get_pre_processing_state(post_service_info)
    logging.info(f"post state(s'): {s_prime}")
    r = reward_calculator(app_stat)

    done_mask = 1.0 if done else 0.0
    transition = (state,a,r,s_prime,done_mask)
    memory.put(transition)

    if memory.size() > required_mem_size:
      train(q, q_target, memory, optimizer, gamma, batch_size)

    if n_epi % copy_param_interval==0 and n_epi != 0:
      logging.info("[%s] Target network updated!" % (scaler.get_scaling_name()))
      q_target.load_state_dict(q.state_dict())

    current_time = dt.datetime.now()

    if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
      scaler.set_active_flag(False)

    n_epi = n_epi+1

    if n_epi > max_apisode:
      scaler.set_active_flag(False)
    if epsilon_value > 0.4 and scaler.has_dataset == False:    
      epsilon_value = epsilon_value - 0.01
    elif scaler.has_dataset == False:
      scaler.set_active_flag(False)
    
    if n_epi % scaler.get_save_model_interval()==0 and n_epi != 0:
      q.save_model("./save_model/"+scaler.get_scaling_name())
      logging.info("[%s] model saved" % (scaler.get_scaling_name()))
    
    time.sleep(scaler.get_interval())

  # Delete AutoScaler object
  if scaler in scaler_list:
    # delete_monitor(scaler)
    scaler_list.remove(scaler)
    logging.info(f"[Expire: {scaler.get_scaling_name()}] DQN Scaling")
  else:
    logging.info(f"[Exit: {scaler.get_scaling_name()}] DQN Scaling")

  q.save_model("./save_model/"+scaler.get_scaling_name())
  logging.info("[%s] model saved" % (scaler.get_scaling_name()))


if __name__ == '__main__':
  dqn_scaling(
    AutoScaler(
    scaling_info=DQN_ScalingInfo(
        sfc_name='docker',
        scaling_name='DQN',
        slo=None,
        duration=0,
        interval=0,
        save_model_interval=20
    ), 
    type='dqn')
  )
