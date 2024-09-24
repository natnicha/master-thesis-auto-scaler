import logging
from torch_dqn import *
from agent import *
from scaling_model import *
from config import config 

import datetime as dt
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
print_interval = 20             # Number of iteration to print result during DQN
max_apisode   = 1000            # Number of episode

scaler_list = []
sfc_update_flag = True

OKGREEN = '\033[92m'
ENDC = '\033[0m'
logging.basicConfig(level=logging.INFO, format=f"{OKGREEN}%(levelname)s{ENDC} %(message)s")

def get_containers_info() -> list:
    # request = requests.get(f"{config['docker']['base_url']}/containers")
    # return request.json()['containers']
    return [
      {
        "cpu_percent": 0.44,
        "id": "98d9ad729bd0",
        "mem_limit_gb": 3.76,
        "mem_percent": 8.34,
        "mem_usage_mb": 321.02,
        "name": "ml.2.ovevrp3"
      },
      {
        "cpu_percent": 0.59,
        "id": "03626f8424fe",
        "mem_limit_gb": 3.76,
        "mem_percent": 8.32,
        "mem_usage_mb": 320.57,
        "name": "ml.1.tjg5uvc"
      }
    ]

def calculate_docker_info(containers_info):
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
  request = requests.post(
    url=f"{config['docker']['base_url']}/scale/{scale}",
    json='',
  )
  if request.status_code == 200:
    return True
  else:
    logging.error(f"scaling pod got: status code {request.status_code}")
    logging.error(f"scaling pod got: {request.json()}")
    raise request.json()

# dqn-threshold(scaler): doing auto-scaling based on dqn
# Input: scaler
# Output: none
def dqn_scaling(scaler: AutoScaler):

  # Initial Processing
  start_time = dt.datetime.now() #+ dt.timedelta(hours = 24)
  containers_info = get_containers_info()
  docker_info = calculate_docker_info(containers_info)
  epsilon_value = epsilon

  # flavors = ni_mon_api.get_vnf_flavors()
  # instance_types = get_sfcr_by_id(sfc_info.sfcr_ids[-1]).nf_chain
  #del instance_types[0] # Flow classifier instance deletion

  # Q-networks
  num_states = 2 # Number of states
  num_actions = 3 # Scale-out, Maintain, Scale-In

  q = Qnet(num_states, num_actions, num_neurons)
  q_target = Qnet(num_states, num_actions, num_neurons)
  q_target.load_state_dict(q.state_dict())
  
  
  if scaler.has_dataset == True:
    q.load_state_dict(torch.load("save_model/"+scaler.get_scaling_name()))
    q_target.load_state_dict(torch.load("save_model/"+scaler.get_scaling_name()))
      
  else:
    logging.info("learning from live data")

  optimizer = optim.Adam(q.parameters(), lr=learning_rate)
  n_epi = 0

  # If there is dataset, read it
  memory = ReplayBuffer(buffer_limit)

  # Start scaling
  scaler.set_active_flag(True)
  
  # Epsilon_value setting
  # epsilon_value = 0.5
  
  if scaler.has_dataset == True:
    epsilon_value = 0.11
    logging.info("has dataset")

  while scaler.get_active_flag():
    # Get state and select action
    state = get_pre_processing_state(docker_info)
    decision = q.sample_action(torch.from_numpy(state).float(), epsilon_value)
    a = decision["action"]
    decision_type = "Policy" if decision["type"] else "R"

    done = False

    # Check whether it is out or in or maintain
    if a == 0:
        logging.info("[%s] Scaling-out! by %s" % (scaler.get_scaling_name(), decision_type))
        scaling_flag = 1
    elif a == 2:
        logging.info("[%s] Scaling-in! by %s" % (scaler.get_scaling_name(), decision_type))
        scaling_flag = -1
    else:
        logging.info("[%s] Maintain! by %s" % (scaler.get_scaling_name(), decision_type))
        scaling_flag = 0

    # For test!!
    # scaling_flag = 0
    # Scaling in or out
    logging.info(f"Epsilon value : {epsilon_value}")
    if scaling_flag != 0:
      num_instances = docker_info['instance_num']
      logging.info(num_instances)

      # Scaling-out
      if scaling_flag > 0:
        # If possible to deploy new VNF instance
        if num_instances < config["instance"]["max_number"]:
          # scale - out command
          scale_pod(ScalingPod.Out)
          logging.info('scaling OUT succeeded')
          # if success; done = True

      # Scaling-in
      elif scaling_flag < 0:
        # If possible to remove VNF instance
        if num_instances > config["instance"]["min_number"]:
          # scale-in
          scale_pod(ScalingPod.In)
          logging.info('scaling IN succeeded')
          # if success; done = True

      # Maintain
      else:
        done = True
        
    exit(1)
    
    # Prepare calculating rewards
    # if scaling_flag == 1 and type_name == "firewall":
    #     print("waiting time for VNF configuration")
        
    # sfc_info = get_sfc_by_name(scaler.get_sfc_name())
    # vnf_info = get_vnf_info(sfc_info)

    s_prime = state_pre_processor(service_info) # TODO: get pre-process state

    # TODO: measure response time
    # response_time = 0.0
    # for rep in range(0,5):
    #     response_time = max(response_time, measure_response_time(scaler, "DQN"))
    
    # response_time = response_time
    
    # if response_time < 0:
    #     break

    # type_instances = get_instances_in_sfc(vnf_info, sfc_info)
    # type_status = get_type_status(type_instances, flavors)

    r = reward_calculator(service_info, response_time)

    done_mask = 1.0 if done else 0.0
    transition = (s,a,r,s_prime,done_mask)
    memory.put(transition)

    if memory.size() > required_mem_size:
      train(q, q_target, memory, optimizer, gamma, batch_size)

    if n_epi % print_interval==0 and n_epi != 0:
      print("[%s] Target network updated!" % (scaler.get_scaling_name()))
      q_target.load_state_dict(q.state_dict())

    current_time = dt.datetime.now() #+ dt.timedelta(hours = 24)

    if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
      scaler.set_active_flag(False)

    n_epi = n_epi+1

    if n_epi > max_apisode:
      scaler.set_active_flag(False)
    if epsilon_value > 0.4 and scaler.has_dataset == False:    
      epsilon_value = epsilon_value - 0.01
    elif scaler.has_dataset == False:
      scaler.set_active_flag(False)
        
    time.sleep(scaler.get_interval())

  # Delete AutoScaler object
  if scaler in scaler_list:
    # delete_monitor(scaler)
    scaler_list.remove(scaler)
    pprint("[Expire: %s] DQN Scaling" % (scaler.get_scaling_name()))
  else:
    pprint("[Exit: %s] DQN Scaling" % (scaler.get_scaling_name()))

  q.save_model("./"+scaler.get_scaling_name())


if __name__ == '__main__':
  dqn_scaling(
    AutoScaler(
    scaling_info=DQN_ScalingInfo(
        sfc_name='docker',
        scaling_name='DQN',
        slo=None,
        duration=0,
        interval=15
    ), 
    type='dqn')
  )
