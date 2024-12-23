import logging
import os
import pandas as pd
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
max_apisode   = 10000              # Number of episode

scaler_list = []
sfc_update_flag = True

OKGREEN = '\033[92m'
WARNING = '\033[93m'
ENDC = '\033[0m'
logging.basicConfig(filename=config['logging']['filename'],
                    filemode='a',
                    level=logging.INFO, 
                    datefmt='%H:%M:%S',
                    format=f"%(asctime)s.%(msecs)d %(levelname)s %(message)s")
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter(f"%(asctime)s {OKGREEN}%(levelname)s{ENDC} %(message)s")
# tell the handler to use this format
console.setFormatter(formatter)
logging.Formatter(f"%(asctime)s.%(msecs)d %(levelname)s %(message)s")
# add the handler to the root logger
logging.getLogger().addHandler(console)

def get_containers_info() -> dict:
    URL = f"{config['docker']['base_url']}/containers/confirm"
    try:
      request = requests.post(URL)
      if request.status_code != 200:
        raise ValueError(f"connection error on {URL}")
    except Exception as e:
      raise e
    return request.json()

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
  containers = containers_info['container']
  online_pods = containers_info['system']['online_pods']
  if (online_pods) == 0:
    raise ValueError("no containers")
  return {
    'instance_num': int(containers_info['system']['online_pods']),
    'cpu_percent': float(containers_info['system']['cpu_percent']),
    'mem_percent': float(containers_info['system']['mem_percent']),
  }

def get_pre_processing_state(docker_info):
    state = []
    state.append(docker_info["cpu_percent"])
    state.append(docker_info["mem_percent"])
    state.append(docker_info["instance_num"])
    return np.array(state)

class ScalingPod():
  In = 'in'
  Out = 'out'

def scale_pod(scale: ScalingPod):
  is_success = False
  while(not is_success):
    try:
      request = requests.post(
        url=f"{config['docker']['base_url']}/pod/scale/{scale}",
        json='',
      )
      logging.info(f"scaling pod got: status code {request.status_code}")
      if request.status_code == 200:
        is_success = True
        return request.json()
      else:
        logging.error(f"scaling pod got: status code {request.status_code}")
        logging.error(f"scaling pod got: {request.json()}")
    except Exception as e:
      logging.error(f"scaling pod got: {e}")

def append_stat(stat :pd.DataFrame, pre_s_CPU: float, pre_s_MEM: float, pre_s_pods: int, sampled_action: str, sampled_action_by: str, epsilon: float, 
                post_s_CPU: float, post_s_MEM: float, post_s_pods: int, post_s_latency: float, post_s_drop_packets: float, is_done: bool,reward: float) -> pd.DataFrame: 
  
  stat.loc[len(stat)] = [dt.datetime.now(), round(pre_s_CPU, 6), round(pre_s_MEM, 6), round(pre_s_pods, 6), sampled_action, sampled_action_by, epsilon, round(post_s_CPU, 6), round(post_s_MEM, 6), post_s_pods, post_s_latency, round(post_s_drop_packets, 6), is_done, round(reward, 6)]
  return stat
  
# dqn-threshold(scaler): doing auto-scaling based on dqn
# Input: scaler
# Output: none
def dqn_scaling(scaler: AutoScaler):
  stat = pd.DataFrame({"timestamp": pd.Series(dtype='datetime64[ns]'),
                        "s_CPU (%)": pd.Series(dtype='float'),
                        "s_MEM (%)": pd.Series(dtype='float'),
                        "s_pods": pd.Series(dtype='int'),
                        "decision_action": pd.Series(dtype='str'),
                        "decision_action_by": pd.Series(dtype='str'),
                        "epsilon": pd.Series(dtype='float'),
                        "s'_CPU (%)": pd.Series(dtype='float'),
                        "s'_MEM (%)": pd.Series(dtype='float'),
                        "s'_pods": pd.Series(dtype='int'),
                        "s'_latency (ms)": pd.Series(dtype='float'),
                        "s'_drop_packets (%)": pd.Series(dtype='float'),
                        "is_done": pd.Series(dtype='str'),
                        "reward": pd.Series(dtype='float')})
    
  # Initial Processing
  start_time = dt.datetime.now() #+ dt.timedelta(hours = 24)
  epsilon_value = epsilon

  # Q-networks
  num_states = 3 # Number of states
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
    try:
      containers_info = get_containers_info()
      service_info = calculate_service_info(containers_info)
    except Exception as e:
      logging.warning(str(e))
      continue
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

    num_instances = service_info['instance_num']
    logging.info(f"current instance: {num_instances}")

    if scaling_flag != 0 and \
      (num_instances+scaling_flag < config["instance"]["min_number"] or \
      num_instances+scaling_flag > config["instance"]["max_number"]):
      logging.info(f"continue due to a number of instance is out of bound")
      logging.info(f"Force to Maintain!")
      scaling_flag = 0

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
    app_stat = None
    while app_stat == None:
      try:
        app_stat = get_app_stat()
        post_service_info = calculate_service_info(app_stat['containers'])
      except Exception as e:
        logging.warning(str(e))
        done = False
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

    stat = append_stat(stat=stat, pre_s_CPU=state[0], pre_s_MEM=state[1], pre_s_pods=service_info['instance_num'], sampled_action=scaling_flag, sampled_action_by=decision["by"], epsilon=epsilon_value,  
                post_s_CPU=s_prime[0], post_s_MEM=s_prime[1], post_s_pods=post_service_info['instance_num'], post_s_latency=app_stat['requests_stat']['avg_latency'], post_s_drop_packets=app_stat['requests_stat']['drop_packet_percentage'], is_done=done, reward=r)

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
      stat.to_csv("learning_stat.csv", mode='a', sep=',', encoding='utf-8', header=False, index=False)
      logging.info("[%s] stat saved" % (scaler.get_scaling_name()))
      stat = stat[0:0]
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
  stat.to_csv("learning_stat.csv", mode='a', sep=',', encoding='utf-8', header=False, index=False)
  logging.info("[%s] stat saved" % (scaler.get_scaling_name()))

if __name__ == '__main__':
  dqn_scaling(
    AutoScaler(
    scaling_info=DQN_ScalingInfo(
        sfc_name='docker',
        scaling_name='DQN',
        slo=None,
        duration=0,
        interval=10,
        save_model_interval=10
    ), 
    type='dqn')
  )
