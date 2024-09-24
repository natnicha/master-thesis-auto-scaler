import logging
import math

def reward_calculator(service_info, response_time):
    alpha = 1.0 # weight1
    beta =  1.0 # weight2
    gamma = 1.5 # weight3

    response_time = response_time/1000.0
    # TODO: find drops and total packets to find loss
    # loss = service_info["drops"]/service_info["packets"] if service_info["packets"] != 0 else 1
    # inst_count = service_info["size"]/(service_info["num_types"]*5)
    loss = 0.01
    inst_count = service_info['instance_num']

    reward = -((alpha*math.log(1+response_time)+(beta*math.log(1+loss))+(gamma*math.log(1+inst_count))))

    logging.info(f"reward value 1 : {alpha*math.log(1+response_time)}")
    logging.info(f"reward value 2 : {beta*math.log(1+loss)}")
    logging.info(f"reward value 3 : {gamma*math.log(1+inst_count)}")
    return reward
