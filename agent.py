import logging
import math

def reward_calculator(service_stat):
    w_resp = 0.33 # weight of response time
    w_drop =  0.33 # weight of dropped packet percentage
    w_instance = 0.33 # weight of a number of instance
    
    inst_count = int(service_stat['containers']['system']['online_pods'])
    logging.info(f"s'pod: {inst_count}")
    logging.info(service_stat['requests_stat'])
    logging.info(f"drop_packet_pct: {service_stat['requests_stat']['drop_packet_percentage']}  {math.log(1+service_stat['requests_stat']['drop_packet_percentage'], 2)}")
    reward = -((w_resp*math.log(1+service_stat['requests_stat']['avg_latency']/1000.0, 2)+(w_drop*math.log(1+service_stat['requests_stat']['drop_packet_percentage'], 2))+(w_instance*math.log(inst_count, 2))))

    logging.info(f"reward value 1 : {w_resp*math.log(1+service_stat['requests_stat']['avg_latency']/1000.0, 2)}")
    logging.info(f"reward value 2 : {w_drop*math.log(1+service_stat['requests_stat']['drop_packet_percentage'], 2)}")
    logging.info(f"reward value 3 : {w_instance*math.log(inst_count, 2)}")
    logging.info(f"sum reward : {reward}")
    return reward
