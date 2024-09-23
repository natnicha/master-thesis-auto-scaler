import math
import os
from pprint import pprint
import subprocess
import time
import datetime as dt

def reward_calculator(service_info, response_time):
    alpha = 1.0 # weight1
    beta =  1.0 # weight2
    gamma = 1.5 # weight3

    response_time = response_time/1000.0
    loss = service_info["drops"]/service_info["packets"] if service_info["packets"] != 0 else 1
    inst_count = service_info["size"]/(service_info["num_types"]*5)

    reward = -((alpha*math.log(1+response_time)+(beta*math.log(1+loss))+(gamma*math.log(1+inst_count))))

    print("reward value 1 : {}".format(alpha*math.log(1+response_time)))
    print("reward value 2 : {}".format(beta*math.log(1+loss)))
    print("reward value 3 : {}".format(gamma*math.log(1+inst_count)))


    return reward


# measure_response_time(): send http requests from a source to a destination
# Input: scaler
# Output: response time
def measure_response_time(scaler, name):
    cnd_path = os.path.dirname(os.path.realpath(__file__))

    dst_ip = get_ip_from_id(scaler.get_monitor_dst_id())
    src_ip = get_ip_from_id(scaler.get_monitor_src_id())

    command = ("sshpass -p %s ssh -o stricthostkeychecking=no %s@%s ./test_http_e2e.sh %s %s %s %s %s" % (
        cfg["traffic_controller"]["password"],
        cfg["traffic_controller"]["username"],
        cfg["traffic_controller"]["ip"],
        src_ip,
        cfg["instance"]["username"],
        cfg["instance"]["password"],
        cfg["traffic_controller"]["num_requests"],
        dst_ip))

    command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"

    # Wait until web server is running
    start_time = dt.datetime.now()

    while True:
        time.sleep(10)
        response = subprocess.check_output(command, shell=True).strip().decode("utf-8")
        print(response)
        if response != "":
            pprint("[%s] %s" % (scaler.get_scaling_name(), response))
            f = open("test_monitor-"+name+".txt", "a+", encoding='utf-8')
            f.write(str(response)+'\n')
            f.close()
            print("write done")
            return float(response)
        elif (dt.datetime.now() - start_time).seconds > 60 or scaler.get_active_flag() == False:
            scaler.set_active_flag(False)
            return -1
