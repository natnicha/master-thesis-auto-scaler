# Welcome to RL-Based-Autoscaler

The main repository for Reinforcement Learning (RL)-based Adaptive Horizontal Autoscaler (AHPA). This repository is [Python](https://www.python.org/)-based and leverages [PyTorch](https://pytorch.org/) to implement Deep Q-Networks (DQN). It includes an agent and its learning procedure, with an ADAM optimizer set as the default.

## About Project

A research under a title of `Adaptive Horizontal Pod Autoscaling (AHPA) Based on Reinforcement Learning in Kubernetes for Machine Learning`, introducing the Adaptive Horizontal Pod Autoscaler (AHPA), which utilizes RL with a Deep Q-Network (DQN) to dynamically adjust the number of Kubernetes Pods for horizontal scaling, enabling both scaling in and scaling out. We evaluate the performance and reliability of AHPA in image classification tasks, comparing its effectiveness against a traditional horizontal autoscaler in Kubernetes.

## Project Components

This project consists of the following three components, distributed across different repositories, working together seamlessly.

- [**RL-Based Autoscaler**](https://github.com/natnicha/master-thesis-auto-scaler): The main repository for RL-based Adaptive Horizontal Autoscaler (AHPA), implemented by Deep Q-Networks (DQN). It includes an agent and its learning procedure, with an ADAM optimizer set as the default.
- [**Docker-Manipulation-API**](https://github.com/natnicha/master-thesis-docker-manipulation-API): The service facilitates an RL agent by enabling seamless communication between the RL agent and the service running on Kubernetes as part of our research.
- [**Image-Classification**](https://github.com/natnicha/master-thesis-image-classification): The target application in our study, image classification, serves image classification application based on user-submitted photos.

## Built With

[<img src="https://www.python.org/static/img/python-logo.png" height="50">](https://www.python.org/) [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" height="50">](https://pytorch.org/) 

<img src="https://img.shields.io/badge/Test-Pass-green"> <img src="https://img.shields.io/badge/Secuiry-Pass-blue">

## Getting Started
To start and test the service, follow the below instruction.

### Setting Up The Environment
1. Installing Python and Dependencies
This repository is developed by Python. So, install [Python](https://www.python.org/) in your working environment.

2. Our repository applies virtual environment development, [Pipenv](https://pipenv.pypa.io/en/latest/). By using this, you can simply install pipenv and create your working environment. Then install all dependencies using `pipenv install` or `pipenv sync`.

### Autoscaler Configuration
The following configuration can be adjusted based on the description provided below.

- **scaling_name** (str=None): The autoscaler number
- **interval** (float=None): The interval for the learning process, specified in seconds. This value will pause the process, putting it to sleep for the specified duration.
- **save_model_interval** (int=None): The saving interval for the learning model, specified as a number of episodes.
- **duration** (float=None): The time limit for the entire learning process, defined in seconds. 
- **start_with_specific_pods_no** (int=None): The initial pod  for the entire learning process, defined in seconds. 
- **is_learn** (bool=None): 
- **has_dataset** (bool=None): 

### Application Configuration
The application configuration is treated in [config.YAML](./config.yaml). This section provides the definition of parameter

```
instance:                               # instance or Pods limitation
  max_number: 5                         # a maxmum of 5 Pods is set
  min_number: 1                         # a minimum of 1 Pod is set
docker:                                 # Docker-Manipulation-API configuration
  base_url: http://localhost:6000       # based URL specified where Docker-Manipulation-API is runing  
logging:                                # logging configuration
  filename: ./agent.log                 # logging filename collecting RL processing progress
```

### Starting the Learning Process
To run this service, you can simply start the lerning process by using the following command.
```
python .\main.py
```


### Testing
You can actively track its learning progress by observing the episode-by-episode updates displayed in your running console. These updates are similar to logging statements, providing key information about the system's behavior, such as performance metrics, reward values, or errors. The following example demonstrates how these statements may appear, helping you to gain insights into the learning process and make adjustments if necessary.

```
2025-01-27 12:33:15,097 INFO learning from live data
2025-01-27 12:33:15,097 INFO  ---------- Episode 0 ---------- 
2025-01-27 12:33:17,471 INFO state (s): [1.4000e-01 1.6546e+02 1.0000e+00]
2025-01-27 12:33:17,506 INFO [DQN] Maintain! by Policy
2025-01-27 12:33:17,507 INFO Epsilon value : 0.5
2025-01-27 12:33:17,508 INFO current instance: 1
2025-01-27 12:34:18,274 INFO post state(s'): [1.4000e-01 1.6546e+02 1.0000e+00]
2025-01-27 12:34:18,275 INFO s'pod: 1
2025-01-27 12:34:18,275 INFO {'avg_latency': 1850.3333333333333, 'drop_packet_count': 0, 'drop_packet_percentage': 0.0, 'total_packet_count': 6}
2025-01-27 12:34:18,276 INFO drop_packet_pct: 0.0  0.0
2025-01-27 12:34:18,276 INFO reward value 1 : 0.49867311307196577
2025-01-27 12:34:18,277 INFO reward value 2 : 0.0
2025-01-27 12:34:18,277 INFO reward value 3 : 0.0
2025-01-27 12:34:18,277 INFO sum reward : -0.49867311307196577
2025-01-27 12:34:48,290 INFO  ---------- Episode 1 ---------- 
...
```

## Contributing
If you have any suggestion that would make our website looks better or more convenience, please fork the repo and create a merge requeste. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thank you again!

1. Fork the Project
2. Create your Feature Branch
    ```
    git checkout -b feature/AwesomeFeature
    ```
3. Commit your Changes
    ```
    git commit -m 'Add some AwesomeFeature'
    ```
4. Push to the Branch
    ```
    git push origin feature/AwesomeFeature
    ```
5. Open a Pull Request

## Acknowledgment
The authors would like to express our sincere gratitude to Dr. habil. Julien Vitay, thesis supervisor from the professorship of Artificial Intelligence (Informatik) at Technische Universitat at Chemnitz, for his expert guidance, unwavering support, and valuable feedback throughout the research and writing process.

We also wish to express our heartfelt appreciation to M.Sc. Florian Zimmer, our research mentor and project advisor from [Fraunhofer-Institut fur Software- und Systemtechnik (ISST)](https://www.isst.fraunhofer.de/). His generous investment of time and effort in providing regular, detailed feedback at every stage of the project was invaluable. Additionally, his insightful advice and guidance were crucial in helping us navigate and overcome the challenges encountered throughout this study. 

Importantly, we would like to gratefully acknowledge the computing time made available to them on the high-performance computer Barnard and Alpha at the, Nationales Hochleistungsrechnen, NHR Center, at Zentrum f¨ur Informationsdienste und Hochleistungsrechnen (ZIH), at Technische Universit¨at Dresden. This center is jointly supported by the Federal Ministry of Education and Research and the state governments participating in the [NHR](www.nhr-verein.de/unsere-partner).

## Project Contributor & Support
This project is exclusively contributed by Natnicha Rodtong. For inquiries, feel free to contact me via [ResearchGate](https://www.researchgate.net/profile/Natnicha-Rodtong) or [email](nat.rodtong@gmail.com).

## Disclaimer
This repository is a component of a master's thesis titled `Adaptive Horizontal Pod Autoscaling (AHPA) Based on Reinforcement Learning in Kubernetes for Machine Learning`. The thesis explores advanced techniques for improving the scalability and efficiency of machine learning workloads in Kubernetes environments using reinforcement learning-based approaches for adaptive horizontal pod autoscaling. The research was conducted at [Laboratory of Artificial Intelligence, Technische Universität Chemnitz (TU Chemnitz)](https://www.tu-chemnitz.de/informatik/KI/index.php.en), Germany, as part of the requirements for completing the Master’s program. 

### Important Notes:
1. No Warranty: This project is provided "as is," without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, or non-infringement.
2. Limitation of Liability: The authors or contributors shall not be held liable for any claim, damages, or other liability arising from the use, misuse, or inability to use the content within this repository.
3. Third-Party Dependencies: This repository may rely on external libraries or tools that are subject to their own licenses. Please ensure compliance with those licenses when using this project.