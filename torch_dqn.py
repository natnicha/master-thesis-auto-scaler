import collections
import random
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

# Experience memory
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

    def writeToFile(self, file_name, transition):
        f = open(file_name, 'a', newline='')
        wr = csv.writer(f)
        wr.writerow(transition)
        f.close()

    def readFromFile(self, file_name):
        f= open(file_name, 'r', newline='')
        rdr = csv.reader(f)

        for line in rdr:
            line[0] = line[0].replace('[ ', '')
            line[0] = line[0].replace('[', '')
            line[0] = line[0].replace(']', '')
            line[0] = line[0].split(' ')

            while '' in line[0]:
                line[0].remove('')

            temp = []
            for value in line[0]:
                temp.append(float(value))
            line[0] = temp

            line[3] = line[3].replace('[ ', '')
            line[3] = line[3].replace('[', '')
            line[3] = line[3].replace(']', '')
            line[3] = line[3].split(' ')

            while '' in line[3]:
                line[3].remove('')

            temp = []
            for value in line[0]:
                temp.append(float(value))
            line[3] = temp

            transition = (np.array(line[0]), int(line[1]), float(line[2]), np.array(line[3]), float(line[4]))
            self.put(transition)

# Q-network
class Qnet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons):
        super(Qnet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_neurons = num_neurons

        self.fc1 = nn.Linear(self.num_inputs, self.num_neurons) # input - hidden layer 1
        self.fc2 = nn.Linear(self.num_neurons, self.num_neurons) # hidden layer 1 - hidden layer 2
        self.fc3 = nn.Linear(self.num_neurons, self.num_neurons) # hidden layer 2 - hidden layer 3
        self.fc4 = nn.Linear(self.num_neurons, self.num_neurons) # hidden layer 3 - hidden layer 4
        self.fc5 = nn.Linear(self.num_neurons, self.num_outputs) # output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    # Epsilon greedy implementation
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()

        # Regarding coin to select action randomly
        if coin < epsilon:
            return { "action": random.randrange(0,self.num_outputs), "by" : "Policy" }
        else :
            return { "action": out.argmax().item(), "by": "Random" }

    def save_model(self, path):
        torch.save(self.state_dict(), path)


# Update Target Q-network
def train(q, q_target, memory, optimizer, gamma, batch_size):
    for _ in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
