from dist import DDist
import ipdb;
import mdptoolbox as mdpt
import numpy as np
import itertools
class MDP:
    def __init__(self, transition_model, reward_model, states, actions,gamma=0.4):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.states = states
        self.gamma = gamma
        self.actions = actions

    def optimal_values(self):
        #transform everything into a state that MDP likes
        P = np.zeros((len(self.actions), len(self.states), len(self.states)))
        R = np.zeros((len(self.states),len(self.actions)))
        for j in range(len(self.states)): #numstates
            R[j, :] = self.reward_model(self.states[j]) #same for all actions
            for i in range(len(self.actions)): #numactions
                trans_ddist = self.transition_model(self.states[j],self.actions[i])
                for k in range(len(self.states)):
                    P[i,j,k] = trans_ddist.prob(self.states[k])
        vi = mdpt.mdp.ValueIteration(P, R, self.gamma)
        vi.run()
        value_function = vi.V
        return value_function
        
    def policy_value_iteration(self, policy, eps = 0.1, max_iters=1000):
        q = TabularQ(self.states)
        r = self.reward_model
        t = self.transition_model
        unsolved = True
        cache = {}
        for _ in range(max_iters):
            qp = q.copy()
            for s in q.states:
                    ap, cache = policy(s, reset = unsolved, cache=cache)
                    v = r(s) + self.gamma * t(s, ap).expectation(lambda sp: q.get(sp))
                    qp.set(s, v)
            unsolved = False
            if max(qp.get(s) - q.get(s) for s in q.states) < eps:
                #print("Breaking at", _)
                break
            q = qp
        return qp
def make_more_sparse(a, p_drop):
    # a is input array
    # n is number of non-zero elements to be reset to zero
    n = int(round(p_drop*a.shape[0]))
    idx = np.flatnonzero(a) # for performance, use np.flatnonzero(a!=0)
    np.put(a, np.random.choice(idx, n, replace=False),0)
    return a
"""
I know this is a terrible idea but I'm out of good ones
"""
def gen_random_MDP(num_states, num_actions, p_drop = 0):
    states = np.linspace(0,num_states-1,num_states)
    actions = np.linspace(0,num_actions-1,num_actions)
    transition_matrix = np.zeros((num_states, num_actions, num_states))
    for i in range(num_states):
        for j in range(num_actions):
            rand_dist = np.random.random(states.shape)
            #randomly make some 0
            rand_dist = make_more_sparse(rand_dist, p_drop) 
            rand_dist /= sum(rand_dist)
            if sum(rand_dist) == 0:
                ipdb.set_trace()
                rand_dist[np.random.randint(rand_dist.shape[0])] = 1
            transition_matrix[i,j] = rand_dist
    reward_matrix = np.random.random(states.shape)
    def transition_model(s,a):
        matrix = transition_matrix[int(s),int(a)]
        dictionary = dict((states[i], matrix[i]) for i in range(matrix.shape[0]))
        return DDist(dictionary)

    def reward_model(s):
        return reward_matrix[int(s)]
    return MDP(transition_model, reward_model, states, actions)
   
#def expectation(self, f):
#        return sum(self.prob(x) * f(x) for x in self.support())
'''
return optimal control for time horizon T
'''
def MPC(mdp, s0, T, reset = True, cache = {}):
    if not reset:
        return cache[s0], cache
    else:
        #tbh iterate through all possible action combinations of T and get the highest reward
        it = itertools.product(mdp.actions, repeat=T)
        cost = dict((action_list, discounted_cost(mdp, s0,action_list,T)) for action_list in it)
        #import ipdb; ipdb.set_trace()
        best_list = max(cost, key=lambda x: cost[x])
        #if s0 == 's0':
        #    print("best_list from s0 with ",T, best_list)
        cache[s0] = best_list[0]
        return best_list[0], cache

def discounted_cost(mdp, s,action_list, T):
    cost = mdp.reward_model(s)
    for i in range(T):
        for s_prime in mdp.states:
            a = action_list[0]
            short_action_list = action_list[1:] 
            cost += mdp.gamma**i*mdp.transition_model(s,a).prob(s_prime)*discounted_cost(mdp, s_prime, short_action_list, T-1)
    return cost
    

class TabularQ:
    def __init__(self, states):
        self.states = states
        self.q = dict([((s), 0.0) for s in states])
    def copy(self):
        q_copy = TabularQ(self.states)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, v):
        self.q[(s)] = v
    def get(self, s):
        return self.q[(s)]

def avg_dist(sub_tuple, q2):
    dists = []
    import numpy as np
    for i,s in zip(range(len(sub_tuple)), q2.states):
        dists.append(abs(sub_tuple[i]-q2.get(s)))
    return  np.mean(dists)


def gen_toy_MDP():
    states = ['s0', 's1', 's2','s3']
    actions = ['a1', 'a2']
    def transition_model(s,a):
        if s == 's1': #doesnt matter what the action is
            return DDist({'s3':1}) 
        elif s == 's3':
            return DDist({'s3':1}) 
        elif s == 's2':
            return DDist({'s2':1}) 
        elif s == 's0':
            if a == 'a1':
                return DDist({'s2':1}) 
            else:
                return DDist({'s1':1}) 
           
 
        else:
            return ValueError()

    def reward_model(s):
        if s == 's0' :
            return 0
        elif s == "s1":
            return 1
        elif s == "s2":
            return 2
        elif s == "s3":
            return 8
        else:
            return 0

    mdp = MDP(transition_model, reward_model, states, actions)
    return mdp
   
def run_test(num_states=5, num_actions=2, tau=2,p_drop = 0):
    mdp = gen_random_MDP(num_states,num_actions, p_drop=p_drop)
    def policy_1(s, reset=False, cache={}):
        return MPC(mdp, s, tau, reset=reset, cache=cache)
    optimal_vi = mdp.optimal_values()
    v1 = mdp.policy_value_iteration(policy_1)
    return avg_dist(optimal_vi,v1)
 
def main():
    avg_dists = []
    num_trials = 3000
    for i in range(num_trials):
        avg_dists.append(run_test(num_states=6, num_actions=2, tau=2, p_drop = 0.8))
        if i % 1000 == 0:
            print("Iteration ", i)
    print("Stats: ")
    print("Mean: ", np.mean(avg_dists))
    print("Stdev: ", np.std(avg_dists))
    

if __name__ == "__main__":
    main()
