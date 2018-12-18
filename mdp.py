from dist import DDist
import itertools
class MDP:
    def __init__(self, transition_model, reward_model, states, actions,gamma=0.9):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.states = states
        self.gamma = gamma
        self.actions = actions
        
    def value_iteration(self, policy, eps = 0.0001, max_iters=1000):
        q = TabularQ(self.states)
        r = self.reward_model
        t = self.transition_model
        for _ in range(max_iters):
            qp = q.copy()
            for s in q.states:
                    ap = policy(s)
                    v = r(s) + self.gamma * t(s, ap).expectation(lambda sp: q.get(sp))
                    qp.set(s, v)
            if max(qp.get(s) - q.get(s) for s in q.states) < eps:
                print("Breaking at", _)
                break
            q = qp
        return qp

#def expectation(self, f):
#        return sum(self.prob(x) * f(x) for x in self.support())
'''
return optimal control for time horizon T
'''
def MPC(mdp, s0, T):
    #tbh iterate through all possible action combinations of T and get the highest reward
    it = itertools.product(mdp.actions, repeat=T)
    cost = dict((action_list, discounted_cost(mdp, s0,action_list,T)) for action_list in it)
    #import ipdb; ipdb.set_trace()
    best_list = max(cost, key=lambda x: cost[x])
    #if s0 == 's0':
    #    print("best_list from s0 with ",T, best_list)
    return best_list[0]

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

def avg_dist(q1, q2):
    dists = []
    import numpy as np
    for s in q1.states:
        dists.append(abs(q1.get(s)-q2.get(s)))
    return  np.mean(dists)
       
 
def main():
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
    def policy_1(s):
        return MPC(mdp, s, 1)
    def policy_2(s):
        return MPC(mdp, s, 2)
    v1 = mdp.value_iteration(policy_1)
    v2 = mdp.value_iteration(policy_2)
    print(v1.q)
    print(v2.q)
    print(avg_dist(v1,v2))
    

if __name__ == "__main__":
    main()
