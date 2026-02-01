import numpy as np
from online.template import *
from offline.greedy import *


def wf_onestep(opt, costs, alpha):
    new_opt = []
    for j in range(len(costs)):
        min_idx = opt.index(min(opt))
        cost_self = opt[j] + costs[j]
        new_opt.append(min(cost_self, opt[min_idx] + costs[j] + alpha))
    min_idx = new_opt.index(min(new_opt))
    return min_idx, new_opt


class WorkFunction(OnlineTemplate):
    def __init__(self, tree_builder, states, alpha):
        super().__init__(tree_builder, states, alpha)
        # Optimal cost vector
        self.opt = [alpha] * len(states)
        self.init_idx = self.idx
        # Init state has no initial movement cost
        self.opt[self.idx] = 0

    def _get_cost(self, states, new_queries):
        # Cost on current query batch
        costs = []
        for tree in states:
            read = tree.eval(new_queries, False)
            costs.append(sum(read))
        return costs

    def add_states(self, new_states):
        self.states.extend(new_states)
        # new WFA cost = min_{s} WFA(s) + dist(s, new_state) for all s in the old states
        min_opt = min(self.opt)
        self.opt.extend([min_opt + self.alpha] * len(new_states))

    # Process new queries and update counters
    def process_queries(self, new_queries):
        self.query_cost += self.curr.eval(new_queries) * len(new_queries)
        # Update work function algorithm
        costs = self._get_cost(self.states, new_queries)
        min_idx, self.opt = wf_onestep(self.opt, costs, self.alpha)
        # Need to change layout
        if min_idx != self.idx:
            self.idx = min_idx
            self._build_layout()
