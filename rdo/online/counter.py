import numpy as np
from online.template import *
from offline.greedy import *
from offline.states import *

#from offline.new_states import *


class CounterManager(OnlineTemplate):
    """RANDOM with state modification"""

    def __init__(self, state_gen, alpha, temperature=1, lag=0, smoothing=1e-2, output_dir=""):
        super().__init__(state_gen, alpha)
        self.counters = [0] * len(self.states)
        self.cost_est = [0] * len(self.states)
        self.new_counters = []
        self.smoothing = smoothing
        self.temperature = temperature
        self.lag = lag
        self.switch_countdown = 0
        self.use_sample = not self.tree_builder.args.load
        self.num_states = []
        self.num_removed = []
        self.out_dir = "resources/labels/sw/" + output_dir
        self.smi_way = 0

    # New phase: include states that were added during the last phase
    def _start_new_phase(self):
        new_states = []
        new_cost_est = []
        # self.to_delete = self.state_gen.detect_redundant_states(self.idx)
        # Delete states from the last phase
        for idx, s in enumerate(self.states):
            if not idx in self.to_delete:
                new_states.append(s)
                new_cost_est.append(self.counters[idx])
                # Update index pointer in the new state space
                if self.curr_state == str(s):
                    self.idx = len(new_states) - 1
        # Add new states
        for idx, s in enumerate(self.new_states):
            if not (idx + len(self.states)) in self.to_delete:
                new_states.append(s)
                new_cost_est.append(self.new_counters[idx])
        self.states = new_states
        self.cost_est = new_cost_est
        # Reset counters
        self.num_states.append(len(self.states))
        self.num_removed.append(len(self.to_delete))
        self.state_gen.delete_states(self.to_delete)
        self.counters = [0] * len(self.states)
        self.new_states = []
        self.new_counters = []
        self.to_delete = []

    # End phase when all counters reach unit movement cost
    def _end_phase(self):
        return np.min(self.counter) >= self.alpha

    def add_state(self, state):
        super().add_state(state)
        self.new_counters.append(np.median(self.counters))

    # When counter is full, switch to a non-full counter with uniform probability
    def change_state(self):
        np.random.seed()

        indices = np.where(np.array(self.counters) < self.alpha)[0]
        if len(indices) == 0:
            self._start_new_phase()
            # Heuristic: stay in current state in the new phase if current state performs well
            if self.cost_est[self.idx] > np.average(self.cost_est) + self.smoothing:
                indices = list(range(len(self.states)))
            else:
                indices = [self.idx]
        else:
            self.cost_est = self.counters

        cost_est = (np.array(self.cost_est)[indices] + self.smoothing).flatten()
        weights = (np.max(cost_est) - cost_est) / (np.sum(np.max(cost_est) - cost_est) + self.smoothing)
        weights = weights ** self.temperature / np.sum(weights ** self.temperature)

        if np.any(weights != weights):
            weights = np.ones(cost_est.shape) * 1. / cost_est.shape[0]

        new_idx = np.random.choice(indices, 1, p=weights)[0]

        pars = self.cal_pars()
        par = pars[new_idx]
        # Check whether we have moved to a different state
        if self.curr_state != str(self.states[new_idx]):
            # print("[T=%d] Decide to change state: %d=>%d" % (self.T, self.idx, new_idx))
            self.idx = new_idx
            self.curr_state = str(self.states[new_idx])
            self._build_layout(self.states[new_idx])
            self.switch_countdown = self.lag
            self.schedule["par"].append(par)

    # Process new queries and update counters
    def process_queries(self, new_queries):
        new_states = self.state_gen.process_queries(new_queries)
        for state in new_states:
            self.add_state(state)

        # print("now states:",len(self.states))
        for q in new_queries:
            self.switch_countdown -= 1
            # Update all counters with query costs
            for j, tree in enumerate(self.states):
                # print(tree,tree.sort_cols)
                # for part in tree.parts:
                # print(part.meta)
                read, _ = tree.eval([q])
                # print(read,_)
                self.counters[j] += read

            # print("-----------")

            for j, tree in enumerate(self.new_states):
                read, _ = tree.eval([q])
                self.new_counters[j] += read

            # Check if need to change state
            if self.counters[self.idx] >= self.alpha:
                self.change_state()
            if self.switch_countdown == 0:
                # print("[T=%d] switching state" % (self.T) )
                self.switch_layout()

            read, pids = self.curr.eval([q])
            self.query_cost += read
            self.schedule["query"].extend(pids)
            self.T += 1

    def cal_pars(self):
        rates = []
        now_path = self.out_dir + "/" + str(self.idx) + ".p-label"
        bids = pickle.load(open(now_path, "rb"))
        # min = 1
        for j, tree in enumerate(self.states):
            new_path = self.out_dir + "/" + str(j) + ".p-label"
            new_bid = pickle.load(open(new_path, "rb"))
            comparison = bids == new_bid
            same_count = np.sum(comparison)
            if self.smi_way == 0:
                rate = 1 - same_count / len(bids)
            else:
                rate = 1 - same_count / (2 * len(bids) - same_count)
            # if rate < min and rate != 0:
            # min = rate
            rates.append(rate)
        '''
        max_value = np.max(rates)
        min = np.min(rates)
        if len(rates) != 1:
            nor = [(x - min) / (max_value - min) for x in rates]
        else:
            nor = [0.5]
        '''
        pars = [0.4 * x + 0.8 for x in rates]
        # pars = [x + 0.2 for x in rates]
        return pars
