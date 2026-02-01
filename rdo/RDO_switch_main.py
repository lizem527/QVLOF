from utils.setup import *
from utils.tree import *
from utils.config import *
from offline.states import *
from online.counter import *
import numpy as np
import argparse
import os


def run_random(fname, k, output_dir, N=3):
    query = []
    movement = []
    schedule = {}
    T = len(queries) // args.interval
    for trials in range(N):
        tb = TreeBuilder(df, df_sample, config, args, k, output_dir)
        init_states = tb.get_init_states(queries)
        sg = StateGenerator(tb, init_states, args.interval, eps, True)
        sg.reset_reservoir(args.res)
        cm = CounterManager(sg, alpha, args.gamma, args.lag, 1e-2, output_dir)

        # Run randomized algorithm
        for i in range(T):
            new_queries = queries[i * args.interval:(i + 1) * args.interval]
            cm.process_queries(new_queries)
        print("q: %f, m: %f" % (cm.query_cost, cm.movement_cost))
        print("Avg #states: %f (max: %d), avg removed: %f" % (
            np.average(cm.num_states), max(cm.num_states), np.average(cm.num_removed)))

        query.append(cm.query_cost)
        movement.append(cm.movement_cost)
        schedule["%s-%d" % (fname, trials)] = cm.schedule
    return schedule, query, movement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random.')
    parser.add_argument('--config', default="demo", help="Config File Path")
    args = Args(parser.parse_args().config)
    # Fix the random seeds for our experiments
    np.random.seed(args.seed)
    np.seterr(invalid='ignore')
    alpha = args.alpha
    eps = args.eps
    print("alpha=%d, epsilon=%.3f" % (alpha, eps))

    fnames, files, parts, config = setup_perfile(args)
    # queries = get_workload_perfile(config, files, fnames, args)
    # print("# queries: %d" % len(queries))
    total_query = 0
    total_movement = 0
    total_size = 0
    for i, fname in enumerate(fnames):
        print(fname)
        df, df_sample, k = get_data(config, args, parts, files[i], fname)
        N = len(df)
        output_dir = "%s/%s-%s-%d-%d-%s" % (config["ds"], fname, args.q, args.interval, k, args.method)
        queries = new_get_workload(config, files, fname, args)
        print("# queries: %d" % len(queries))

        # Run randomized algorithm
        schedule, query, movement = run_random(fname, k, output_dir, 1)

        total_size += N
        total_query += np.average(query) * N
        total_movement += np.average(movement) * N
        print("[%s] Query: %f, %f, Movement: %f, %f" % (
            fname, np.average(query), np.std(query), np.average(movement), np.std(movement)))
        if args.policy == "oracle":
            pickle.dump(schedule, open(
                "resources/schedule/random/%s-%s-%s-%d-%s-%d-oracle-%d-%d.p" % (config["ds"],
                                                                                fname, args.q, args.k, args.method,
                                                                                args.alpha, args.gamma, args.lag),
                "wb"))
        else:
            pickle.dump(schedule, open(
                "resources/schedule/random/%s-%s-%s-%d-%s-%d-%.2f-%d-%d.p" % (config["ds"],
                                                                              fname, args.q, args.k, args.method,
                                                                              args.alpha, args.eps, args.gamma,
                                                                              args.lag), "wb"))
    print("[Random (%s,%d)] Query: %f, Movement: %f" % (
        args.policy, args.interval, total_query / total_size, total_movement / total_size))
    print("Total size: %d" % total_size)
