import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.argparsers.simulationargparser import SimulationArgumentParser
from utils.datasimulation import DataSimulation
from algorithms.PairRank.PairRank import PairRank
from algorithms.PairRank.olRankNet import olRankNet
from algorithms.PairRank.olLambdaRank import olLambdaRank
from algorithms.PairRank.FairExpPairRank import FairExpPairRank


def func_pairrank(args, dir_name):
    ranker_params = {"learning_rate": args.lr, "learning_rate_decay": args.lr_decay, "update": args.update,
                     "_lambda":       args._lambda, "alpha": args.alpha, "refine": args.refine, "rank": args.rank,
                     "ind":           args.ind, }
    sim_args, other_args = parser.parse_all_args(ranker_params)
    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.lr, args.lr_decay, args.update, args._lambda, args.alpha,
                                                      args.refine, args.rank, args.ind, args.seed, )

    run_name = dir_name + ranker_name
    ranker = [(run_name, PairRank, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_olranknet(args, dir_name):
    ranker_params = {"learning_rate": args.lr, "learning_rate_decay": args.lr_decay, "update": args.update,
                     "_lambda":       args._lambda, "alpha": args.alpha, "refine": args.refine, "rank": args.rank,
                     "ind":           args.ind, "mlp_dims": args.mlpdims, "batch_size": args.batch_size,
                     "epoch":         args.epoch}

    sim_args, other_args = parser.parse_all_args(ranker_params)
    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.lr, args.lr_decay, args.update, args._lambda,
                                                               args.alpha, args.refine, args.rank, args.ind,
                                                               args.mlpdims, args.batch_size, args.epoch, args.seed, )

    run_name = dir_name + ranker_name
    ranker = [(run_name, olRankNet, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_ollambdarank(args, dir_name):
    ranker_params = {"learning_rate": args.lr, "learning_rate_decay": args.lr_decay, "update": args.update,
                     "_lambda":       args._lambda, "alpha": args.alpha, "refine": args.refine, "rank": args.rank,
                     "ind":           args.ind, "mlp_dims": args.mlpdims, "batch_size": args.batch_size,
                     "epoch":         args.epoch}

    sim_args, other_args = parser.parse_all_args(ranker_params)
    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.lr, args.lr_decay, args.update, args._lambda,
                                                               args.alpha, args.refine, args.rank, args.ind,
                                                               args.mlpdims, args.batch_size, args.epoch, args.seed, )

    run_name = dir_name + ranker_name
    ranker = [(run_name, olLambdaRank, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_fairexppairrank(args, dir_name):
    ranker_params = {"learning_rate": args.lr, "learning_rate_decay": args.lr_decay, "update": args.update,
                     "_lambda":       args._lambda, "alpha": args.alpha, "refine": args.refine, "rank": args.rank,
                     "ind":           args.ind, "decay_mode": args.decay_mode, "unfairness": args.unfairness,
                     "fair_alpha":    args.fair_alpha, "fair_epsilon": args.fair_epsilon}
    sim_args, other_args = parser.parse_all_args(ranker_params)
    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.lr, args.lr_decay, args.update, args._lambda, args.alpha,
                                                      args.refine, args.rank, args.ind, args.seed, )

    run_name = dir_name + ranker_name
    ranker = [(run_name, FairExpPairRank, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def set_sim_and_run(args):
    cm = args.click_models[0]
    n_impr = args.n_impressions
    n_results = args.n_results
    algo = args.algo.upper()
    dir_name = "algo/{}/{}/{}/{}/".format(algo, cm, n_impr, n_results)

    switcher = {"PAIRRANK":         lambda: func_pairrank(args, dir_name),
                "OLRANKNET":        lambda: func_olranknet(args, dir_name),
                "OLLAMBDARANK":     lambda: func_ollambdarank(args, dir_name),
                'FAIREXP_PAIRRANK': lambda: func_fairexppairrank(args, dir_name)}

    return switcher.get(algo, lambda: "ERROR: algorithm type not valid")()


# get input parameters
if __name__ == "__main__":
    DESCRIPTION = "Run script for testing framework."
    parser = SimulationArgumentParser(description=DESCRIPTION)
    input_args = parser.parse_sim_args()
    set_sim_and_run(input_args)
