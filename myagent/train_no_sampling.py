from no_sampling import FullTreeCFRTrainer

trainer = FullTreeCFRTrainer(
    max_q          = 10,
    price_buckets  = 2,    # {-1,0,+1}
    depth          = 5,    # exhaust game tree up to 5 rounds
    alpha          = 0.5,
    eps_converge   = 2e-4, # convergence tolerance
    max_iters      = 50_000
)
trainer.train_until_converged()
policy = trainer.average_strategy()
pathlib.Path("cfr_fulltree.policy.json").write_text(json.dumps(policy))