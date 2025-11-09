    length = len(ep["reward"]) - 1
    score = float(ep["reward"].astype(np.float64).sum())
    logger.add({"length": length, "score": score}, prefix="episode")
    print(f"Episode has {length} steps and return {score:.1f}.")
    stats = {}
    # for key in ep:
    #   if 'custom' in key:
    #     stats[key] = ep[key]
    # for key in args.log_keys_video:
    #   if key in ep:
    #     stats[f"policy_{key}"] = ep[key]
    custom_values = ['stages', 'condition_1', 'condition_2', 'condition_3',
             'gradients_exact', 'mu_gradients']
    def log(key, value):
      if key == 'log_surprise_mean':
        stats['log_surprise_mean'] = value[10] # Set value index to wanted.
      if key in custom_values:
        stats[key] = np.round(value, decimals=4)
      if re.match(args.log_keys_sum, key):
        stats[f"sum_{key}"] = value.sum()
      if re.match(args.log_keys_mean, key):
        stats[f"mean_{key}"] = value.mean()
      if re.match(args.log_keys_max, key):
        stats[f"max_{key}"] = value.max(0).mean()
    
    debug_info = {
      "stages": [],
      "reconstruction_error_1": [],
      "reconstruction_error_2": [],
      "reconstruction_error_3": [], 
    }
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      log(key, value)
    
    for key, value in ep_info.items():
      log(key, value)

    logger.add(metrics.result())
    logger.add(timer.stats(), prefix="timer")
    logger.write(fps=True)

    metrics.add(stats, prefix="stats")