# v3: stable recipe + 3 seeds

With the v2 recipe (5-epoch LR schedule, early stop after 4 evaluations without improvement) no run finished its
first epoch: selection happened at peak LR, where event F1 on the selection games jumps by +-0.05 between consecutive
evaluations, and changing only the seed moved test event F1 by 0.073 (v2_delays_views_hardneg: 0.576 seed 42,
0.503 seed 1). v3 trains a fixed 2 epochs with the LR decaying linearly to 0, evaluates every 1000 steps, never stops
early, and runs every configuration with seeds 42, 1 and 2 (report mean +- std).
