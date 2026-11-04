import json, sys, glob, os
keys = ['ce_gen','ce_recon','frac_exhausted','frac_exhausted_gen','g_mean','active_slots','slot_norm_mean','eff_rank_v','L_jepa']
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(f"runs/*/{pat}/final.json")):
        d = json.load(open(f)); n = os.path.basename(os.path.dirname(f))
        print(n.ljust(22), ' '.join(f'{k}={d[k]:.3f}' for k in keys if k in d))
