from config import CFG
from bidirectional_pipeline import run_ehr, run_empo3

# ======================================================================
# MAIN
# ======================================================================

if CFG["mode"] == "ehr":
    run_ehr(CFG["ehr"])
elif CFG["mode"] == "empo3":
    run_empo3(CFG["empo3"])
elif CFG["mode"] == "both":
    run_ehr(CFG["ehr"])
    run_empo3(CFG["empo3"])
else:
    print("Unknown CFG['mode']:", CFG["mode"])
