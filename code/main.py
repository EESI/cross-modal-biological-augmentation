from config import CFG
from pipeline import run_ehr, run_empo500

# ======================================================================
# MAIN
# ======================================================================

if CFG["mode"] == "ehr":
    run_ehr(CFG["ehr"])
elif CFG["mode"] == "empo500":
    run_empo500(CFG["empo500"])
elif CFG["mode"] == "both":
    run_ehr(CFG["ehr"])
    run_empo500(CFG["empo500"])
else:
    print("Unknown CFG['mode']:", CFG["mode"])
