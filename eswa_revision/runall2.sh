cd /tmp/claude-0/run
export OMP_NUM_THREADS=1
python3 proto2.py --scen baseline --workers 2 > r2_baseline.log 2>&1
python3 refs2.py > r2_refs.log 2>&1
python3 alerts2.py > r2_alerts.log 2>&1
python3 proto2.py --scen era5 --workers 2 > r2_era5.log 2>&1
python3 proto2.py --scen operational --workers 2 > r2_oper.log 2>&1
python3 baselines2.py > r2_baselines.log 2>&1
python3 ablate2.py > r2_ablate.log 2>&1
python3 fill_audit.py > r2_fill.log 2>&1
python3 proto2.py --scen baseline --workers 2 --no-purge > r2_nopurge.log 2>&1
echo DONE > r2_done.flag
