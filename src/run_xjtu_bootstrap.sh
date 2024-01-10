#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

datasets=("xjtu")
corrs=("bootstrap")
echo "=============================================================================================="
echo "Starting script"
echo "=============================================================================================="
for dataset in ${datasets[@]}; do
        for corr in ${corrs[@]}; do
                echo "Starting dataset run <$dataset> <$corr>"
                python $base_path/test_find_resume_NCV.py --dataset $dataset --typedata $corr
                rm $base_path/data/XJTU-SY/csv/$corr/*
                echo "Tuning dataset <$dataset> <$corr> done"
                echo -e "\n\n\n\n\n"
                echo "=============================================================================================="
                echo -e "\n\n\n\n\n"
        done
done
echo "Finished executing datasets"