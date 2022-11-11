import subprocess
from pathlib import Path


spmv_variants = ["work_oriented", "group_mapped", "original", "thread_mapped"]

# Modify the following lines to change the parameters of the experiment.
executable_path = "~/loops/build/bin"
dataset_path = "/data/gunrock/gunrock_dataset/mario-2TB/large"
datasets = ["hollywood-2009"]

# Build commands for each variand and dataset.
commands = []
for variant in spmv_variants:
    for dataset in datasets:
        commands.append(str(executable_path) + "/loops.spmv." + str(variant) + " -m " + str(dataset_path) + "/" + str(dataset) + "/" + str(dataset) + ".mtx")

# Run all commands.
for command in commands:
    popen = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    popen.wait()
    output = popen.stdout.read()
    print(output)
