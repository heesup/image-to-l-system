import subprocess

res = subprocess.run(["squeue", "-j", "38142709"], capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
