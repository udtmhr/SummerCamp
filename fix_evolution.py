import re

with open("luxai2021/rl/evolution.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == "canonical_":
        continue
    if "pop(\"kl_coefficient\"" in line:
        continue
    if "Always set ppo_config.kl_coefficient" in line:
        line = line.replace("ppo_config.kl_coefficient and", "")
    new_lines.append(line)

with open("luxai2021/rl/evolution.py", "w") as f:
    f.write("".join(new_lines))
print("Fixed evolution.py syntax error and popped lines")
