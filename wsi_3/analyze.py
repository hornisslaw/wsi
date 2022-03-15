import numpy as np

results = {
    "games": 0,
    "depth": [],
    "fsc": [],
    "winners": [],
    "total_moves": [],
}


def load_results():
    with open("results.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            else:
                temp = line.split(":")
                results["depth"].append(int(temp[0][-1]))
                results["fsc"].append(int(temp[1]))
                results["winners"].append(temp[2])
                results["total_moves"].append(int(temp[3]))
            results["games"] = len(results["winners"])
    return lines


def prercentage_win(winners):
    return winners.count("fox") / len(winners)


load_results()
depth = int(np.average(results["depth"]))
fsc = int(np.average(results["fsc"]))
winners = prercentage_win(results["winners"])
avg = np.average(results["total_moves"])
games = results["games"]

print(
    f"Results for {games} games: \n"
    f"Depth={depth},\n"
    f"Fsc={fsc},\n"
    f"Fox={winners*100}% winratio,\n"
    f"Hounds={(1-winners)*100}% winratio,\n"
    f"Average moves per game={avg}.\n"
)
