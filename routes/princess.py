import logging
from flask import request, jsonify
from routes import app
import heapq

logger = logging.getLogger(__name__)


def floyd_warshall(n, edges):
    """All-pairs shortest path distances using Floyd-Warshall."""
    INF = 10**12
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
        dist[v][u] = min(dist[v][u], w)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


@app.route("/princess-diaries", methods=["POST"])
def princess_diaries():
    data = request.get_json(force=True)
    tasks_raw = data.get("tasks", [])
    subway_raw = data.get("subway", [])
    start_station = data.get("starting_station", 0)

    # Parse tasks
    tasks = []
    for t in tasks_raw:
        tasks.append({
            "name": t["name"],
            "start": t["start"],
            "end": t["end"],
            "station": t["station"],
            "score": t["score"]
        })
    tasks.sort(key=lambda x: x["end"])  # sort by end time

    # Parse subway
    stations = set()
    for t in tasks:
        stations.add(t["station"])
    stations.add(start_station)
    max_station = max(stations) + 1
    edges = []
    for e in subway_raw:
        u, v = e["connection"]
        w = e["fee"]
        edges.append((u, v, w))

    # Compute all-pairs shortest paths
    dist = floyd_warshall(max_station, edges)

    # Weighted interval scheduling DP
    n = len(tasks)
    dp = [0] * (n + 1)
    prev = [-1] * n  # best previous task index
    # also track (score, -fee, schedule) to break ties
    opt = [(0, 0, [])]  # dp[0] = (score, -fee, [])

    for i in range(1, n + 1):
        task = tasks[i - 1]
        # find best j < i such that task j ends <= task i starts
        j = i - 1
        while j > 0 and tasks[j - 1]["end"] > task["start"]:
            j -= 1

        # option1 = skip task i
        skip = opt[i - 1]

        # option2 = take task i
        take_score = task["score"] + opt[j][0]
        # compute fee for full path
        schedule = opt[j][2] + [task]
        fee = dist[start_station][schedule[0]["station"]]
        for k in range(len(schedule) - 1):
            fee += dist[schedule[k]["station"]][schedule[k + 1]["station"]]
        fee += dist[schedule[-1]["station"]][start_station]

        take = (take_score, -fee, schedule)

        # pick best
        if take_score > skip[0]:
            opt.append(take)
        elif take_score < skip[0]:
            opt.append(skip)
        else:
            # same score, pick smaller fee
            if take[1] > skip[1]:
                opt.append(take)
            else:
                opt.append(skip)

    best_score, neg_fee, best_schedule = opt[-1]
    best_fee = -neg_fee
    best_names = [t["name"] for t in sorted(best_schedule, key=lambda x: x["start"])]

    return jsonify({
        "max_score": int(best_score),
        "min_fee": int(best_fee),
        "schedule": best_names
    })
