import logging
from flask import request, jsonify
from routes import app
import heapq
import bisect

logger = logging.getLogger(__name__)


def dijkstra(n, graph, src):
    """Shortest path distances from src using Dijkstra."""
    dist = [10**12] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


@app.route("/princess-diaries", methods=["POST"])
def princess_diaries():
    data = request.get_json(force=True)
    tasks_raw = data.get("tasks", [])
    subway_raw = data.get("subway", [])
    start_station = data.get("starting_station", 0)

    # Build graph
    stations = set([start_station] + [t["station"] for t in tasks_raw])
    max_station = max(stations) + 1
    graph = [[] for _ in range(max_station)]
    for e in subway_raw:
        u, v = e["connection"]
        w = e["fee"]
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Parse + sort tasks by end time
    tasks = []
    for i, t in enumerate(tasks_raw):
        tasks.append({
            "id": i,
            "name": t["name"],
            "start": t["start"],
            "end": t["end"],
            "station": t["station"],
            "score": t["score"]
        })
    tasks.sort(key=lambda x: x["end"])

    n = len(tasks)
    end_times = [t["end"] for t in tasks]

    # Precompute shortest paths from relevant stations
    unique_stations = list(stations)
    dist_map = {}
    for s in unique_stations:
        dist_map[s] = dijkstra(max_station, graph, s)

    # DP
    dp = [(0, 0, [])]  # (score, -fee, schedule list)

    for i, task in enumerate(tasks, start=1):
        # find latest compatible task with binary search
        j = bisect.bisect_right(end_times, task["start"])

        # Option 1: skip
        skip = dp[i - 1]

        # Option 2: take
        take_score = task["score"] + dp[j][0]
        schedule = dp[j][2] + [task]

        # compute fee
        fee = dist_map[start_station][schedule[0]["station"]]
        for k in range(len(schedule) - 1):
            fee += dist_map[schedule[k]["station"]][schedule[k + 1]["station"]]
        fee += dist_map[schedule[-1]["station"]][start_station]

        take = (take_score, -fee, schedule)

        # choose better
        if take_score > skip[0]:
            dp.append(take)
        elif take_score < skip[0]:
            dp.append(skip)
        else:
            # same score, prefer lower fee
            if take[1] > skip[1]:
                dp.append(take)
            else:
                dp.append(skip)

    best_score, neg_fee, best_schedule = dp[-1]
    best_fee = -neg_fee
    best_names = [t["name"] for t in sorted(best_schedule, key=lambda x: x["start"])]

    result = {
        "max_score": int(best_score),
        "min_fee": int(best_fee),
        "schedule": best_names
    }
    logger.info("📊 Princess Diaries result: %s", result)
    return jsonify(result)
