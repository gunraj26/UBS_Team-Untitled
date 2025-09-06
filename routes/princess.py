import logging
from flask import request, jsonify
from routes import app
import heapq
import bisect

logger = logging.getLogger(__name__)


def dijkstra(n, graph, src):
    """Shortest path distances from src using Dijkstra."""
    INF = 10**12
    dist = [INF] * n
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

    if not tasks_raw:
        return jsonify({"max_score": 0, "min_fee": 0, "schedule": []})

    # Build graph (ensure it covers all stations in subway + tasks + start)
    stations = set([start_station] + [t["station"] for t in tasks_raw])
    for e in subway_raw:
        stations.update(e["connection"])
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
    relevant_stations = set([start_station] + [t["station"] for t in tasks])
    dist_map = {s: dijkstra(max_station, graph, s) for s in relevant_stations}

    # DP: dp[i] = best up to task i
    # store (score, fee, schedule)
    dp = [(0, 0, [])]

    for i, task in enumerate(tasks, start=1):
        # find latest compatible task j < i
        j = bisect.bisect_right(end_times, task["start"])

        # Option 1: skip
        skip = dp[i - 1]

        # Option 2: take
        prev_score, prev_fee, prev_sched = dp[j]
        new_score = prev_score + task["score"]

        if not prev_sched:
            # first task: start -> task -> start
            travel_fee = dist_map[start_station][task["station"]] + dist_map[task["station"]][start_station]
        else:
            last_station = prev_sched[-1]["station"]
            travel_fee = prev_fee - dist_map[last_station][start_station]  # remove return-to-start
            travel_fee += dist_map[last_station][task["station"]]         # add transition
            travel_fee += dist_map[task["station"]][start_station]        # add new return-to-start

        take = (new_score, travel_fee, prev_sched + [task])

        # Choose better
        if take[0] > skip[0]:
            dp.append(take)
        elif take[0] < skip[0]:
            dp.append(skip)
        else:
            # same score → pick smaller fee
            if take[1] < skip[1]:
                dp.append(take)
            else:
                dp.append(skip)

    best_score, best_fee, best_schedule = dp[-1]
    best_names = [t["name"] for t in sorted(best_schedule, key=lambda x: x["start"])]

    result = {
        "max_score": int(best_score),
        "min_fee": int(best_fee),
        "schedule": best_names
    }
    logger.info("📊 Princess Diaries result: %s", result)
    return jsonify(result)
