import json
import logging

from flask import request

from routes import app

logger = logging.getLogger(__name__)


def merge_intervals(intervals):
    """
    Merge overlapping or contiguous intervals.
    Intervals that touch (e.g., [5,8] and [8,10]) are treated as a single busy period.
    """
    if not intervals:
        return []
    # Sort intervals by start time
    intervals_sorted = sorted(intervals, key=lambda iv: iv[0])
    merged = []
    for start, end in intervals_sorted:
        if not merged:
            merged.append([start, end])
            continue
        last_start, last_end = merged[-1]
        # If current interval overlaps or touches the previous, merge them
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged


def minimum_boats_required(intervals):
    """
    Compute the maximum number of simultaneous bookings.
    Treat end times as exclusive so [5,8] and [8,10] do not overlap.
    """
    if not intervals:
        return 0
    events = []
    for start, end in intervals:
        events.append((start, 1))   # start of booking
        events.append((end, -1))    # end of booking
    # Sort events; process end (-1) before start (+1) at the same time
    events.sort(key=lambda ev: (ev[0], ev[1]))
    max_overlap = 0
    current_overlap = 0
    for _, delta in events:
        current_overlap += delta
        if current_overlap > max_overlap:
            max_overlap = current_overlap
    return max_overlap


@app.route('/sailing-club', methods=['POST'])
def evaluate2():
    """
    Handle POST requests containing test cases for the sailing club.
    Each test case should have an 'id' and an 'input' list of [start, end] bookings.
    Returns merged busy periods and the minimum number of boats needed.
    """
    data = request.get_json(force=True, silent=True) or {}
    logging.info("data sent for evaluation {}".format(data))

    test_cases = data.get("testCases", [])
    solutions = []

    for case in test_cases:
        case_id = case.get("id")
        intervals = case.get("input", [])

        # Validate and normalize intervals
        valid_intervals = []
        for iv in intervals:
            try:
                start, end = iv
                start_int = int(start)
                end_int = int(end)
                if start_int > end_int:
                    start_int, end_int = end_int, start_int
                valid_intervals.append([start_int, end_int])
            except Exception:
                logger.warning("Skipping invalid interval %s in test case %s", iv, case_id)
                continue

        # Merge intervals for part 1
        merged_slots = merge_intervals(valid_intervals)
        # Calculate minimum boats for part 2
        min_boats = minimum_boats_required(valid_intervals)

        solutions.append({
            "id": case_id,
            "sortedMergedSlots": merged_slots,
            "minBoatsNeeded": min_boats
        })

    result = {"solutions": solutions}
    logging.info("My result :{}".format(result))
    return json.dumps(result)
