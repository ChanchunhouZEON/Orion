#!/usr/bin/env python3
# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Attribute GIL traces without treating notifications as targeted wakeups."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import numpy as np

EVENTS = {
    1:'take_enter', 2:'entry_mutex_acquired', 3:'cond_wait_begin',
    4:'cond_wait_end', 5:'drop_request', 6:'gil_acquired',
    7:'drop_enter', 8:'gil_cleared_before_notify', 9:'drop_mutex_released',
    10:'forced_handoff_begin', 11:'forced_handoff_end', 12:'take_exit',
    13:'request_begin', 14:'request_end',
}
COLUMNS = ['e2e','input','release','native','gil_return','wrap','ffi_total',
           'entry_mutex','condition_wait','other_take','take_total',
           'drop_total_before_handoff','forced_handoff','acquire_to_request_end',
           'wait_loops','timeouts','take_enter_ns','take_exit_ns']


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a): return None
    a.sort()
    return dict(mean_us=float(a.mean()/1000), **{
        f'p{p}_us':float(a[max(0,int(np.ceil(len(a)*p/100))-1)]/1000)
        for p in (50,95,99)})


def paired_duration(events, begin, end):
    pending = None
    total = count = 0
    for timestamp, _, code, _ in events:
        if code == begin:
            if pending is not None: raise ValueError('nested event pair')
            pending = int(timestamp)
        elif code == end:
            if pending is None: raise ValueError('unmatched event end')
            total += int(timestamp) - pending
            count += 1
            pending = None
    if pending is not None: raise ValueError('incomplete event pair')
    return total, count


def requests_reference(events, samples):
    """Only attribute a request with exactly one complete GIL reacquisition.

    Interpreter preemption can add more acquisitions inside one request. Keep
    those in raw data and report exclusion; never guess which was search return.
    """
    if len(events) and np.any(events[1:,0] < events[:-1,0]):
        raise ValueError('nonmonotonic trace')
    begins = np.flatnonzero(events[:,2] == 13)
    ends = np.flatnonzero(events[:,2] == 14)
    if len(begins) != len(samples) or len(ends) != len(samples):
        raise ValueError('trace/request count mismatch')
    rows = []
    excluded = 0
    for i, (begin, end) in enumerate(zip(begins, ends)):
        e = events[begin:end+1]
        if end <= begin or np.any(e[:,1] != i+1):
            raise ValueError('request identity mismatch')
        codes = e[:,2]
        single = [1,2,6,7,8,9,12,13,14]
        if any(np.sum(codes == code) != 1 for code in single):
            excluded += 1
            continue
        t = {code:int(e[codes == code,0][0]) for code in single}
        if not (t[13] <= t[7] <= t[8] <= t[9] <= t[1] <= t[2] <= t[6] <= t[12] <= t[14]):
            excluded += 1
            continue
        wait, loops = paired_duration(e,3,4)
        handoff, _ = paired_duration(e,10,11)
        total = t[12]-t[1]
        mutex = t[2]-t[1]
        other = total-mutex-wait
        if other < 0: raise ValueError('overlapping take intervals')
        timeouts = int(e[codes == 4,3].sum())
        rows.append([*map(int,samples[i]),mutex,wait,other,total,t[9]-t[7],handoff,
                     t[14]-t[6],loops,timeouts,t[1],t[12]])
    return np.asarray(rows,dtype=np.uint64).reshape(-1,len(COLUMNS)), excluded


def requests(events, samples):
    """Vectorized equivalent of the small reference parser for million-row traces."""
    n=len(samples)
    if len(events) and np.any(events[1:,0] < events[:-1,0]):
        raise ValueError('nonmonotonic trace')
    tagged=events[events[:,1] != 0]
    if len(tagged) and (tagged[:,1].max() > n or np.any(tagged[1:,1] < tagged[:-1,1])):
        raise ValueError('request identity mismatch')
    expected=np.arange(1,n+1,dtype=np.uint64)
    for code in (13,14):
        if not np.array_equal(tagged[tagged[:,2] == code,1],expected):
            raise ValueError('trace/request count mismatch')
    t={}
    valid=np.ones(n,dtype=bool)
    for code in (1,2,6,7,8,9,12,13,14):
        selected=tagged[tagged[:,2] == code]
        ids=selected[:,1].astype(np.intp)
        valid &= np.bincount(ids,minlength=n+1)[1:] == 1
        times=np.zeros(n+1,dtype=np.uint64)
        times[ids]=selected[:,0]
        t[code]=times[1:]
    sequence=(13,7,8,9,1,2,6,12,14)
    for first,last in zip(sequence,sequence[1:]): valid &= t[first] <= t[last]
    def durations(begin,end):
        a=tagged[tagged[:,2] == begin]
        b=tagged[tagged[:,2] == end]
        if not np.array_equal(a[:,1],b[:,1]) or np.any(b[:,0] < a[:,0]):
            raise ValueError('unmatched event pair')
        ids=a[:,1].astype(np.intp)
        total=np.zeros(n+1,dtype=np.uint64)
        np.add.at(total,ids,b[:,0]-a[:,0])
        return total[1:],np.bincount(ids,minlength=n+1)[1:]
    wait,loops=durations(3,4)
    handoff,_=durations(10,11)
    timeouts=np.zeros(n+1,dtype=np.uint64)
    ended=tagged[tagged[:,2] == 4]
    np.add.at(timeouts,ended[:,1].astype(np.intp),ended[:,3])
    total=t[12]-t[1]
    mutex=t[2]-t[1]
    if np.any((mutex+wait > total)&valid): raise ValueError('overlapping take intervals')
    matrix=np.column_stack((samples,mutex,wait,total-mutex-wait,total,
                            t[9]-t[7],handoff,t[14]-t[6],loops,timeouts[1:],t[1],t[12]))
    return matrix[valid].astype(np.uint64),int(np.sum(~valid))


def pending_acquirers(rows):
    # Clock resolution can give an uncontended acquisition identical start/end
    # timestamps. Such zero-width intervals must not contribute a negative count.
    # Equal-time starts are unordered: report strictly earlier, still-active
    # attempts, a conservative lower bound rather than inventing tie ordering.
    nonzero=rows[:,-1] > rows[:,-2]
    starts=np.sort(rows[nonzero,-2])
    ends=np.sort(rows[nonzero,-1])
    counts=np.searchsorted(starts,rows[:,-2],side='left')-np.searchsorted(ends,rows[:,-2],side='right')
    if np.any(counts < 0): raise ValueError('invalid acquisition intervals')
    return counts


def hold_intervals(events):
    # The first drop may belong to a pre-trace acquisition; the last acquisition
    # may not have a recorded drop. Exclude those partial intervals.
    acquisitions=np.flatnonzero(events[:,2] == 6)
    releases=np.flatnonzero(events[:,2] == 8)
    positions=np.searchsorted(releases,acquisitions,side='right')
    keep=positions < len(releases)
    acquisitions=acquisitions[keep]
    following=releases[positions[keep]]
    return list(zip(events[acquisitions,0].tolist(),events[following,0].tolist(),events[acquisitions,1].tolist()))


def analyze(folder):
    records = json.loads((folder/'results.json').read_text())
    grouped = defaultdict(list)
    controls = defaultdict(lambda:defaultdict(list))
    exclusions = defaultdict(int)
    ownership = defaultdict(list)
    all_drop_requests = defaultdict(int)
    burst_rows = defaultdict(list)
    quanta = defaultdict(set)
    for record in records:
        key = (record['dataset'],record['effort'],record['concurrency'])
        controls[key][record['mode']].append(record)
        if record['mode'] != 'traced': continue
        path = folder/record['name']
        trial = []
        for sample_path in sorted(path.glob('worker*-samples.npy')):
            samples = np.load(sample_path,allow_pickle=False)
            events = np.load(sample_path.with_name(sample_path.name.replace('-samples','-events')),allow_pickle=False)
            if len(events)>1: quanta[key].add(int(np.gcd.reduce(np.diff(events[:,0]))))
            rows, excluded = requests(events,samples)
            exclusions[key] += excluded
            grouped[key].append(rows)
            trial.append(rows)
            all_drop_requests[key] += int(np.sum(events[:,2] == 5))
            ownership[key].extend(end-start for start,end,req in hold_intervals(events) if req != 0)
        # Query-specific contention: attempts from other queries that have
        # already entered take_gil but have not exited when this attempt starts.
        # This excludes interpreter-internal reacquisitions and is a lower bound.
        rows = np.concatenate(trial)
        starts = np.sort(rows[:,-2])
        pending = pending_acquirers(rows)
        recent = np.searchsorted(starts,rows[:,-2],side='left') - np.searchsorted(starts,np.maximum(rows[:,-2],5000)-5000,side='left')
        burst_rows[key].append(np.column_stack((pending,recent,rows[:,10])))
    summary = []
    for key, chunks in sorted(grouped.items()):
        rows = np.concatenate(chunks)
        distribution = {name:stats(rows[:,i]) for i,name in enumerate(COLUMNS[:14])}
        tail_cut = np.sort(rows[:,0])[int(np.ceil(len(rows)*.99))-1]
        tail = rows[rows[:,0] >= tail_cut]
        control = {mode:dict(qps_median=statistics.median(r['qps'] for r in recs),
                            p99_us_median=statistics.median(r['latency']['p99_us'] for r in recs),
                            rounds=len(recs)) for mode,recs in controls[key].items()}
        bursts = np.concatenate(burst_rows[key])
        buckets = {}
        for name,col in [('pending_query_acquirers',0),('query_attempts_in_previous_5us',1)]:
            buckets[name] = [dict(count=value,requests=int(np.sum(bursts[:,col] == value)),
                                 take=stats(bursts[bursts[:,col] == value,2]))
                             for value in sorted(set(map(int,bursts[:,col])))]
        summary.append(dict(dataset=key[0],effort=key[1],concurrency=key[2],
                            requests=len(rows),excluded_requests=exclusions[key],
                            observed_clock_quanta_ns=sorted(quanta[key]),
                            timed_wait_timeouts=int(rows[:,15].sum()),
                            drop_requests_all_events=all_drop_requests[key],
                            waited_fraction=float(np.mean(rows[:,14] > 0)),
                            reacquire_iterations_mean=float(rows[:,14].mean()),
                            stages=distribution,
                            slowest_one_percent=dict(count=len(tail),means_us={
                                name:float(tail[:,i].mean()/1000) for i,name in enumerate(COLUMNS[:14])}),
                            ownership_after_query_acquisition=stats(ownership[key]),
                            controls=control,contention=buckets))
    (folder/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# GIL internals diagnostic','',
             'CPython 3.9.19, isolated non-PGO build. Units: µs. Same native extension and snapshots.',
             'Condition wait includes pthread mutex reacquisition and OS scheduling. It is not pure sleep time.',
             'Ownership wall time includes descheduling. Notifications do not identify a recipient.',
             'On this host, C timestamps have an observed 1 µs quantum; do not interpret sub-µs event differences individually. Means can be fractional.',
             'Only requests with one unambiguous GIL reacquisition enter attribution; raw data retains exclusions.',
             '', '| Dataset | L | C | Waited | Entry mutex mean | Condition mean | Other take mean | GIL return P99 | Native mean | Wrap mean | Timeouts |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        d=s['stages']
        lines.append(f"| {s['dataset']} | {s['effort']} | {s['concurrency']} | {s['waited_fraction']:.1%} | {d['entry_mutex']['mean_us']:.3f} | {d['condition_wait']['mean_us']:.3f} | {d['other_take']['mean_us']:.3f} | {d['gil_return']['p99_us']:.2f} | {d['native']['mean_us']:.2f} | {d['wrap']['mean_us']:.3f} | {s['timed_wait_timeouts']} |")
    lines += ['', '## Observer overhead', '', 'QPS: median of three rounds; P99: median of per-round P99, not pooled.',
              '| Dataset | L | C | Bare QPS | Marked QPS | Traced QPS | Trace vs marked |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        c=s['controls']
        if len(c) != 3: continue
        q=[c[m]['qps_median'] for m in ('bare','marked','traced')]
        lines.append(f"| {s['dataset']} | {s['effort']} | {s['concurrency']} | {q[0]:,.0f} | {q[1]:,.0f} | {q[2]:,.0f} | {q[2]/q[1]-1:+.2%} |")
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    analyze(parser.parse_args().results)
