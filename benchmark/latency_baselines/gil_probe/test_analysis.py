# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
import unittest
import numpy as np
from analyze import requests, requests_reference, hold_intervals, paired_duration, pending_acquirers


def trace(codes, request=1):
    return np.asarray([[100+i*10,request,code,0] for i,code in enumerate(codes)],dtype=np.uint64)


class AnalysisTests(unittest.TestCase):
    def test_split_and_request_pairing(self):
        events=trace([13,7,8,9,1,2,3,4,6,12,14])
        samples=np.asarray([[100,1,2,3,4,5,6]],dtype=np.uint64)
        out,excluded=requests(events,samples)
        self.assertEqual(excluded,0)
        self.assertEqual(out.shape,(1,18))
        self.assertEqual(list(out[0,7:11]),[10,10,30,50])
        self.assertEqual(int(out[0,14]),1)
        self.assertEqual(int(out[0,15]),0)
        np.testing.assert_array_equal(out,requests_reference(events,samples)[0])

    def test_multiple_acquisitions_excluded_not_misattributed(self):
        events=trace([13,7,8,9,1,2,6,12,7,8,9,1,2,6,12,14])
        out,excluded=requests(events,np.zeros((1,7),dtype=np.uint64))
        self.assertEqual(excluded,1)
        self.assertEqual(out.shape,(0,18))

    def test_invalid_pairing_rejected(self):
        with self.assertRaises(ValueError): paired_duration(trace([3]),3,4)
        with self.assertRaises(ValueError): paired_duration(trace([4]),3,4)
        with self.assertRaises(ValueError): requests(trace([13,14]),np.zeros((2,7),dtype=np.uint64))
        events=trace([13,14]);events[1,0]=0
        with self.assertRaises(ValueError): requests(events,np.zeros((1,7),dtype=np.uint64))

    def test_ownership_excludes_partial_edges(self):
        events=trace([8,1,2,6,12,14,13,7,8,1,2,6])
        self.assertEqual(hold_intervals(events),[(130,180,1)])

    def test_same_timestamp_acquisitions_are_not_negative(self):
        rows=np.asarray([[10,10],[10,20],[15,15],[20,30],[21,22]],dtype=np.uint64)
        np.testing.assert_array_equal(pending_acquirers(rows),[0,0,1,0,1])

    def test_vectorized_reference_agree_for_mixed_requests(self):
        events=np.concatenate([trace([13,7,8,9,1,2,3,4,3,4,6,12,14]),
                               trace([13,7,8,9,1,2,6,12,14],2)+[200,0,0,0]])
        samples=np.zeros((2,7),dtype=np.uint64)
        a,excluded=requests(events,samples)
        b,reference_excluded=requests_reference(events,samples)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(excluded,reference_excluded)


if __name__=='__main__': unittest.main()
