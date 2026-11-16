# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Public native search must preserve the facade's contract without a Python frame."""
import concurrent.futures
import dataclasses
import gc
import inspect
import sys
import unittest
import numpy as np
from orion_ann import Index,SearchResult,OrionError
from orion_ann._native import _Index
import test_index as fixtures


def forwarded(self,queries,*,k=10,search_list_size=64,query_format=None,scores=False):
    return self._handle.search(queries,k,search_list_size,query_format=query_format,scores=scores,_result_type=SearchResult)

class NativeSearchEntryTests(unittest.TestCase):
    setUp=fixtures.IndexTests.setUp
    arrays=fixtures.IndexTests.arrays
    make=fixtures.IndexTests.make

    def assert_result(self,a,b):
        self.assertIs(type(b),SearchResult)
        self.assertTrue(dataclasses.is_dataclass(b))
        self.assertEqual(a.score_metric,b.score_metric)
        for key in ('ids','counts','scores','external_ids'):
            x,y=getattr(a,key),getattr(b,key)
            if x is None:self.assertIsNone(y)
            else:np.testing.assert_array_equal(x,y)

    def test_native_entry_signature_doc_and_no_forwarder_frame(self):
        base,q=self.arrays();index=self.make(base);index.calibrate(q,k=10,search_list_size=32)
        self.assertTrue(inspect.isbuiltin(index.search))
        expected=inspect.signature(forwarded)
        bound=inspect.signature(index.search);unbound=inspect.signature(Index.search)
        for actual,skip in ((bound,True),(unbound,False)):
            reference=list(expected.parameters.values())[1 if skip else 0:]
            self.assertEqual([(p.name,p.kind,p.default) for p in actual.parameters.values()],[(p.name,p.kind,p.default) for p in reference])
        self.assertIn('Search a batch',index.search.__doc__)
        self.assertIn('Search a batch',Index.search.__doc__)
        observed=[]
        def profile(frame,event,arg):
            if event=='call':observed.append((frame.f_code.co_name,frame.f_code.co_filename))
        sys.setprofile(profile)
        try:actual=index.search(q[:1])
        finally:sys.setprofile(None)
        self.assertFalse(any(name=='search' and 'orion_ann' in filename for name,filename in observed),observed)
        self.assert_result(forwarded(index,q[:1]),actual)
        self.assert_result(actual,Index.search(index,q[:1]))
        self.assert_result(actual,Index.search(index,queries=q[:1]))
        self.assert_result(actual,Index.search(self=index,queries=q[:1]))
        with self.assertRaises(TypeError):index.search(q[:1],10,64)
        with self.assertRaises(TypeError):index.search(q[:1],unknown=True)
        # Keep normal non-data-descriptor and subclass override behavior.
        class Derived(Index):
            def search(self,*args,**kwargs):return 'override'
        self.assertEqual(Derived(index._handle).search(q[:1]),'override')
        index.search=lambda *a,**k:'instance override'
        self.assertEqual(index.search(q[:1]),'instance override')
        del index.search
        self.assert_result(actual,index.search(q[:1]))

    def test_types_metrics_shapes_scores_and_errors_match_forwarder(self):
        for base_type in (np.float32,np.uint8):
            for metric in ('squared-l2','inner-product','cosine'):
                base,q=self.arrays(dtype=base_type,dimension=13)
                index=self.make(base,metric=metric,external_ids=list(range(1000,1064)))
                index.calibrate(q,k=3,search_list_size=24)
                for dtype in (np.float32,np.uint8,np.int8,np.uint16,np.uint32):
                    for n in (0,1,2,3,8):
                        values=q[:n].astype(dtype);before=values.copy()
                        for scores in (False,True):
                            self.assert_result(forwarded(index,values,k=3,search_list_size=24,scores=scores),index.search(values,k=3,search_list_size=24,scores=scores))
                        np.testing.assert_array_equal(before,values)
                for values,options in [(q[:1],dict(k=2)),(q[:1],dict(k=3,search_list_size=1)),(q[:,::2],dict(k=3)),(q.astype(np.float64),dict(k=3))]:
                    errors=[]
                    for method in (lambda *a,**kw:forwarded(index,*a,**kw),index.search):
                        with self.assertRaises(OrionError) as e:method(values,**options)
                        errors.append(e.exception.report)
                    self.assertEqual(errors[0],errors[1])

    def test_concurrent_results_and_bound_method_ownership(self):
        base,q=self.arrays(dimension=13);index=self.make(base);index.calibrate(q,k=3,search_list_size=24)
        expected=forwarded(index,q,k=3,search_list_size=24,scores=True)
        def run(i):
            out=index.search(q[i:i+1],k=3,search_list_size=24,scores=True)
            np.testing.assert_array_equal(expected.ids[i],out.ids[0])
            np.testing.assert_array_equal(expected.scores[i],out.scores[0])
            return out
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:outputs=list(pool.map(run,range(len(q))))
        bound=index.search;del index;gc.collect()
        self.assert_result(outputs[0],bound(q[:1],k=3,search_list_size=24,scores=True))
        del bound;gc.collect();self.assertEqual(outputs[0].ids.shape,(1,3))

    @unittest.skipUnless(hasattr(_Index,'_start_search_timing'),'search-timing is disabled')
    def test_timing_samples_are_thread_local_and_consistent(self):
        base,q=self.arrays();index=self.make(base);index.calibrate(q,k=3,search_list_size=24)
        def run(i):
            index._handle._start_search_timing(4)
            for j in range(4):index.search(q[(i+j)%len(q):((i+j)%len(q))+1],k=3,search_list_size=24)
            samples=index._handle._finish_search_timing();self.assertEqual(len(samples),4)
            for input_ns,release,native,reacquire,wrap,total in samples:
                self.assertGreater(native,0);self.assertGreater(wrap,0)
                self.assertLessEqual(input_ns+release+native+reacquire+wrap,total)
            self.assertEqual(index._handle._finish_search_timing(),[])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run,range(8)))

if __name__=='__main__':unittest.main()
