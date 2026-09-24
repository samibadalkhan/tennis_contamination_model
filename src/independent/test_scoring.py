"""Semantic checks for the independent scoring implementation."""
import unittest
import numpy as np
from src import forecast
from src.independent.scoring import iid,iid_vector,play,posterior_burst,posterior_iid,posterior_iid_vectorized


class ScoringTests(unittest.TestCase):
    def test_iid_against_existing_recursion(self):
        for a,b in [(0.5,0.5),(0.68,0.60),(0.72,0.58),(0.58,0.72),(0.9,0.9)]:
            for bo in [3,5]:
                self.assertAlmostEqual(iid(a,b,bo),forecast.p_match(a,b,bo),places=8)

    def test_all_rules_symmetry(self):
        for at,to in [(0,7),(6,7),(6,10),(12,7)]:
            self.assertAlmostEqual(iid(.68,.60,5,at,to)+iid(.60,.68,5,at,to),1.,places=10)

    def test_vectorized_exact(self):
        a=np.array([.5,.68,.72,.58,.9]);b=np.array([.5,.60,.58,.72,.9])
        for bo in [3,5]:
            for at,to in [(0,7),(6,7),(6,10),(12,7)]:
                expected=np.array([iid(x,y,bo,at,to) for x,y in zip(a,b)])
                np.testing.assert_allclose(iid_vector(a,b,bo,at,to),expected,rtol=1e-12,atol=1e-12)

    def test_batched_posterior_exact(self):
        pa=np.array([[.68,.67,.66],[.72,.71,.70],[.6,.61,.62]])
        pb=np.array([[.60,.61,.62],[.58,.59,.60],[.66,.65,.64]])
        bo=np.array([3,5,3]);at=np.array([6,0,6]);to=np.array([7,7,10])
        expected=np.array([np.mean([iid(x,y,b,a,t) for x,y in zip(pa[i],pb[i])])
                           for i,(b,a,t) in enumerate(zip(bo,at,to))])
        np.testing.assert_allclose(posterior_iid_vectorized(pa,pb,bo,at,to,batch_size=2),expected,
                                   rtol=1e-12,atol=1e-12)

    def test_non_iid_zero_severity_limit(self):
        a=np.array([[.68]]);b=np.array([[.60]])
        for at,to in [(0,7),(6,7),(6,10),(12,7)]:
            sim=posterior_burst(a,b,np.array([5]),np.array([at]),np.array([to]),40000,41,0.)[0]
            self.assertLess(abs(sim-iid(.68,.60,5,at,to)),.012)

    def test_counts_and_equal_impairment(self):
        for i in range(100):
            win,k1,n1,k2,n2,points=play(.64,.64,3,6,7,0,100000,2.,0,100000,2.)
            self.assertEqual(n1+n2,points)
            self.assertTrue(0<=k1<=n1 and 0<=k2<=n2)
            self.assertIn(win,[0,1])


if __name__=='__main__':unittest.main()
