import T_LinearPrototype
import numpy as np
import unittest

print("================================\n")
print("Running the T_linear test suite \n")
print("================================\n")



#---Test t_linear object with scale




FunctionName = "t_linear"
TestArray  = np.zeros((5,6))
TestArray  += np.arange(6)

class Test_TLinear(unittest.TestCase):


    def test_Tlinear_scale_scalar(self):
        print("\nTesting :", self, " Result: . = PASS , F = FAIL , E = Error)")
        ScalarScale = 3
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = T_LinearPrototype.t_linear( parameters = FunctionParams , reverse_flag = 0 )
        expected_output = np.array([[ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.]])
        output = test_object.apply( TestArray )
        #print(output)
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")
"""
    def test_Tlinear_scale_scalar_dimensions(self):
        print("\nTesting :", self, " Result: . = PASS , F = FAIL , E = Error)")
        ScalarScale = 3
        dimensionInput = np.array([6,5])
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': dimensionInput }
        test_object = T_LinearPrototype.t_linear( parameters = FunctionParams , reverse_flag = 0 )
        expected_output = np.array([[ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.]])
        output = test_object.apply( TestArray )
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")

    def test_Tlinear_scale_vector(self):
        print("\nTesting :", self, " Result: . = PASS , F = FAIL , E = Error)")
        VectorScale = np.empty(6)
        ScalarScale = 3
        VectorScale.fill(3)
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = T_LinearPrototype.t_linear( parameters = FunctionParams , reverse_flag = 0 )        expected_output = np.array([[ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.], [ 0., 3., 6., 9., 12., 15.]])
        output = test_object.apply( TestArray )
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")
"""

if __name__ == '__main__':
    unittest.main()
