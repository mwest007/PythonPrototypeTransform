import T_LinearPrototype as transform
import numpy as np
import unittest
from astropy.io import fits

print("========================================\n")
print("Running the t_linear test suite \n")
print("========================================\n")
print("Result: . = PASS , F = FAIL , E = Error")
print("========================================\n")

#===Function Name
FunctionName = "t_linear"

#===TestArrays

TestArray_1x3NumpyArray = np.arange(3.0).reshape(1, 3)
TestArray_1x4NumpyArray = np.arange(4.0).reshape(1, 4)
TestArray_1x5NumpyArray = np.arange(5.0).reshape(1, 5)

TestArray_3x1PythonArray = (2.,3.,4.)
TestArray_3x1NumpyArray = np.arange(3.0).reshape(3, 1)
TestArray_3x3NumpyArray = np.arange(9.0).reshape(3, 3)
TestArray_3x4NumpyArray = np.arange(12.).reshape(3, 4)

TestArray_4x1PythonArray = (6.,2.,3.,4.)
TestArray_4x1NumpyArray = np.arange(4.0).reshape(4, 1)
TestArray_4x3NumpyArray = np.arange(12.).reshape(4, 3)
TestArray_5x5NumpyArray = np.arange(16.).reshape(4, 4)

TestArray_5x1NumpyArray = np.arange(5.0).reshape(5, 1)
TestArray_5x1PythonArray = (6.,2.,3.,4.,4.)
TestArray_5x5NumpyArray = np.arange(25.).reshape(5, 5)

TestArray_fits = fits.open('test.fits') #Needs test SWAP fits file
TestArray_fits = TestArray_fits[0].data


class test_t_linear(unittest.TestCase):

    def test_t_linear_scale_scalar(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = 3
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )
        
        #---Outputs
        output = test_object.apply(TestArray)
        expected_output = np.array([[0. ], \
                                    [15.], \
                                    [30.], \
                                    [45.], \
                                    [60.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal(output , expected_output)
        self.assertEqual(numpytest, None, "Should be None")        


    def test_t_linear_scale_scalar_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = 3
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[0.,   1.,  2.,  3.,  4.], \
                                    [15.,  6.,  7.,  8.,  9.], \
                                    [30., 11., 12., 13., 14.], \
                                    [45., 16., 17., 18., 19.], \
                                    [60., 21., 22., 23., 24.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")     


    def test_t_linear_scale_5x1pythonArray_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = TestArray_5x1PythonArray
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[0.,   2.,  6.,  12., 16.], \
                                    [30.,  12., 21., 32., 36.], \
                                    [60.,  22., 36., 52., 56.], \
                                    [90.,  32., 51., 72., 76.], \
                                    [120., 42., 66., 92., 96.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None") 

    def test_t_linear_scale_1x5NumpyArray_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale=TestArray_1x5NumpyArray
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[0.,  1.,  4.,  9., 16.], \
                                    [0.,  6., 14., 24., 36.], \
                                    [0., 11., 24., 39., 56.], \
                                    [0., 16., 34., 54., 76.], \
                                    [0., 21., 44., 69., 96.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None") 


    def test_t_linear_scale_3x1NumpyArray_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = TestArray_3x1NumpyArray
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[0.,  1.,  4.,   3.,  4.], \
                                    [0.,  6., 14.,   8.,  9.], \
                                    [0., 11., 24.,  13., 14.], \
                                    [0., 16., 34.,  18., 19.], \
                                    [0., 21., 44.,  23., 24.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None") 

    def test_t_linear_scale_1x3NumpyArray_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = TestArray_1x3NumpyArray
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[0.,  1.,  4.,   3.,  4.], \
                                    [0.,  6., 14.,   8.,  9.], \
                                    [0., 11., 24.,  13., 14.], \
                                    [0., 16., 34.,  18., 19.], \
                                    [0., 21., 44.,  23., 24.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None") 


    def test_t_linear_no_params(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = TestArray_1x3NumpyArray
        FunctionParams = {}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[ 0.,  1.,  2.,  3.,  4.], \
                                    [ 5.,  6.,  7.,  8.,  9.], \
                                    [10., 11., 12., 13., 14.], \
                                    [15., 16., 17., 18., 19.], \
                                    [20., 21., 22., 23., 24.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None") 

    def test_t_linear_scale_rot_30_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30.
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[0.50,  0.87,  2.], \
                                    [4.59,  1.96,  5.], \
                                    [8.70,  3.06,  8.]])
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 







def run_specific_tests():
    """Tests to complete

    Run only the tests in the specified classes, add to the square brackets 
    below comma deliminated list of classes to check
    
    """

    test_classes_to_run = [test_t_linear]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)



if __name__ == '__main__':
    run_specific_tests()
    #run_all_tests()
