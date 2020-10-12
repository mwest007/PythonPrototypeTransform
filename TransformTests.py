import TransformPrototype as transform
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
        ScalarScale = np.array([3])
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}

        test_object = transform.t_linear( parameters = FunctionParams  )


        
        #---Outputs
        output = test_object.apply(TestArray, paddedmatrix=0)
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
        ScalarScale = np.array([3])
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams )

        #---Outputs
        output = test_object.apply(TestArray, paddedmatrix = 1)
        expected_output = np.array([[0.,   1.,  2.,  3.,  4.], \
                                    [15.,  6.,  7.,  8.,  9.], \
                                    [30., 11., 12., 13., 14.], \
                                    [45., 16., 17., 18., 19.], \
                                    [60., 21., 22., 23., 24.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")     

    def test_t_linear_scale_scalar_dims_2_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = np.array([3])
        dimsScale = 2
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': dimsScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray )
        expected_output = np.array([[0.,   1.,  2.,  3.,  4.], \
                                    [15.,  6.,  7.,  8.,  9.], \
                                    [30., 11., 12., 13., 14.], \
                                    [45., 16., 17., 18., 19.], \
                                    [60., 21., 22., 23., 24.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")     

    def test_t_linear_scale_scalar_dims_3_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = 3
        dimsScale = 3
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': dimsScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray)
        expected_output = np.array([[0.,   3.,  2.,  3.,  4.], \
                                    [15., 18.,  7.,  8.,  9.], \
                                    [30., 33., 12., 13., 14.], \
                                    [45., 48., 17., 18., 19.], \
                                    [60., 63., 22., 23., 24.]])
        
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
        output = test_object.apply(TestArray )
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
        output = test_object.apply(TestArray )
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
        output = test_object.apply(TestArray )
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
        output = test_object.apply(TestArray )
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
        output = test_object.apply(TestArray )
        expected_output = np.array([[ 0.,  1.,  2.,  3.,  4.], \
                                    [ 5.,  6.,  7.,  8.,  9.], \
                                    [10., 11., 12., 13., 14.], \
                                    [15., 16., 17., 18., 19.], \
                                    [20., 21., 22., 23., 24.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None") 

    def test_t_linear_rot_30_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30.
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray )
        expected_output = np.array([[0.50,  0.87,  2.], \
                                    [4.59,  1.96,  5.], \
                                    [8.70,  3.06,  8.]])
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 


    def test_t_linear_rot_30_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = (1.,2.,3.)
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray )
        expected_output = np.array([[-0.017494919,    1.0333929,     1.982875], \
                                    [   3.0287824,    3.9260869,    5.0410632], \
                                    [   6.0750597,     6.818781,    8.0992515]])


        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 



    def test_t_linear_rot_inv_rot_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30.
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray )
        output = test_object.apply(output , backward=1 )

        expected_output = TestArray
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 





    def test_t_linear_pre_post_inv_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30.
        preScale = 5.
        postScale = 10.
        FunctionParams = {'rot': rotScale, 'pre': preScale, 'post': postScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray)
        output = test_object.apply(output, backward=1)

        expected_output = TestArray
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 


    def test_t_linear_rot_30_pad_back_reverse(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 1 )

        #---Outputs
        output = test_object.apply( TestArray , backward=1)
        expected_output = np.array([[0.50,  0.87,  2.], \
                                    [4.59,  1.96,  5.], \
                                    [8.70,  3.06,  8.]])

        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 

    def test_t_linear_rot_30_pad_reverse(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 1 )

        #---Outputs
        output = test_object.apply( TestArray , backward=0)
        expected_output = np.array([[-0.5 ,  0.87,  2.  ], \
                                    [ 0.6 ,  4.96,  5.  ], \
                                    [ 1.7 ,  9.06,  8.  ]])
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 

    def test_t_linear_rot_30_pad_back(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams )

        #---Outputs
        output = test_object.apply( TestArray , backward=1)

        expected_output = np.array([[-0.5 ,  0.87,  2.  ], \
                                    [ 0.6 ,  4.96,  5.  ], \
                                    [ 1.7 ,  9.06,  8.  ]])

        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 


    def test_t_linear_rot_30_pad_forward_and_back(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams )

        #---Outputs
        output = test_object.apply( TestArray , backward = 0)
        output = test_object.apply( output , backward = 1)

        expected_output = TestArray
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 
 


    def test_ndcoords_3(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = (3)

        #---Outputs
        output = transform.ndcoords( TestArray )

        expected_output = np.array([[ 0.0 ], \
                                    [ 1.0 ], \
                                    [ 2.0 ]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")


    def test_t_linear_rot_30_padded_reverse_flag(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30.
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 1 )

        #---Outputs
        output = test_object.apply(TestArray )
        expected_output = np.array([[-0.5 ,  0.87,  2.  ],\
                                    [ 0.6 ,  4.96,  5.  ],\
                                    [ 1.7 ,  9.06,  8.  ]])
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 


    def test_t_linear_rot_30_padded_no_flag(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        rotScale = 30.
        FunctionParams = {'rot': rotScale}
        test_object = transform.t_linear( parameters = FunctionParams )

        #---Outputs
        output = test_object.apply(TestArray )
        expected_output = np.array([[0.50,  0.87,  2.], \
                                    [4.59,  1.96,  5.], \
                                    [8.70,  3.06,  8.]])
        
        #---Tests
        numpytest = np.testing.assert_almost_equal( output , expected_output, decimal=2 )
        self.assertEqual( numpytest, None, "Should be None") 


    def test_t_linear_pre_100(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        preTest = 100.
        FunctionParams = {'pre': preTest}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray )
        expected_output = np.array([[100,    101,     2], \
                                    [   103,    104,    5], \
                                    [   106,     107,    8]])


    def test_t_linear_pre_100_no_pad(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_3x3NumpyArray
        preTest = 100.
        FunctionParams = {'pre': preTest}
        test_object = transform.t_linear( parameters = FunctionParams )

        #---Outputs
        output = test_object.apply(TestArray )
        expected_output = np.array([[100,    101,     2], \
                                    [   103,    104,    5], \
                                    [   106,     107,    8]])



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
