import T_LinearPrototype as transform
import numpy as np
import unittest
from astropy.io import fits

print("================================\n")
print("Running the t_linear test suite \n")
print("================================\n")
print()
print(" Result: . = PASS , F = FAIL , E = Error")

#===Function Name
FunctionName = "t_linear"

#===TestArrays

TestArray_1x3NumpyArray = np.arange(3.0).reshape(1, 3)
TestArray_1x4NumpyArray = np.arange(4.0).reshape(1, 4)
TestArray_1x5NumpyArray = np.arange(5.0).reshape(1, 5)

TestArray_3x1PythonArray = (2,3,4)
TestArray_3x1NumpyArray = np.arange(3.0).reshape(3, 1)
TestArray_3x4NumpyArray = np.arange(12).reshape(3, 4)

TestArray_4x1PythonArray = (6,2,3,4)
TestArray_4x1NumpyArray = np.arange(4.0).reshape(4, 1)
TestArray_4x3NumpyArray = np.arange(12).reshape(4, 3)

TestArray_5x1NumpyArray = np.arange(5.0).reshape(5, 1)
TestArray_5x1PythonArray = (6,2,3,4,4)
TestArray_5x5NumpyArray = np.arange(25).reshape(5, 5)

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

    def test_t_linear_scale_vector_pythonArray_padded(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = TestArray_5x1PythonArray
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )

        #---Outputs
        output = test_object.apply(TestArray , padded_matrix=1)
        expected_output = np.array([[0.,   2.,  6., 12., 20.], \
                                    [5.,  12., 21., 32., 45.], \
                                    [10., 22., 36., 52., 70.], \
                                    [15., 32., 51., 72., 95.], \
                                    [20., 42., 66., 92., 120.]])
        
        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None") 





class Blagtest_t_linear(unittest.TestCase):

    def test_t_linear_scale_scalar(self):
        print("\nTesting :", self)

        #---Inputs
        TestArray = TestArray_5x5NumpyArray
        ScalarScale = 3
        FunctionParams = {'matrix': None, 'rot': None, 'scale': ScalarScale, 'pre': None, 'post': None, 'dims': None}
        test_object = transform.t_linear( parameters = FunctionParams , reverse_flag = 0 )
        
        #---Outputs
        output = test_object.apply( TestArray )
        print(output)
        expected_output = np.array([[0., 3., 6., 9., 12.], \
                                    [0., 3., 6., 9., 12.], \
                                    [0., 3., 6., 9., 12.], \
                                    [0., 3., 6., 9., 12.], \
                                    [0., 3., 6., 9., 12.]])
        
        expected_output = np.array([[0.,   1.,  2.,  3.,  4.], \
                                    [5.,  6.,  7.,  8.,  9.], \
                                    [10., 11., 12., 13., 14.], \
                                    [15., 16., 17., 18., 19.], \
                                    [20., 21., 22., 23., 24.]])

        #---Tests
        numpytest = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( numpytest, None, "Should be None")   


class ExampleTest(unittest.TestCase):
    def GENERIC_TEST(self):
        print("\nTesting :", self, " Result: . = PASS , F = FAIL , E = Error)")
        #---Inputs
        FunctionParams = {'PARAM': None}
        test_object = transform.TRANSFORM( parameters = FunctionParams )
        TestArray = None

        #---Outputs
        output = test_object.apply( TestArray )
        expected_output = None
        
        #---Tests
        #output = np.testing.assert_array_equal( output , expected_output )
        self.assertEqual( output, expected_output, "Should be None")

class TestClassA(unittest.TestCase):
    def testOne(self):
        # test code
        pass
    def testTwo(self):
        # test code
        pass


def run_specific_tests():
    # Run only the tests in the specified classes

    test_classes_to_run = [test_t_linear, TestClassA]

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
