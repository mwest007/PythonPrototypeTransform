from PythonPrototypeTransform.PrototypeTransform import TestPerson
import PythonPrototypeTransform.PrototypeTransform as PrototypeTransform
import numpy as np
import numpy.testing
import unittest
print("================================\n")
print("Running the Transform test suite\n")
print("================================\n")



print(PrototypeTransform.ndcoords([3,2]))



#---Class inputs

class test_person_test(unittest.TestCase):

    def testOfDoCatYearsAddUp(self):
        print("\nTesting :", self, " Result: . = PASS , F = FAIL , E = Error)")
        c = TestPerson("Harry", 3)
        self.assertEqual( c.myAgeCatYears(), 21, "Should be 21")



class test_ndcoords(unittest.TestCase):

    def test_ndcoords_test1(self):
        print("\nTesting :", self, " Result: . = PASS , F = FAIL , E = Error)")
        Test1Array = [4]
        test1_output = PrototypeTransform.ndcoords( Test1Array )
        expected_test1_result = np.array([[0.], [1.], [2.], [3.]])

        Test2Array = [3,2]
        test2_output = PrototypeTransform.ndcoords( Test2Array )
        expected_test2_result = np.array([[[0., 0.], [1., 0.], [2., 0.]], [[0., 1.], [1., 1.], [2., 1.]]])
        numpy.testing.assert_array_equal( test2_output, expected_test2_result, "[[0]]")





if __name__ == '__main__':
    unittest.main()


