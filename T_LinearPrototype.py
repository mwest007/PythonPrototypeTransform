from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
import numpy as np
import copy


class Transform(ABC):
    def __init__(self, name, parameters, reverse_flag):
 

#    def __init__(, , input_coord, input_unit, output_coord, output_unit, 
#                 , , input_dim = None, output_dim = None):
        self.name = name
        self.parameters = parameters
        self.reverse_flag = reverse_flag

    def inverse(self):
        pass

    @abstractmethod
    def apply(self, data, backward = 0, padded_matrix=0):
        return copy.deepcopy(data)

#---The invert 
    @abstractmethod
    def invert(self, data, padded_matrix=0):
        return copy.deepcopy(data)

#    @abstractmethod
#    def inputDataShape( self, data, backward = 0 ):
#        pass




class t_linear(Transform):

    """Linear (affine) transformations with optional offset

    t_linear implements simple matrix multiplication with offset,
    also known as the affine transformations.

    You specify the linear transformation with pre-offset, a mixing
    matrix, and a post-offset.  That overspecifies the transformation, so
    you can choose your favorite method to specify the transform you want.
    The inverse transform is automagically generated, provided that it
    actually exists (the transform matrix is invertible).  

    Extra dimensions in the input vector are ignored, so if you pass a
    3xN vector into a 3-D linear transformation, the final dimension is passed
    through unchanged.

    The parameters are passed in through a dictionary with keyword and value


    Parameters
    ----------
    s, scale, Scale : np.array, python array, list or tuple 

        A scaling scalar, 1D vector, or matrix.  If you specify a vector
        it is treated as a diagonal matrix (for convenience), the vector length 
        must be less than or equalt to the dimension of the transform dimension, 
        if less than 
        the transform dimension only dimesnions up to the vector size will be 
        acted on.  The Scale gets left-multiplied with the transformation matrix 
        you specify (or the identity), so that if you specify both a scale and a 
        matrix the scaling is perfomred after processes such as rotation or 
        skewing. A scale can not be applied together with an additional 
        transformation matrix.


    r, rot, rota, rotation, Rotation : scalar, np.array, python array, list, 
        tuple 

        A rotation angle in degrees -- useful for 2-D and 3-D data only.  If
        you pass in a scalar, it specifies a rotation from the 0th axis toward
        the 1st axis.  If you pass in a 3-vector as either a PDL or an array
        ref (as in "rot=>[3,4,5]"), then it is treated as a set of Euler
        angles in three dimensions, and a rotation matrix is generated that
        does the following, in order:
        
        Rotate by rot->(2) degrees from 0th to 1st axis
        Rotate by rot->(1) degrees from the 2nd to the 0th axis
        Rotate by rot->(0) degrees from the 1st to the 2nd axis

        The rotation matrix is left-multiplied with the transformation matrix
        you specify, so that if you specify both rotation and a general matrix
        the rotation happens after the more general operation.

        Of course, you can duplicate this functionality -- and get more
        general -- by generating your own rotation matrix and feeding it in
        with the 'matrix' option.

    m, matrix, Matrix: np.array, python array, list, or tuple 

        The transformation matrix.  It needs to be square.  If it is invertible 
        (note: must be square for that), then you automatically get an inverse 
        transform too.

    pre, preoffset, offset, Offset: np.array, python array, list, or tuple  

        The vector to be added to the (mxn) data array before being multiplied 
        by any input matrix (equivalent of CRVAL in FITS, if you are converting 
        from scientific to pixel units). pre will accept: a single element vector 
        which is broadcast over the whole input data array, a python array or 
        numpy nx1 column array vector, where the values are broadcast over 
        corresponding rows, or a corresponding vector array of similar (mxn)
        dimensions to the input array.

    post, postoffset, shift, Shift: np.array, python array, list, or tuple 

        The vector to be added to the (mxn) data array after being multiplied 
        by any input matrix (equivalent of CRPIX-1 in FITS, if you are 
        converting from scientific to pixel units). pre will accept: a single 
        element vector which is broadcast over the whole input data array, a 
        python array or numpy n'x1 column array vector, where the values are 
        broadcast over corresponding rows, or a corresponding vector array of 
        similar (m'xn') dimensions to the output array.

    d, dim, dims, Dims: np.array, python array, list, or tuple 

        Most of the time it is obvious how many dimensions you want to deal 
        with: if you supply a matrix, it defines the transformation; if you 
        input offset vectors in the 'pre' and 'post' options, those define the 
        number of dimensions.  But if you only supply scalars, there is no way 
        to tell, and the default number of dimensions is 2. A matrix of the same
        size as the input array will be output.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.
    type_without_description

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation.
    common_parameters_listed_above : type
        Explanation.

    Raises
    ------


    See Also
    --------


    Notes
    -----

    Need to add different variable names

    References
    ----------


    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print([x + 3 for x in a])
    [4, 5, 6]
    >>> print("a n b")
    a
    b
    """

    def __init__(self, parameters, reverse_flag, name = 't_linear'):
        super().__init__(name, parameters, reverse_flag)

#===This section adds key parameters if not present and makes them 'None'. It
#   also parses python arrays into numpy arrays if required. It checks for 
#   inputs which will create errors

        if ( (self.parameters['matrix'] is not None) and \
           (self.parameters['scale'] is not None) ):

            raise ValueError("Linear transform: a matrix and a scale were supplied, the transform cannot handle both. Pass separately")

        if ( (self.parameters['matrix'] is not None) and \
           (self.parameters['rot'] is not None) ):

            raise ValueError("Linear transform: a matrix and a rotation parameter were supplied, the transform cannot handle both. Pass separately")

        if not 'matrix' in self.parameters:
            self.parameters['matrix'] = None
        if (self.parameters['matrix'] is not None): 
            if type(self.parameters['matrix']) is not np.ndarray:
                self.parameters['matrix'] = np.array(self.parameters['matrix'])

        if not 'rot' in self.parameters:
            self.parameters['rot'] = None
        if (self.parameters['rot'] is not None): 
            if type(self.parameters['rot']) is not np.ndarray:
                self.parameters['rot'] = np.array(self.parameters['rot'])

        if not 'scale' in self.parameters:
            self.parameters['scale'] = None
        if (self.parameters['scale'] is not None): 
            if type(self.parameters['scale']) is not np.ndarray:
                self.parameters['scale'] = \
                np.atleast_1d(np.array(self.parameters['scale']))

        if not 'pre' in self.parameters:
            self.parameters['pre'] = None
        if (self.parameters['pre'] is not None): 
            if type(self.parameters['pre']) is not np.ndarray:
                self.parameters['pre'] = \
                np.atleast_1d(np.array(self.parameters['pre']))

        if not 'post' in self.parameters:
            self.parameters['post'] = None
        if (self.parameters['post'] is not None): 
            if type(self.parameters['post']) is not np.ndarray:
                self.parameters['post'] = \
                np.atleast_1d(np.array(self.parameters['post']))

        if not 'dims' in self.parameters:
            self.parameters['dims'] = None


#===This section uses the input parameters to determine the out put dimensions
        if self.parameters['matrix'] is not None:
            self.input_dim = self.parameters['matrix'].shape[0]
            self.output_dim = self.parameters['matrix'].shape[1]
            if (self.parameters['matrix'].shape[1] != \
                self.parameters['matrix'].shape[0]):
                
                raise ValueError("transform: t_linear: matrix: expects a square input matrix. Stopping here")
        else:
            if self.parameters['rot'] is not None and \
            type( self.parameters['rot'] ) is np.ndarray:

                if self.parameters['rot'].size == 1:
                    self.input_dim = 2
                    self.output_dim = 2
                elif self.parameters['rot'].size == 3:
                    self.input_dim = 3
                    self.output_dim = 3
                else:
                    raise ValueError("transform: t_linear: rot: expected single angular rotation or three Eular angles. Stopping here")

            elif self.parameters['scale'] is not None and \
                 type(self.parameters['scale']) is np.ndarray or \
                 type(self.parameters['scale']) is list or \
                 type(self.parameters['scale']) is tuple :
                self.input_dim = self.parameters['scale'].shape[0]
                self.output_dim = self.parameters['scale'].shape[0]

            elif self.parameters['pre'] is not None and \
                 type(self.parameters['pre']) is np.ndarray:
                self.input_dim = self.parameters['pre'].shape[0]
                self.output_dim = self.parameters['pre'].shape[0]

            elif self.parameters['post'] is not None and \
                 type(self.parameters['post']) is np.ndarray:
                self.input_dim = self.parameters['post'].shape[0]
                self.output_dim = self.parameters['post'].shape[0]

            elif self.parameters['dims'] is not None:
                    self.input_dim = self.parameters['dims']
                    self.output_dim = self.parameters['dims']

            else:
                print("transform: t_linear: dims: Insufficient dimensions specified, assuming 2-D transform")
                self.input_dim = 2
                self.output_dim = 2

#---An identity matrix is built based on the input and output dimensions.
            self.parameters['matrix'] = \
              np.zeros((self.input_dim, self.output_dim))

            np.fill_diagonal(self.parameters['matrix'], 1)

#===If rotation matrix is specified apply rotation and multiply by identify 
#   matrix 
        if self.parameters['rot'] is not None :

            if self.parameters['rot'].size == 3:
                rotationOutput = Rotation.from_euler('xyz', \
                    [ self.parameters['rot'][0] , \
                      self.parameters['rot'][1] , \
                      self.parameters['rot'][2] ], degrees=True)

                rot_matrix = rotationOutput.as_matrix()                

            if self.parameters['rot'].size == 1:
                theta = np.deg2rad(self.parameters['rot'])
                rot_matrix = np.array(( (np.cos(theta), -np.sin(theta)) , \
                                        (np.sin(theta),  np.cos(theta)) ))

            self.parameters['matrix'] = \
              np.matmul(self.parameters['matrix'], rot_matrix)


#===Applies a scale, if scale is not an array, scale is treated as a scalar and 
#   values multiplied as such

        if (self.parameters['scale'] is not None):
            if self.parameters['scale'].ndim > 1:
                raise ValueError("transform: t_linear: scale: Scale only accepts scalars and 1D arrays")
            elif self.parameters['scale'].size == 1:
                for j in range(self.parameters['matrix'].shape[0]):
                    self.parameters['matrix'][j][j] *= self.parameters['scale']
            else:
                for j in range(self.parameters['matrix'].shape[0]):
                    self.parameters['matrix'][j] *= \
                    self.parameters['scale'][j]

#---need to check if array is invertable and set flag. Compute the 
#   (multiplicative) inverse of the matrix (np.linalg.inv), if not, flag.
        self._non_invertible = 0
        try:
            self.inverseMatrix = np.linalg.inv(self.parameters['matrix'])
        except np.linalg.LinAlgError:
            self.inverseMatrix = None
            self._non_invertible = 1

#===Calculate the inverse transform if possible
    def invert(self, data, padded_matrix=0):
        """apply the inverse transform

        If invertable creates the inverse transform for given parameters to the 
        input array (data). The transforms are applied in this order: 
    
        outdata = (data - post) * inverseMatrixTransform - pre

        Parameters
        ----------

        data : np.array
  
            array to be transformed

        padded_matrix : scalar {1|0}

            a scalar indicating if padding is required around the array output. 
            This pads the output array to match the input array with values from 
            the input array. This is useful to return inverted transforms with 
            the initial input values and dimensions. Set to default as default. 
        """
        if not self._non_invertible:

#---Create copy of input data to avoid modifying input matrix
            data = copy.copy(data)

#---Test to see if the array dimensions are sufficient for the proposed 
#   transform
            matrixDimension = self.parameters['matrix'].shape[0]
            if matrixDimension > np.shape(data)[-1]:
                raise ValueError(f"Transform: T_linear: apply: transform requires at least a {np.shape(data)[-1]}  dimension columns array ")

#---Create a deep copy of the data and remove any post transformation is specified
            if self.parameters['post'] is not None:
                dataWithoutPostTransform = \
                  copy.deepcopy(data[..., 0:matrixDimension]) - \
                  self.parameters['post']
            
            else:
                dataWithoutPostTransform = \
                  copy.deepcopy(data[..., 0:matrixDimension])

#---Perform inverse matrix multipliction and remove any pre transform offset if 
#   specified.
            if self.parameters['pre'] is not None:
                dataWithoutPreTransform = \
                  np.matmul(dataWithoutPostTransform, self.inverseMatrix) - \
                  self.parameters['pre']
            
            else:
                dataWithoutPreTransform = \
                np.matmul(dataWithoutPostTransform, self.inverseMatrix)

            outdata = copy.deepcopy(dataWithoutPreTransform)

            if padded_matrix :
                paddeddata = data
                paddeddata[0:outdata.shape[0], 0:outdata.shape[1]] = outdata
                outdata = paddeddata

            return outdata

        else:
            print("trying to invert a non-invertible matrix.")

#===Apply the transforms. These are applied in this order outdata = post + 
#   (matrixTransform)*(data + pre)
    def apply(self, data, backward = 0, padded_matrix=0):

        """apply the transform for given parameters to the input array (data). 
        The transforms are applied in this order:
    
        outdata = (data + pre) * transform + post

        Parameters
        ----------

        data : np.array
  
            array to be transformed

        padded_matrix : scalar {1|0}

            a scalar indicating if padding is required around the array output. 
            This pads the output array to match the input array with values from 
            the input array. This is useful to return inverted transforms with 
            the initial input values and dimensions. Set to default as default. 
        """


#---Create copy of input data to avoid modifying input matrix
        data = copy.copy(data)


#---Test for reversible flags
        if (not backward and not self.reverse_flag) or \
           (backward and self.reverse_flag):


#---Test to see if the array dimensions are sufficient for the proposed transform
            matrixDimension = self.parameters['matrix'].shape[0]

            if matrixDimension > np.shape(data)[-1]:
                raise ValueError(f"Transform: T_linear: apply: transform requires at least a {np.shape(data)[-1]}  dimension columns array ")

#---Create a deep copy of the data array and add the pre transform offset if 
#   specified. Read out the resulting array
            if self.parameters['pre'] is not None:
                dataWithPreTransform = \
                  copy.deepcopy(data[..., 0:matrixDimension]) + \
                  self.parameters['pre']
            
            else:
                dataWithPreTransform = \
                  copy.deepcopy(data[..., 0:matrixDimension])

            outdata = copy.deepcopy(dataWithPreTransform)

#---Perform matrix multipliction and add post transform offset if specified.
            if self.parameters['post'] is not None:
                outdata[..., 0:matrixDimension] = \
                np.matmul(dataWithPreTransform, self.parameters['matrix']) + \
                self.parameters['post']
            else:
                outdata[..., 0:matrixDimension] = \
                np.matmul(dataWithPreTransform, self.parameters['matrix'])

            if padded_matrix :
                paddeddata = data
                paddeddata[0:outdata.shape[0], 0:outdata.shape[1]] = outdata
                outdata = paddeddata

            return outdata

    def __str__(self):
        outString =  f"Transform name: {self.name}\n"\
                     f"Input parameters: {self.parameters}\n"\
                     f"Non-Invertible: {self._non_invertible}\n"\
                     f"Inverse Matrix: {self.inverseMatrix}\n"
        return outString 

