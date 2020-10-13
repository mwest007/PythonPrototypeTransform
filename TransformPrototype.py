from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
from scipy.interpolate import interpn
import numpy as np
import copy

__author__ = "Matthew J West, Craig DeForest, Jake-R-W"
__copyright__ = "By all the listed autors: GPLv2, There is no warranty."
__credits__ = ["Matthew J West", "Craig DeForest", "Jake-R-W"]
__license__ = "-"
__version__ = "1.0.0"
__maintainer__ = "Matthew J West"
__email__ = "mwest at swri.boulder.edu"
__status__ = "Production"


def ndcoords(*dims):
    """Returns an enumerated list of coordinates for given dimensions

    Returns an enumerated list of coordinates for given dimensions, initizilzed 
    to a tuple, adding an extra dim on the front to accommodate
    the vector coordinate index

    Parameters
    ----------
    *dims : tuple, list or numpy array dimensions of the input array

    Returns
    -------
    array(float64)

    Usage
    ----- 
    $indices = ndcoords($tuple)
    $indices = ndcoords($list)
    $indices = ndcoords(np.ndarray())

    Notes
    -----
    Enumerate pixel coordinates for an N-D variable, you feed
    in a dimension list and get out a piddle whose 0th dimension runs over
    dimension index and whose 1st through Nth dimensions are the
    dimensions given in the input.  If you feed in a piddle instead of a
    perl list.

    Examples
    --------
    >>> print(ndcoords([3,2]))
    >>> [[[0. 0.]
    >>>   [1. 0.]
    >>>   [2. 0.]]
    >>> 
    >>>  [[0. 1.]
    >>>   [1. 1.]
    >>>   [2. 1.]]]
    """

    grid_size = []
    if type(dims[0]) is tuple \
    or type(dims[0]) is list \
    or type(dims[0]) is np.ndarray:
        for i in range(len(dims[0])):
            grid_size.append(range(dims[0][i]))
    else:
        for i in range(len(dims)):
            grid_size.append(range(dims[i]))

    out = np.mgrid[grid_size]

    out = out.astype('float64').transpose()
    return out


class Transform(ABC):
    def __init__(self, name, parameters, reverse_flag):
 
#    def __init__(, , input_coord, input_unit, output_coord, output_unit, 
#                 , , input_dim = None, output_dim = None):
        self.name = name
        self.parameters = parameters
        self.reverse_flag = reverse_flag

    def apply(self, data, backward = 0, paddedmatrix = 1):
        """apply the transform for given parameters to the input array (data). 
        
        Parameters
        ----------
        data : np.array
              array to be transformed
        backward : scalar
            scalar {1|0} if set to 0 a forward transformation is applied, if set
            to 0 the inverse transformation is applied. The value is 
            automatically set to 0.
        paddedmatrix : scalar {1|0}
            A scalar indicating if padding is required around the array output. 
            This pads the output array to match the input array with values from 
            the input array. This is useful to return inverted transforms with 
            the initial input values and dimensions. Set to default as default. 

        Raises
        ------
        transform: t_linear: apply: transform requires at least a {np.shape(data)[-1]}  dimension columns array
        transform: t_linear: apply: Cannot pad data to match input dimensions
        """

        data = copy.copy(data)
        matrixDimension = self.parameters['matrix'].shape[0]
        if matrixDimension > np.shape(data)[-1]:
            raise ValueError(f"transform: t_linear: invert: transform requires at least a {np.shape(data)[-1]}  dimension columns array ")

#---Perform the transform
        outdata = self.forwardtransform(data, backward)

#---Pads the output data so it matches the input data, if padded_matrix and 
#   outdata.shape[0] < output_dim OR check if output_dim == input_dim
        if paddedmatrix :
            if outdata.ndim > 1:
                if outdata.shape[1] < data.shape[1]:
                    if outdata.shape[0] > data.shape[0]:
                        raise ValueError(f"transform: t_linear: apply: Cannot pad data to match input dimensions")
                    paddeddata = data
                    paddeddata[0:outdata.shape[0], 0:outdata.shape[1]] = outdata
                    outdata = paddeddata

        return outdata


    def map(self, data=None, template=None, opts=None):
        """return map of given data. 
        
        Parameters
        ----------
        data : np.array
              array to be mapped
        template : None
        opts : None

        Raises
        ------
        """
#---make the output dimensions the same as the data shape
        self.output_dim = data.shape

#---Create an output array of same dimensions as the input array and fill with floating point values
        output = np.empty(shape=self.output_dim, dtype=np.float64)

#---Create dimensions 
        output_dim = output.shape

#---Returns an enumerated list of coordinates for given dimensions
        ndc = ndcoords(output_dim)
        # ndc = ndc.reshape((np.product(ndc.shape[:-1]), ndc.shape[-1]))

#---Here we apply the transform 
        inverseTransform = self.apply(ndc, backward=1)
        
        pixel_grid = [np.arange(coordinates) for coordinates in data.shape]
        print(len(pixel_grid))
        print(pixel_grid)
        #print(pixel_grid[1:5,1:5])

#---Interpolated array - Multidimensional interpolation on regular grids.
        x = interpn(points=pixel_grid, values=data, method='linear', xi=inverseTransform, bounds_error=False, fill_value=0)

        """
        points - The points defining the regular grid in n dimensions.
        values - The data on the regular grid in n dimensions.
        xi - The coordinates to sample the gridded data at
        method - The method of interpolation to perform. Supported are “linear” and “nearest”, and “splinef2d”. “splinef2d” is only supported for 2-dimensional data.
        bounds_error - If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used.
        fill_value - If provided, the value to use for points outside of the interpolation domain. If None, values outside the domain are extrapolated. Extrapolation is not supported by method “splinef2d”.
        """

        output[:] = x

#---Return the transpose of the array
        return output.transpose()



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
    t_linear accepts tranform parameters as a dictionary { parameter:value} 

    parameters{s, scale, Scale : np.array, python array, list or tuple 
        A scaling scalar, 1D vector, or matrix.  If you specify a scalar a 2-D 
        output transform is assumed. If a vector is specified it is treated as a 
        diagonal matrix (for convenience), the vector length must be less than 
        or equal to the dimension of the transform dimension, if less than 
        the transform dimension only dimensions up to the vector size will be 
        acted on.  If a 1-D vector output is required, a 1-D Vector should be
        specified. The Scale gets left-multiplied with the transformation matrix 
        you specify (or the identity), so that if you specify both a scale and a 
        matrix the scaling is perfomred after processes such as rotation or 
        skewing. A scale can not be applied together with an additional 
        transformation matrix.

    parameters{r, rot, rota, rotation, Rotation : scalar, np.array, 
    python array, list, tuple}
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

    parameters{m, matrix, Matrix: np.array, python array, list, or tuple} 
        The transformation matrix.  It needs to be square.  If it is invertible 
        (note: must be square for that), then you automatically get an inverse 
        transform too.

    parameters{pre, preoffset, offset, Offset: np.array, python array, list, or 
    tuple  }
        The vector to be added to the (mxn) data array before being multiplied 
        by any input matrix (equivalent of CRVAL in FITS, if you are converting 
        from scientific to pixel units). pre will accept: a single element vector 
        which is broadcast over the whole input data array, a python array or 
        numpy nx1 column array vector, where the values are broadcast over 
        corresponding rows, or a corresponding vector array of similar (mxn)
        dimensions to the input array.

    parameters{post, postoffset, shift, Shift: np.array, python array, list, or 
    tuple}
        The vector to be added to the (mxn) data array after being multiplied 
        by any input matrix (equivalent of CRPIX-1 in FITS, if you are 
        converting from scientific to pixel units). pre will accept: a single 
        element vector which is broadcast over the whole input data array, a 
        python array or numpy n'x1 column array vector, where the values are 
        broadcast over corresponding rows, or a corresponding vector array of 
        similar (m'xn') dimensions to the output array.

    parameters{d, dim, dims, Dims: np.array, python array, list, or tuple}
        Most of the time it is obvious how many dimensions you want to deal 
        with: if you supply a matrix, it defines the transformation; if you 
        input offset vectors in the 'preoffset' and 'postoffset' options, those 
        define the number of dimensions.  But if you only supply scalars, there 
        is no way to tell, and the default number of dimensions is 2. A matrix 
        of the same size as the input array will be output.

    Returns
    -------
    apply: np.array
        apply returns an array of dimensions specified by the input 
        parameters. The array contains the forward coordinate transform of the 
        input arrray. If reverse_flag is invoked at the transform class an 
        inverse transform is performed (if possible), see invert below.

    invert: np.array
        invert returns an array of dimensions specified by the input 
        parameters. The array contains the inverse coordinate transform of the 
        input arrray.


    Other Parameters
    ----------------
    -

    Raises
    ------
    transform: t_linear: scale: Scale only accepts scalars and 1D arrays
    
    Linear transform: a matrix and a scale were supplied, the transform cannot 
    handle both. Pass separately
    
    Linear transform: a matrix and a rotation parameter were supplied, the 
    transform cannot handle both. Pass separately
    
    transform: t_linear: matrix: expects a square input matrix. Stopping here
    
    transform: t_linear: rot: expected single angular rotation or three Eular 
    angles. Stopping here


    See Also
    --------
    -

    Notes
    -----
    -

    References
    ----------
    See DeForest, Solar Physics, v219 2004 - ON RE-SAMPLING OF SOLAR IMAGES,
     DOI: 10.1023/B:SOLA.0000021743.24248.b0

    Examples
    --------

    >>> InputArray = np.arange(9.0).reshape(3, 3)
    >>> test_object = transform.t_linear( parameters = {'rotation': 30} )
    >>> test_output = test_object.apply( InputArray )
    >>> 
    array([[-0.5 ,  0.87,  2.  ],
           [ 0.6 ,  4.96,  5.  ],
           [ 1.7 ,  9.06,  8.  ]])

    >>> InputArray = np.arange(9.0).reshape(3, 3)
    >>> test_object = transform.t_linear( parameters = {'rotation': 30} )
    >>> test_output = test_object.invert( InputArray )
    >>> 
    array([[0.50,  0.87,  2.],
           [4.59,  1.96,  5.],
           [8.70,  3.06,  8.]])
    """

    def __init__(self, parameters, reverse_flag = None, name = 't_linear'):
        super().__init__(name, parameters, reverse_flag=reverse_flag)


#---allows for variable variable names
        if 'd' in self.parameters:
            self.parameters['dimensions'] = self.parameters['d']
        if 'dim' in self.parameters:
            self.parameters['dimensions'] = self.parameters['dim']
        if 'dims' in self.parameters:
            self.parameters['dimensions'] = self.parameters['dims']
        if 'Dims' in self.parameters:
            self.parameters['dimensions'] = self.parameters['Dims']

        if 'm' in self.parameters:
            self.parameters['matrix'] = self.parameters['m']
        if 'Matrix' in self.parameters:
            self.parameters['matrix'] = self.parameters['Matrix']

        if 'post' in self.parameters:
            self.parameters['postoffset'] = self.parameters['post']
        if 'shift' in self.parameters:
            self.parameters['postoffset'] = self.parameters['shift']
        if 'Shift' in self.parameters:
            self.parameters['postoffset'] = self.parameters['Shift']

        if 'pre' in self.parameters:
            self.parameters['preoffset'] = self.parameters['pre']
        if 'offset' in self.parameters:
            self.parameters['preoffset'] = self.parameters['offset']
        if 'Offset' in self.parameters:
            self.parameters['preoffset'] = self.parameters['Offset']

        if 'r' in self.parameters:
            self.parameters['rotation'] = self.parameters['r']
        if 'rot' in self.parameters:
            self.parameters['rotation'] = self.parameters['rot']
        if 'rota' in self.parameters:
            self.parameters['rotation'] = self.parameters['rota']
        if 'Rotation' in self.parameters:
            self.parameters['rotation'] = self.parameters['Rotation']

        if 's' in self.parameters:
            self.parameters['scale'] = self.parameters['s']
        if 'Scale' in self.parameters:
            self.parameters['scale'] = self.parameters['Scale']

#---add key parameters if not present and makes them 'None'. It
#   also parses python arrays into numpy arrays if required. It checks for 
#   inputs which will create errors
        if not 'matrix' in self.parameters:
            self.parameters['matrix'] = None
        if (self.parameters['matrix'] is not None): 
            if type(self.parameters['matrix']) is not np.ndarray:
                self.parameters['matrix'] = np.array(self.parameters['matrix'])

        if not 'rotation' in self.parameters:
            self.parameters['rotation'] = None
        if (self.parameters['rotation'] is not None): 
            if type(self.parameters['rotation']) is not np.ndarray:
                self.parameters['rotation'] = np.array(self.parameters['rotation'])

        if not 'scale' in self.parameters:
            self.parameters['scale'] = None

        if self.parameters['scale'] is not None and \
                 type(self.parameters['scale']) is int or \
                 type(self.parameters['scale']) is float or \
                 type(self.parameters['scale']) is complex :

            self.parameters['scale'] = np.atleast_1d(self.parameters['scale'],self.parameters['scale'])
        
        if self.parameters['scale'] is not None and \
                 type(self.parameters['scale']) is np.ndarray or \
                 type(self.parameters['scale']) is list or \
                 type(self.parameters['scale']) is tuple :

            self.parameters['scale'] = np.atleast_1d(self.parameters['scale'])

            if self.parameters['scale'].ndim == 2:
                if (self.parameters['scale'].shape[0] == 1) or (self.parameters['scale'].shape[1] == 1):
                    self.parameters['scale'] = self.parameters['scale'].ravel()

            if self.parameters['scale'].ndim > 2:
                raise ValueError("transform: t_linear: scale: Scale only accepts scalars and 1D arrays")


        if not 'preoffset' in self.parameters:
            self.parameters['preoffset'] = None
        if (self.parameters['preoffset'] is not None): 
            if type(self.parameters['preoffset']) is not np.ndarray:
                self.parameters['preoffset'] = \
                np.atleast_1d(np.array(self.parameters['preoffset']))
                if self.parameters['preoffset'].size == 1:
                    self.parameters['preoffset'] = \
                    np.array([self.parameters['preoffset'][0],self.parameters['preoffset'][0]])

        if not 'postoffset' in self.parameters:
            self.parameters['postoffset'] = None
        if (self.parameters['postoffset'] is not None): 
            if type(self.parameters['postoffset']) is not np.ndarray:
                self.parameters['postoffset'] = \
                np.atleast_1d(np.array(self.parameters['postoffset']))
                if self.parameters['postoffset'].size == 1:
                    self.parameters['postoffset'] = \
                    np.array([self.parameters['postoffset'][0],self.parameters['postoffset'][0]])

        if not 'dimensions' in self.parameters:
            self.parameters['dimensions'] = None

#---Create error if marix and scale or matrix and rot
        if ( (self.parameters['matrix'] is not None) and \
           (self.parameters['scale'] is not None) ):

            raise ValueError("Linear transform: a matrix and a scale were supplied, the transform cannot handle both. Pass separately")

        if ( (self.parameters['matrix'] is not None) and \
           (self.parameters['rotation'] is not None) ):

            raise ValueError("Linear transform: a matrix and a rotation parameter were supplied, the transform cannot handle both. Pass separately")

#---This section uses the input parameters to determine the out put dimensions
        if self.parameters['matrix'] is not None:
            self.input_dim = self.parameters['matrix'].shape[0]
            self.output_dim = self.parameters['matrix'].shape[1]
            if (self.parameters['matrix'].shape[1] != \
                self.parameters['matrix'].shape[0]):
                
                raise ValueError("transform: t_linear: matrix: expects a square input matrix. Stopping here")
        else:
            if self.parameters['rotation'] is not None and \
            type( self.parameters['rotation'] ) is np.ndarray:

                if self.parameters['rotation'].size == 1:
                    self.input_dim = 2
                    self.output_dim = 2
                elif self.parameters['rotation'].size == 3:
                    self.input_dim = 3
                    self.output_dim = 3
                else:
                    raise ValueError("transform: t_linear: rot: expected single angular rotation or three Eular angles. Stopping here")

            elif self.parameters['scale'] is not None:
                if type(self.parameters['scale']) is np.ndarray or \
                 type(self.parameters['scale']) is list or \
                 type(self.parameters['scale']) is tuple :
                    self.input_dim = self.parameters['scale'].shape[0]
                    self.output_dim = self.parameters['scale'].shape[0]
                else:
                     print("transform: t_linear: scale: Scalar detected assuming 2-D tranform")
                     self.parameters['scale'] = np.atleast_1d(np.array(self.parameters['scale']))
                     self.input_dim = 2
                     self.output_dim = 2                    

            elif self.parameters['preoffset'] is not None and \
                 type(self.parameters['preoffset']) is np.ndarray:
                self.input_dim = self.parameters['preoffset'].shape[0]
                self.output_dim = self.parameters['preoffset'].shape[0]

            elif self.parameters['postoffset'] is not None and \
                 type(self.parameters['postoffset']) is np.ndarray:
                self.input_dim = self.parameters['postoffset'].shape[0]
                self.output_dim = self.parameters['postoffset'].shape[0]

            elif self.parameters['dimensions'] is not None:
                    self.input_dim = self.parameters['dimensions']
                    self.output_dim = self.parameters['dimensions']

            else:
                print("transform: t_linear: dims: Insufficient dimensions specified, assuming 2-D transform")
                self.input_dim = 2
                self.output_dim = 2

#---An identity matrix is built based on the input and output dimensions.
            self.parameters['matrix'] = \
              np.zeros((self.input_dim, self.output_dim))

            np.fill_diagonal(self.parameters['matrix'], 1)

#---If rotation matrix is specified apply rotation and multiply by identify 
#   matrix 
        if self.parameters['rotation'] is not None :

            if self.parameters['rotation'].size == 3:
                rotationOutput = Rotation.from_euler('xyz', \
                    [ self.parameters['rotation'][0] , \
                      self.parameters['rotation'][1] , \
                      self.parameters['rotation'][2] ], degrees=True)

                rot_matrix = rotationOutput.as_matrix()                

            if self.parameters['rotation'].size == 1:
                theta = np.deg2rad(self.parameters['rotation'])
                rot_matrix = np.array(( (np.cos(theta), -np.sin(theta)) , \
                                        (np.sin(theta),  np.cos(theta)) ))

            self.parameters['matrix'] = \
              np.matmul(self.parameters['matrix'], rot_matrix)


#---Applies a scale, if scale is not an array, scale is treated as a scalar and 
#   values multiplied as such
        if (self.parameters['scale'] is not None):
            if self.parameters['scale'].size == 1:
                for j in range(self.parameters['matrix'].shape[0]):
                    self.parameters['matrix'][j][j] *= self.parameters['scale']
            else:
                for j in range(self.parameters['matrix'].shape[0]):
                    self.parameters['matrix'][j] *= \
                    self.parameters['scale'][j]


#===Calculate the inverse transform if possible
    def reversetransform(self, data):

        """apply the inverse transform
        If invertable creates the inverse transform for given parameters to the 
        input array (data). The transforms are applied in this order: 
            outdata = (data - post) * inverseMatrixTransform - pre

        Parameters
        ----------
        data : np.array
            array to be transformed

        Raises
        ------
        transform: t_linear: reversetransform: trying to invert a non-invertible matrix.
        """

#---need to check if array is invertable and set flag. Compute the 
#   (multiplicative) inverse of the matrix (np.linalg.inv), if not, flag.
        self._non_invertible = 0
        try:
            self.inverseMatrix = np.linalg.inv(self.parameters['matrix'])
        except np.linalg.LinAlgError:
            self.inverseMatrix = None
            self._non_invertible = 1


        if not self._non_invertible:

#---Test to see if the array dimensions are sufficient for the proposed 
#   transform
            matrixDimension = self.parameters['matrix'].shape[0]

#---Create a deep copy of the data and remove any post transformation is specified
            if self.parameters['postoffset'] is not None:
                dataWithoutPostTransform = \
                  copy.copy(data[..., 0:matrixDimension]) - \
                  self.parameters['postoffset']
            
            else:
                dataWithoutPostTransform = \
                  copy.copy(data[..., 0:matrixDimension])

#---Perform inverse matrix multipliction and remove any pre transform offset if 
#   specified.
            if self.parameters['preoffset'] is not None:
                dataWithoutPreTransform = \
                  np.matmul(dataWithoutPostTransform, self.inverseMatrix) - \
                  self.parameters['preoffset']
            
            else:
                dataWithoutPreTransform = \
                np.matmul(dataWithoutPostTransform, self.inverseMatrix)

            outdata = dataWithoutPreTransform

            return outdata

        else:
            raise ValueError("transform: t_linear: reversetransform: trying to invert a non-invertible matrix.")

    def forwardtransform(self, data, backward):

        """apply the transform for given parameters to the input array (data). 
        The transforms are applied in this order:
            outdata = (data + pre) * transform + post

        Parameters
        ----------
        data : np.array
            array to be transformed

        backward : scalar
            scalar {1|0} if set to 0 a forward transformation is applied, if set
            to 0 the inverse transformation is applied. The value is 
            automatically set to 0.
        """

#---Test for reversible flags
        if (not backward and not self.reverse_flag) or \
           (backward and self.reverse_flag):

#---Test to see if the array dimensions are sufficient for the proposed transform
            matrixDimension = self.parameters['matrix'].shape[0]

#---Create a deep copy of the data array and add the pre transform offset if 
#   specified. Read out the resulting array
            if self.parameters['preoffset'] is not None:
                dataWithPreTransform = \
                  copy.copy(data[..., 0:matrixDimension]) + \
                  self.parameters['preoffset']
            
            else:
                dataWithPreTransform = \
                  copy.copy(data[..., 0:matrixDimension])

            outdata = dataWithPreTransform

#---Perform matrix multipliction and add post transform offset if specified.
            if self.parameters['postoffset'] is not None:
                outdata[..., 0:matrixDimension] = \
                np.matmul(dataWithPreTransform, self.parameters['matrix']) + \
                self.parameters['postoffset']
            else:
                outdata[..., 0:matrixDimension] = \
                np.matmul(dataWithPreTransform, self.parameters['matrix'])
    
            return outdata

        else:
            return self.reversetransform( data )


    def __str__(self):
        outString = f"Transform name: {self.name}\n"\
                    f"Input parameters: {self.parameters}\n"\
                    f"Non-Invertible: {self._non_invertible}\n"\
                    f"Inverse Matrix: {self.inverseMatrix}\n"\
                    f"matrix:{self.parameters['matrix']}\n"\
                    f"scale:{self.parameters['scale']}\n"\
                    f"rot: "f"{self.parameters['rot']}\n"\
                    f"pre: {self.parameters['pre']}\n"\
                    f"post: {self.parameters['post']}\n"\
                    f"dims: {self.parameters['dims']}\n"\
                    f"Reverse Flag: {self.reverse_flag}\n"\
                    f"Non-Invertible: {self._non_invertible}"
                    #f"Input Coord: {self.input_coord}\n"
                    #f"Input Unit:{self.input_unit}\n"
                    #f"Output Coord: {self.output_coord}\n"
                    #f"Output Units: {self.output_unit}\n"
                    #f"Input Dim: {self.input_dim}\n"
                    #f"Output Dim: {self.output_dim}"
        return outString 

