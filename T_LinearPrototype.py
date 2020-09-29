from abc import ABC, abstractmethod
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
    def apply(self, data, backward = 0):
        pass

#---The invert 
    def invert(self, data):
        return self.apply(data, backward = 1)

#    @abstractmethod
#    def inputDataShape( self, data, backward = 0 ):
#        pass




class t_linear(Transform):

    def __init__(self, parameters, reverse_flag, name = 't_linear'):
        super().__init__(name, parameters, reverse_flag)

#---Begin by defining the the input and output shape of the matrix. If a matrix
#   is present, the dimensions are used, else the rotation, scale, pre shift,
#   post shift, defined or assumed dimensions are used. With no attributes, a
#   2-D transform is assumed. If no matrix is entered, one is built based upon 
#   the parameters used.



    def apply(self, data, backward = 0):

#---this bit adds key parameters if not present and makes them None
        if not 'matrix' in self.parameters:
            self.parameters['matrix'] = None
        if not 'rot' in self.parameters:
            self.parameters['rot'] = None
        if not 'scale' in self.parameters:
            self.parameters['scale'] = None
        if not 'pre' in self.parameters:
            self.parameters['pre'] = None
        if not 'post' in self.parameters:
            self.parameters['post'] = None
        if not 'dims' in self.parameters:
            self.parameters['dims'] = None

        if self.parameters['matrix'] is not None:

            self.input_dim = self.parameters['matrix'].shape[0]
            self.output_dim = self.parameters['matrix'].shape[1]
            
            #print(self.parameters[ 'matrix' ].shape[0])
            #print(self.parameters[ 'matrix' ].shape[1])

        else:

            if self.parameters['rot'] is not None and \
            type( self.parameters['rot'] ) is np.ndarray:

                if self.parameters['rot'].size == 1:
                    self.input_dim = 2
                    self.output_dim = 2
                elif self.parameters['rot'].size == 3:
                    self.input_dim = 3
                    self.output_dim = 3

                print("We have a rot")

            elif self.parameters['scale'] is not None and \
                 type(self.parameters['scale']) is np.ndarray or \
                 type(self.parameters['scale']) is list or \
                 type(self.parameters['scale']) is tuple :
                self.parameters['scale'] = np.array(self.parameters['scale'])
                self.input_dim = self.parameters['scale'].shape[0]
                self.output_dim = self.parameters['scale'].shape[0]
                print("We have a scale")

            elif self.parameters['pre'] is not None and \
                 type(self.parameters['pre']) is np.ndarray:

                self.input_dim = self.parameters['pre'].shape[0]
                self.output_dim = self.parameters['pre'].shape[0]
                print("We have a pre")

            elif self.parameters['post'] is not None and \
                 type(self.parameters['post']) is np.ndarray:

                self.input_dim = self.parameters['post'].shape[0]
                self.output_dim = self.parameters['post'].shape[0]
                print("We have a post")

            elif self.parameters['dims'] is not None:
                #self.parameters['dims'] = np.array(self.parameters['dims'])

                self.input_dim = self.parameters['dims']
                self.output_dim = self.parameters['dims']
#                print("We have dims")
                print(self.parameters['dims'])

            else:

                print("Insufficient dimensions specified, assuming input data \
                    shape")
                self.input_dim = data.shape[1]
                self.output_dim = data.shape[1]
                print("We have nothing")
            
#            print("it's input dimension is: ", self.input_dim)
#            print("it's output dimension is: ", self.output_dim)

#---An identity matrix is built based on the input and output dimensions, through a 
#   zero matrix with the diagnal filled with 1's.

            self.parameters['matrix'] = \
                                    np.zeros((self.input_dim, self.output_dim))
            np.fill_diagonal(self.parameters['matrix'], 1)

#            print(self.parameters['matrix'] )
#---If scale is not an array, scale is treated as a scalar and values multiplied 

        if (self.parameters['scale'] is not None) and not \
                           (isinstance((self.parameters['scale']), np.ndarray)):
            print("scalar")
            for j in range(self.parameters['matrix'].shape[0]):
                self.parameters['matrix'][j][j] *= self.parameters['scale']
 
        elif type(self.parameters['scale']) is np.ndarray:
            print("vector")

            if self.parameters['scale'].shape[0] > data.shape[1]:
                raise ValueError(f"Transform vector exceeds number of matrix \
                    columns, {np.shape(data)[-1]} or fewer elements expected")
            else:
                if self.parameters['scale'].ndim > 1:
                    raise ValueError("Scale only accepts scalars and 1D arrays")
                else:
                    for j in range(self.parameters['matrix'].shape[0]):
                        self.parameters['matrix'][j] *= \
                        self.parameters['scale'][j]

#---To Delete

        # print(f"data size: {data[0].size}")'
        # og_shape = data.shape
        # if data.ndim == 1:
        #     data = data.reshape((1, 1, data.size))
        # else:
        #    data = data.reshape((data.shape[0], 1, data[0].size))
 #       print("outside apply")

        if (not backward and not self.reverse_flag) or \
           (backward and self.reverse_flag):

#---Test to see if the array dimensions are correct
            matrixDimension = self.parameters['matrix'].shape[0]
            if matrixDimension > np.shape(data)[-1]:
#---Needs rewording
                raise ValueError(f"Linear transform: transform requires at \
                      least a {np.shape(data)[-1]}  dimension columns array ")

#---consider changing pre and post to define what they do - offset!
            if self.parameters['pre'] is not None:
#---Creates a deep copy of the data array and adds the pre defined offset.
                dataWithPreTransform = \
                                 copy.deepcopy(data[..., 0:matrixDimension]) + \
                                 self.parameters['pre']
            else:
#---If no preset, creates copy of the data array
                dataWithPreTransform = \
                                 copy.deepcopy(data[..., 0:matrixDimension])

#---Create the output data array
            outdata = copy.deepcopy(dataWithPreTransform)

#            print(matrixDimension)
#            print(data[..., 0:matrixDimension])
#            print(dataWithPreTransform)

            # print(f"out shape: {out.shape}")

            if self.parameters['post'] is not None:
#---If no post set movement, creates copy of the data array, this multiplies the array with the input matrix and adds an offset
                outdata[..., 0:matrixDimension] = \
                np.matmul(dataWithPreTransform, self.parameters['matrix']) + \
                self.parameters['post']
            else:
#---If no postset, creates copy of the data array and then mutiplies by the offset
                outdata[..., 0:matrixDimension] = \
                np.matmul(dataWithPreTransform, self.parameters['matrix'])

            # out = out.reshape(og_shape)
            return outdata

        elif not self._non_invertible:
            print("In Two")

        else:
            print("trying to invert a non-invertible matrix.")



    def __str__(self):
        out_string =  f"Transform name: {self.name}\n"\
                      f"Input parameters: {self.parameters}\n"\
                      f"Non-Invertible: {self._non_invertible}\n"
        return out_string 

