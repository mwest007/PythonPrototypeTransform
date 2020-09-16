#!/usr/bin/env python

"""Transform - Coordinate transforms, image warping, and N-D functions.

Transform is a convenient way to represent coordinate transformations and 
resample images.  It embodies functions mapping R^N -> R^M, both with and 
without inverses.  Provision exists for parametrizing functions, and for 
composing them.  You can use this part of the Transform object to keep track of
arbitrary functions mapping R^N -> R^M with or without inverses.

Usage
----- 
outputarray = Transform.ndcoords( inputarray )

"""

import numpy as np
import pandas as pd
import astropy.units as units
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interpn, RegularGridInterpolator
import copy

__author__ = "Matthew J West, Craig DeForest, Jake-R-W"
__copyright__ = "By all the listed autors: GPLv2, There is no warranty."
__credits__ = ["Matthew J West", "Craig DeForest", "Jake-R-W"]
__license__ = "-"
__version__ = "1.0.0"
__maintainer__ = "Matthew J West"
__email__ = "mwest@swri.boulder.edu"
__status__ = "Production"


class TestPerson:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello my name is " + self.name);

    def myAgeCatYears(self):
        return (self.age * 7);



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






class Transform( ABC ):

    def __init__(self, name, input_coord, input_unit,
                 output_coord, output_unit, parameters,
                 reverse_flag, input_dim = None,
                 output_dim = None):
        """Transform - Coordinate transforms, image warping, and N-D functions

        Transform is a convenient way to represent coordinate transformations 
        and resample images.  It embodies functions mapping R^N -> R^M, both 
        with and without inverses.  Provision exists for parametrizing 
        functions, and for composing them.  You can use this part of the 
        Transform object to keep track of arbitrary functions mapping R^N -> R^M
        with or without inverses.

        The simplest way to use a Transform object is to transform vector data 
        between coordinate systems.  The apply method accepts an array or
        variable whose 0th dimension is coordinate index (all other dimensions 
        are threaded over) and transforms the vectors into the new coordinate 
        system.

        Transform also includes image resampling, via the map method. You define
        a coordinate transform using a Transform object, then use it to remap an
        image PDL.  The output is a remapped, resampled image.

        You can define and compose several transformations, then apply them all 
        at once to an image.  The image is interpolated only once, when all the 
        composed transformations are applied.

        In keeping with standard practice, but somewhat counterintuitively, the 
        map engine uses the inverse transform to map coordinates FROM the 
        destination dataspace (or image plane) TO the source dataspace; hence 
        transform keeps track of both the forward and inverse transform.

        For terseness and convenience, most of the constructors are exported
        into the current package with the name t_<transform> >>, so the 
        following (for example) are synonyms:
.
    
        Attributes
        ----------
        says_str : str
            a formatted string to print out what the animal says
        name : str
            the name of the animal
        sound : str
            the sound that the animal makes
        num_legs : int
            the number of legs the animal has (default 4)
    
name : 
input_coord : 
input_unit : 
output_coord : 
output_unit : 
parameters : 
reverse_flag : 
input_dim = None,
                 output_dim = None



        Methods
        -------
        says(sound=None)
            Prints the animals name and what sound it makes
    
        Usage
        ----- 

        """
        self.name = name
        self.input_coord = input_coord
        self.input_unit = input_unit
        self.output_coord = output_coord
        self.output_unit = output_unit
        self.parameters = parameters
        self._non_invertible = 0
        self.reverse_flag = reverse_flag
        self.input_dim = input_dim
        self.output_dim = output_dim


    @abstractmethod
    def apply(self, data, backward=0):
        """
        apply creates the standard method for the class
    
        ...
    
        Attributes
        ----------
        -    
        
        Methods
        -------
        -

        Usage
        ----- 

        """

        pass


    def apply(self, data, backward=0):
        # print(f"data size: {data[0].size}")'
        # og_shape = data.shape
        # if data.ndim == 1:
        #     data = data.reshape((1, 1, data.size))
        # else:
        #    data = data.reshape((data.shape[0], 1, data[0].size))
        # print(data.shape)
        if (not backward and not self.reverse_flag) or (backward and self.reverse_flag):
            d = self.parameters['matrix'].shape[0]
            # print(data.shape)
            if d > np.shape(data)[-1]:
                raise ValueError(f"Linear transform: transform is {np.shape(data)[-1]} data only ")

            if self.parameters['pre'] is not None:
                x = copy.deepcopy(data[..., 0:d]) + self.parameters['pre']
            else:
                x = copy.deepcopy(data[..., 0:d])

            out = copy.deepcopy(data)
            # print(f"out shape: {out.shape}")
            if self.parameters['post'] is not None:
                out[..., 0:d] = np.matmul(x, self.parameters['matrix']) + self.parameters['post']
            else:
                out[..., 0:d] = np.matmul(x, self.parameters['matrix'])

            # out = out.reshape(og_shape)
            return out

        elif not self._non_invertible:

            d = self.inv.shape[0]
            if d > np.shape(data)[-1]:
                raise ValueError(f"Linear transform: transform is {np.shape(data)[-1]} data only ")

            if self.parameters['pre'] is not None:
                x = copy.deepcopy(data[..., 0:d]) + self.parameters['pre']
            else:
                x = copy.deepcopy(data[..., 0:d])

            out = copy.deepcopy(data)
            if self.parameters['post'] is not None:
                out[..., 0:d] = np.matmul(x, self.inv) + self.parameters['post']
            else:
                out[..., 0:d] = np.matmul(x, self.inv)

            return out
        else:
            print("trying to invert a non-invertible matrix.")




    def match(self, pdl, opts=None):
        return self.map(pdl=pdl, opts=opts)

    def invert(self, data):
        return self.apply(data, backward=1)

    def Map(self, data=None, template=None, pdl=None, opts=None):
        """
        Map Function - Transform image coordinates (Transform method). Resample 
        a scientific image to the same coordinate system as another.
    
        ...
    
        Attributes
        ----------
        says_str : str
            a formatted string to print out what the animal says
        name : str
            the name of the animal
        sound : str
            the sound that the animal makes
        num_legs : int
            the number of legs the animal has (default 4)
    
        Methods
        -------
        says(sound=None)
            Prints the animals name and what sound it makes
    
        Usage
        ----- 

        import TransformMatt.transformMap as transformMap
        im1 = transformMap( im0 )
        """
        var = 2
        if var:
            print('it exists')
        else:
            print( 'nope it does not')
            tmp = dims #$in
#            tmp = [$in->dims] #Create an array of variables from dims and put in array tmp

        print( "In Transform Map")
        #$my = 
        return 0

