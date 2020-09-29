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
    s, scale, Scale : np.array
        A scaling scalar, vector, or matrix.  If you specify a vector
        it is treated as a diagonal matrix (for convenience).  It gets
        left-multiplied with the transformation matrix you specify (or the
        identity), so that if you specify both a scale and a matrix the
        scaling is done after the rotation or skewing or whatever.

    r, rot, rota, rotation, Rotation : scalar; np.array
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

    m, matrix, Matrix: np.array

        The transformation matrix.  It does not even have to be square, if you 
        want to change the dimensionality of your input.  If it is invertible 
        (note: must be square for that), then you automatically get an inverse 
        transform too.

    pre, preoffset, offset, Offset: float

        The vector to be added to the (mxn) data array before being multiplied 
        by any input matrix (equivalent of CRVAL in FITS, if you are converting 
        from scientific to pixel units). pre will accept: a single element vector 
        which is broadcast over the whole input data array, a python array or 
        numpy nx1 column array vector, where the values are broadcast over 
        corresponding rows, or a corresponding vector array of similar (mxn)
        dimensions to the input array.

    post, postoffset, shift, Shift: float

        The vector to be added to the (mxn) data array after being multiplied 
        by any input matrix (equivalent of CRPIX-1 in FITS, if you are 
        converting from scientific to pixel units). pre will accept: a single 
        element vector which is broadcast over the whole input data array, a 
        python array or numpy n'x1 column array vector, where the values are 
        broadcast over corresponding rows, or a corresponding vector array of 
        similar (m'xn') dimensions to the output array.

    d, dim, dims, Dims

        Most of the time it is obvious how many dimensions you want to deal 
        with: if you supply a matrix, it defines the transformation; if you 
        input offset vectors in the 'pre' and 'post' options, those define the 
        number of dimensions.  But if you only supply scalars, there is no way 
        to tell, and the default number of dimensions is 2.  This provides a way 
        to do, e.g., 3-D scaling: just set dims=3.

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


    References
    ----------


    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print([x + 3 for x in a])
    [4, 5, 6]
    >>> print("a\nb")
    a
    b
    """






