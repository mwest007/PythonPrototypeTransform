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
from scipy.interpolate import interpn
from scipy.interpolate import RegularGridInterpolator
import copy

__author__ = "Matthew J West, Craig DeForest, Jake-R-W"
__copyright__ = "By all the listed autors: GPLv2, There is no warranty."
__credits__ = ["Matthew J West", "Craig DeForest", "Jake-R-W"]
__license__ = "-"
__version__ = "1.0.0"
__maintainer__ = "Matthew J West"
__email__ = "mwest@swri.boulder.edu"
__status__ = "Production"


def ndcoords(*dims):
    """
    number of coordinates indexer for given dimensions, initizilzed to a tuple

    ...

    Attributes
    ----------
    dims : tuple, list or numpy array
        dimensions of the input array
    grid_size : str
        the name of the animal
    sound : str
        the sound that the animal makes
    out : numpy mesh grid
        return a dense multi-dimensional “meshgrid” modified  and transposed 
        array cast to float64.

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes

    Usage
    ----- 
    
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

    #return a dense multi-dimensional “meshgrid”
    out = np.mgrid[grid_size]

    #returns a modified transposed array cast to float64.
    out = out.astype('float64').transpose()
    return out



class Transform( ABC ):

    def __init__(self, name, input_coord, input_unit,
                 output_coord, output_unit, parameters,
                 reverse_flag, input_dim = None,
                 output_dim = None):
        """
        The Transform class
    
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

    def invert(self, data):
        return self.apply(data, backward=1)

    def Map(self, data=None, template=None, pdl=None, opts=None):
        """
        Map Function - Transform image coordinates (Transform method)
    
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
        if tmp is None:
            tmp = dims #$in
#            tmp = [$in->dims] #Create an array of variables from dims and put in array tmp

        print( "In Transform Map")
        #$my = 
        return 0




"""
*PDL::map = \&map;
sub map {
  my($me) = shift; <---reads first value, deletes it and shifts array to left
  my($in) = shift;

  if(UNIVERSAL::isa($me,'PDL') && UNIVERSAL::isa($in,'PDL::Transform')) {
      my($x) = $in;
      $in = $me;
      $me = $x;
  }  <---this looks to see if the called class is a part of PDL or transform.

  barf ("PDL::Transform::map: source is not defined or is not a PDL\n")
    unless(defined $in and  UNIVERSAL::isa($in,'PDL'));

  barf ("PDL::Transform::map: source is empty\n")
    unless($in->nelem);

  my($tmp) = shift;
  my($opt) = shift;

  # Check for options-but-no-template case
  if(ref $tmp eq 'HASH' && !(defined $opt)) {
    if(!defined($tmp->{NAXIS})) {  # FITS headers all have NAXIS.
      $opt = $tmp;
      $tmp = undef;
    }
  }


"""

#  croak("PDL::Transform::map: Option 'p' was ambiguous and has been removed. You probably want 'pix' or 'phot'.") if exists($opt->{'p'});

"""

  $tmp = [$in->dims]  unless(defined($tmp));

  # Generate an appropriate output piddle for values to go in
  my($out);
  my(@odims);
  my($ohdr);
  if(UNIVERSAL::isa($tmp,'PDL')) {
    @odims = $tmp->dims;

    my($x);
    if(defined ($x = $tmp->gethdr)) {
      my(%b) = %{$x};
      $ohdr = \%b;
    }
  } elsif(ref $tmp eq 'HASH') {
    # (must be a fits header -- or would be filtered above)
    for my $i(1..$tmp->{NAXIS}){
      push(@odims,$tmp->{"NAXIS$i"});
    }
    # deep-copy fits header into output
    my %foo = %{$tmp};
    $ohdr = \%foo;
  } elsif(ref $tmp eq 'ARRAY') {
    @odims = @$tmp;
  } else {
    barf("map: confused about dimensions of the output array...\n");
  }

  if(scalar(@odims) < scalar($in->dims)) {
    my @idims = $in->dims;
    push(@odims, splice(@idims,scalar(@odims)));
  }

  $out = PDL::new_from_specification('PDL',$in->type,@odims);
  $out->sethdr($ohdr) if defined($ohdr);

  if($PDL::Bad::Status) {
    # set badflag on output all the time if possible, to account for boundary violations
    $out->badflag(1);
  }

  ##############################
  ## Figure out the dimensionality of the
  ## transform itself (extra dimensions come along for the ride)
  my $nd = $me->{odim} || $me->{idim} || 2;
  my @sizes = $out->dims;
  my @dd = @sizes;

  splice @dd,$nd; # Cut out dimensions after the end

  # Check that there are elements in the output fields...
  barf "map: output has no dims!\n"
        unless(@dd);
  my $ddtotal = 1;
  map {$ddtotal *= $_} @dd;
  barf "map: output has no elements (at least one dim is 0)!\n"
     unless($ddtotal);


  ##############################
  # If necessary, generate an appropriate FITS header for the output.

  my $nofits = _opt($opt, ['nf','nofits','NoFITS','pix','pixel','Pixel']);

  ##############################
  # Autoscale by transforming a subset of the input points' coordinates
  # to the output range, and pick a FITS header that fits the output
  # coordinates into the given template.
  #
  # Autoscaling always produces a simple, linear mapping in the FITS header.
  # We support more complex mappings (via t_fits) but only to match a pre-existing
  # FITS header (which doesn't use autoscaling).
  #
  # If the rectify option is set (the default) then the image is rectified
  # in scientific coordinates; if it is clear, then the existing matrix
  # is used, preserving any shear or rotation in the coordinate system.
  # Since we eschew CROTA whenever possible, the CDi_j formalism is used instead.
  my $f_in = (defined($in->hdr->{NAXIS}) ? t_fits($in,{ignore_rgb=>1}) : t_identity());

  unless((defined $out->gethdr && $out->hdr->{NAXIS})  or  $nofits) {
      print "generating output FITS header..." if($PDL::Transform::debug);

      $out->sethdr($in->hdr_copy) # Copy extraneous fields...
        if(defined $in->hdr);

      my $samp_ratio = 300;

      my $orange = _opt($opt, ['or','orange','output_range','Output_Range'],
                        undef);

      my $omin;
      my $omax;
      my $osize;


      my $rectify = _opt($opt,['r','rect','rectify','Rectify'],1);


      if (defined $orange) {
          # orange always rectifies the coordinates -- the output scientific
          # coordinates *must* align with the axes, or orange wouldn't make
          # sense.
        print "using user's orange..." if($PDL::Transform::debug);
        $orange = pdl($orange) unless(UNIVERSAL::isa($orange,'PDL'));
        barf "map: orange must be 2xN for an N-D transform"
          unless ( (($orange->dim(1)) == $nd )
                   && $orange->ndims == 2);

        $omin = $orange->slice("(0)");
        $omax = $orange->slice("(1)");
        $osize = $omax - $omin;

        $rectify = 1;

      } else {

          ##############################
          # Real autoscaling happens here.

          if(!$rectify and ref( $f_in ) !~ /Linear/i) {
              if( $f_in->{name} ne 'identity' ) {
                 print STDERR "Warning: map can't preserve nonlinear FITS distortions while autoscaling.\n";
              }
              $rectify=1;
          }

          my $f_tr = ( $rectify ?
                       $me x $f_in :
                       (  ($me->{name} eq 'identity') ?  # Simple optimization for match()
                          $me :                          # identity -- just matching
                          !$f_in x $me x $f_in           # common case
                       )
                       );

          my $samps = (pdl(($in->dims)[0..$nd-1]))->clip(0,$samp_ratio);

          my $coords = ndcoords(($samps + 1)->list);

          my $t;
          my $irange = _opt($opt, ['ir','irange','input_range','Input_Range'],
                            undef);

          # If input range is defined, sample that quadrilateral -- else
          # sample the quad defined by the boundaries of the input image.
          if(defined $irange) {
              print "using user's irange..." if($PDL::Transform::debug);
              $irange = pdl($irange) unless(UNIVERSAL::isa($irange,'PDL'));
              barf "map: irange must be 2xN for an N-D transform"
                  unless ( (($irange->dim(1)) == $nd )
                           && $irange->ndims == 2);

              $coords *= ($irange->slice("(1)") - $irange->slice("(0)")) / $samps;
              $coords += $irange->slice("(0)");
              $coords -= 0.5; # offset to pixel corners...
              $t = $me;
          } else {
              $coords *= pdl(($in->dims)[0..$nd-1]) / $samps;
              $coords -= 0.5; # offset to pixel corners...
              $t = $f_tr;
          }
          my $ocoords = $t->apply($coords)->mv(0,-1)->clump($nd);

          # discard non-finite entries
          my $oc2  = $ocoords->range(
              which(
                  $ocoords->
                  xchg(0,1)->
                  sumover->
                  isfinite
              )
              ->dummy(0,1)
              );

          $omin = $oc2->minimum;
          $omax = $oc2->maximum;

          $osize = $omax - $omin;
          my $tosize;
          ($tosize = $osize->where($osize == 0)) .= 1.0;
      }

      my ($scale) = $osize / pdl(($out->dims)[0..$nd-1]);

      my $justify = _opt($opt,['j','J','just','justify','Justify'],0);
      if($justify) {
          my $tmp; # work around perl -d "feature"
          ($tmp = $scale->slice("0")) *= $justify;
          $scale .= $scale->max;
          $scale->slice("0") /= $justify;
      }

      print "done with autoscale. Making fits header....\n" if($PDL::Transform::debug);
      if( $rectify ) {
          # Rectified header generation -- make a simple coordinate header with no
          # rotation or skew.
          print "rectify\n" if($PDL::Transform::debug);
          for my $d(1..$nd) {
              $out->hdr->{"CRPIX$d"} = 1 + ($out->dim($d-1)-1)/2 ;
              $out->hdr->{"CDELT$d"} = $scale->at($d-1);
              $out->hdr->{"CRVAL$d"} = ( $omin->at($d-1) + $omax->at($d-1) ) /2 ;
              $out->hdr->{"NAXIS$d"} = $out->dim($d-1);
              $out->hdr->{"CTYPE$d"} = ( (defined($me->{otype}) ?
                                          $me->{otype}->[$d-1] : "")
                                         || $in->hdr->{"CTYPE$d"}
                                         || "");
              $out->hdr->{"CUNIT$d"} = ( (defined($me->{ounit}) ?
                                          $me->{ounit}->[$d-1] : "")
                                         || $in->hdr->{"CUNIT$d"}
                                         || $in->hdr->{"CTYPE$d"}
                                         || "");
          }
          $out->hdr->{"NAXIS"} = $nd;

          $out->hdr->{"SIMPLE"} = 'T';
          $out->hdr->{"HISTORY"} .= "Header written by PDL::Transform::Cartography::map";

          ### Eliminate fancy newfangled output header pointing tags if they exist
          ### These are the CROTA<n>, PCi_j, and CDi_j.
          for $k(keys %{$out->hdr})      {
              if( $k=~m/(^CROTA\d*$)|(^(CD|PC)\d+_\d+[A-Z]?$)/ ){
                  delete $out->hdr->{$k};
              }
          }
      } else {
          # Non-rectified output -- generate a CDi_j matrix instead of the simple formalism.
          # We have to deal with a linear transformation: we've got:  (scaling) x !input x (t x input),
          # where input is a linear transformation with offset and scaling is a simple scaling. We have
          # the scaling parameters and the matrix for !input -- we have only to merge them and then we
          # can write out the FITS header in CDi_j form.
          print "non-rectify\n" if($PDL::Transform::debug);
          my $midpoint_val = (pdl(($out->dims)[0..$nd-1])/2 * $scale)->apply( $f_in );
          print "midpoint_val is $midpoint_val\n" if($PDL::Transform::debug);
          # linear transformation
          unless(ref($f_in) =~ m/Linear/) {
              croak("Whups -- got a nonlinear t_fits transformation.  Can't deal with it.");
          }

          my $inv_sc_mat = zeroes($nd,$nd);
          $inv_sc_mat->diagonal(0,1) .= $scale;
          my $mat = $f_in->{params}->{matrix} x $inv_sc_mat;
          print "scale is $scale; mat is $mat\n" if($PDL::Transform::debug);

          print "looping dims 1..$nd: " if($PDL::Transform::debug);
          for my $d(1..$nd) {
              print "$d..." if($PDL::Transform::debug);
              $out->hdr->{"CRPIX$d"} = 1 + ($out->dim($d-1)-1)/2;
              $out->hdr->{"CRVAL$d"} = $midpoint_val->at($d-1);
              $out->hdr->{"NAXIS$d"} = $out->dim($d-1);
              $out->hdr->{"CTYPE$d"} = ( (defined($me->{otype}) ?
                                          $me->{otype}->[$d-1] : "")
                                         || $in->hdr->{"CTYPE$d"}
                                         || "");
              $out->hdr->{"CUNIT$d"} = ( (defined($me->{ounit}) ?
                                          $me->{ounit}->[$d-1] : "")
                                         || $in->hdr->{"CUNIT$d"}
                                         || $in->hdr->{"CTYPE$d"}
                                         || "");
              for my $e(1..$nd) {
                  $out->hdr->{"CD${d}_${e}"} = $mat->at($d-1,$e-1);
                  print "setting CD${d}_${e} to ".$mat->at($d-1,$e-1)."\n" if($PDL::Transform::debug);
              }
          }

          ## Eliminate competing header pointing tags if they exist
          for $k(keys %{$out->hdr}) {
              if( $k =~ m/(^CROTA\d*$)|(^(PC)\d+_\d+[A-Z]?$)|(CDELT\d*$)/ ) {
                  delete $out->hdr->{$k};
              }
          }
      }



    }

  $out->hdrcpy(1);

  ##############################
  # Sandwich the transform between the input and output plane FITS headers.
  unless($nofits) {
      $me = !(t_fits($out,{ignore_rgb=>1})) x $me x $f_in;
  }

  ##############################
  ## Figure out the interpND options
  my $method = _opt($opt,['m','method','Method'],undef);
  my $bound = _opt($opt,['b','bound','boundary','Boundary'],'t');


  ##############################
  ## Rubber meets the road: calculate the inverse transformed points.
  my $ndc = PDL::Basic::ndcoords(@dd);
  my $idx = $me->invert($ndc->double);

  barf "map: Transformation had no inverse\n" unless defined($idx);

  ##############################
  ## Integrate ?  (Jacobian, Gaussian, Hanning)
  my $integrate = ($method =~ m/^[jghJGH]/) if defined($method);

  ##############################
  ## Sampling code:
  ## just transform and interpolate.
  ##  ( Kind of an anticlimax after all that, eh? )
  if(!$integrate) {
    my $x = $in->interpND($idx,{method=>$method, bound=>$bound});
    my $tmp; # work around perl -d "feature"
    ($tmp = $out->slice(":")) .= $x; # trivial slice prevents header overwrite...
    return $out;
  }

  ##############################
  ## Anti-aliasing code:
  ## Condition the input and call the pixelwise C interpolator.
  ##

  barf("PDL::Transform::map: Too many dims in transformation\n")
        if($in->ndims < $idx->ndims-1);

  ####################
  ## Condition the threading -- pixelwise interpolator only threads
  ## in 1 dimension, so squish all thread dimensions into 1, if necessary
  my @iddims = $idx->dims;
  if($in->ndims == $#iddims) {
        $in2 = $in->dummy(-1,1);
  } else {
        $in2 = ( $in
                ->reorder($nd..$in->ndims-1, 0..$nd-1)
                ->clump($in->ndims - $nd)
                ->mv(0,-1)
               );
  }

  ####################
  # Allocate the output array
  my $o2 = PDL->new_from_specification($in2->type,
                                    @iddims[1..$#iddims],
                                    $in2->dim(-1)
                                   );

  ####################
  # Pack boundary string if necessary
  if(defined $bound) {
    if(ref $bound eq 'ARRAY') {
      my ($s,$el);
      foreach $el(@$bound) {
        barf "Illegal boundary value '$el' in range"
          unless( $el =~ m/^([0123fFtTeEpPmM])/ );
        $s .= $1;
      }
      $bound = $s;
    }
    elsif($bound !~ m/^[0123ftepx]+$/  && $bound =~ m/^([0123ftepx])/i ) {
      $bound = $1;
    }
  }

  ####################
  # Get the blur and minimum-sv values
  my $blur  =  _opt($opt,['blur','Blur'],1.0);
  my $svmin =  _opt($opt,['sv','SV'],1.0);
  my $big   =  _opt($opt,['big','Big'],1.0);
  my $flux  =  _opt($opt,['phot','photometry'],0);
  my @idims = $in->dims;

  $flux = ($flux =~ m/^[1fF]/);
  $big = $big * max(pdl(@idims[0..$nd]));
  $blur = $blur->at(0) if(ref $blur);
  $svmin =  $svmin->at(0)  if(ref $svmin);

  my $bv;
  if($PDL::Bad::Status  and $in->badflag){
      $bv = $in->badvalue;
  } else {
      $bv = 0;
  }

  ### The first argument is a dummy to set $GENERIC.
  $idx = double($idx) unless($idx->type == double);
  print "Calling _map_int...\n" if($PDL::Transform::debug);
  &PDL::_map_int( $in2->flat->index(0),
        $in2, $o2, $idx,
        $bound, $method, $big, $blur, $svmin, $flux, $bv);

  my @rdims = (@iddims[1..$#iddims], @idims[$#iddims..$#idims]);
  {
     my $tmp; # work around perl -d "feature"
     ($tmp = $out->slice(":")) .= $o2->reshape(@rdims);
  }
  return $out;
}



"""























"""
        # if template is not None:
        #     self.output_dim = template
        # else:
        self.output_dim = data.shape
        out = np.empty(shape=self.output_dim, dtype=np.float64)
        dd = out.shape
        ndc = ndcoords(dd)
        # ndc = ndc.reshape((np.product(ndc.shape[:-1]), ndc.shape[-1]))
        idx = self.apply(ndc, backward=1)
        pixel_grid = [np.arange(x) for x in data.shape]
        x = interpn(points=pixel_grid, values=data, method='linear', xi=idx, bounds_error=False,
                    fill_value=0)
        out[:] = x
        return out.transpose()


    def unmap(self, data=None, template=None, pdl=None, opts=None):


        apply
        invert
        map
        match
        map
        unmap
        t_inverse
        t_compose
        t_wrap
        t_identity
        t_lookup
        t_linear
        t_scale
        t_offset
        t_rot
        t_fits
        t_code
        t_cylindrical
        t_radial
        t_quadratic
        t_cubic
        t_quartic
        t_spherical
        t_projective
"""









