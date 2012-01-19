import numpy as np
import time
import datetime
import sys
import math

def count_nonzero(array):
    '''Return the number of nonzero elements in this array'''
    return int((array != 0).sum())

def filled_array(value, shape, dtype):
    '''Create a numpy array of given `shape` and `dtype` filled with the scalar `value`.'''
    a = np.empty(shape=shape, dtype=dtype)
    a.fill(value)
    return a

def ufloat_to_str(x):
    msd = -int(math.floor(math.log10(x.std_dev())))
    return '%.*f +/- %.*f' % (msd, round(x.nominal_value, msd),
                              msd, round(x.std_dev(), msd))

def progress(seq):
    "Print progress while iterating over `seq`."
    n = len(seq)
    print '[' + ' '*21 + ']\r[',
    sys.stdout.flush()
    update_interval = max(n // 10, 1)
    for i, item in enumerate(seq):
        if i % update_interval == 0:
            print '.',
            sys.stdout.flush()
        yield item
    print ']'
    sys.stdout.flush()

def debugger_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
       # we are in interactive mode or we don't have a tty-like
       # device, so we call the default hook
       sys.__excepthook__(type, value, tb)
    else:
       import traceback, pdb
       # we are NOT in interactive mode, print the exception...
       traceback.print_exception(type, value, tb)
       print
       # ...then start the debugger in post-mortem mode.
       pdb.pm()

def enable_debug_on_crash():
    "Start the PDB console when an uncaught exception propagates to the top."
    sys.excepthook = debugger_hook

# allow profile decorator to exist, but do nothing if not running under
# kernprof
try:
    profile_if_possible = profile
except NameError:
    profile_if_possible = lambda x: x

def timeit(func):
    "A decorator to print the time elapsed in a function call."
    def f(*args, **kwargs):
        t0 = time.time()
        retval = func(*args, **kwargs)
        elapsed = time.time() - t0
        print '%s elapsed in %s().' % (datetime.timedelta(seconds=elapsed), func.__name__)
        return retval
    return f

def read_csv(filename):
    """Return an array of comma-separated values from `filename`."""
    f = open(filename)

    points = []
    for line in f:
        try:
            points.append([float(s) for s in line.split(',')])
        except ValueError:
            pass

    f.close()

    return np.array(points)

def offset(points, x):
    """
    Return the set of points obtained by offsetting the edges of the profile
    created by `points` by an amount `x`.

    Args:
        - points: array
            Array of points which define the 2-D profile to be offset.
        - x: float
            Distance to offset the profile; a positive `x` value will offset
            the profile in the direction of the profile path rotated 90 degrees
            clockwise.
    """
    points = np.asarray(points)
    points = np.array([points[0] - (points[1] - points[0])] + list(points) + [points[-1] - (points[-2] - points[-1])])
    
    offset_points = []
    for i in range(1,len(points)-1):
        v1 = np.cross(points[i]-points[i-1], (0,0,1))[:2]
        v1 /= np.linalg.norm(v1)
        v1 *= x

        a = points[i-1] + v1
        b = points[i] + v1

        v2 = np.cross(points[i+1]-points[i], (0,0,1))[:2]
        v2 /= np.linalg.norm(v2)
        v2 *= x

        c = points[i] + v2
        d = points[i+1] + v2

        m = np.empty((2,2))
        m[:,0] = b-a
        m[:,1] = c-d

        try:
            j = np.linalg.solve(m, c-a)[0]
        except np.linalg.linalg.LinAlgError as e:
            offset_points.append(b)
            continue

        offset_points.append((a + j*(b-a)))

    return np.array(offset_points)

def memoize_method_with_dictionary_arg(func):
    def lookup(*args):
        # based on function by Michele Simionato
        # http://www.phyast.pitt.edu/~micheles/python/
        # Modified to work for class method with dictionary argument

        assert len(args) == 2
        # create hashable arguments by replacing dictionaries with tuples of items
        dict_items = args[1].items()
        dict_items.sort()
        hashable_args = (args[0], tuple(dict_items))
        try:
            return func._memoize_dic[hashable_args]
        except AttributeError:
            # _memoize_dic doesn't exist yet.

            result = func(*args)
            func._memoize_dic = {hashable_args: result}
            return result
        except KeyError:
            result = func(*args)
            func._memoize_dic[hashable_args] = result
            return result
    return lookup

