Surface Models
==============

Chroma includes a variety of surface models, which control the interaction of photons with objects. The ``Surface`` class contains many (typically wavelength-dependent) surface parameters, some of which apply to each surface model.

To select a particular model for a surface, set ``surface.model = MODEL`` where ``MODEL`` is one of ``{ SURFACE_DEFAULT, SURFACE_SPECULAR, SURFACE_DIFFUSE, SURFACE_COMPLEX, SURFACE_WLS }``.

``SURFACE_DEFAULT``
-------------------

The default surface model approximates real surfaces by combining a diffuse and specular reflection component. Surfaces may also be absorbing and detecting. The wavelength-dependent probabilities for detection, absorption, diffuse reflection, and specular reflection are provided in uniformly-spaced float arrays of length ``n``, and should sum to unity. The corresponding wavelengths for these arrays are computed from the first (``wavelength0``) and the step size (``step``).

Parameters::

    float *detect
    float *absorb
    float *reflect_diffuse
    float *reflect_specular
    unsigned int n
    float step
    float wavelength0
    
``SURFACE_SPECULAR``
--------------------

A perfect specular reflector. Behavior is identical to ``SURFACE_DEFAULT`` with 100% specular reflection at all wavelengths, but is marginally faster. This model has no parameters.

``SURFACE_DIFFUSE``
-------------------

A perfect diffuse reflector. Behavior is identical to ``SURFACE_DEFAULT`` with 100% diffuse reflection at all wavelengths, but is marginally faster. This model has no parameters.

``SURFACE_COMPLEX``
-------------------

This surface model uses a complex index of refraction (``eta``, ``k``) to compute transmission, (specular) reflection, and absorption probabilities. Surfaces may also be photon detectors, but the detection probability is conditional on detection (``detect`` should be normalized to 1). If a surface is not transmissive (``bool transmissive``), transmitted photons are absorbed.

This model accounts for incoming photon polarization in its calculations.

Parameters::

    float *detect
    float *eta
    float *k
    unsigned int n
    float step
    float wavelength0
    float thickness
    bool transmissive

``SURFACE_WLS``
---------------

A model of wavelength-shifting surfaces. Surfaces may absorb and reflect (these probabilities should sum to 1). Reflection is diffuse. If a photon is absorbed, it may be reemitted with a probability given in ``float *reemit`` (normalized to 1). The reemission spectrum CDF is defined in ``float* reemission_wavelength`` (x) and ``float* reemission_cdf`` (y), both of length ``reemission_n``. The CDF must start at 0 and end at 1. This model does not enforce conservation of energy, and cannot reemit multiple photons!

Parameters::

    float *absorb
    float *reflect
    float *reemit
    float *reemission_wavelength
    float *reemission_cdf
    unsigned int n
    unsigned int reemission_n
    float step
    float wavelength0

