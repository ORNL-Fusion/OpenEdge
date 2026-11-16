.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: surf_collide particulate/bounce

surf_collide particulate/bounce command
=======================================

Syntax
------

.. parsed-literal::

   surf_collide ID particulate/bounce pstick pvel pdiff keyword value ...

* ID = user-assigned name for the surface collision model
* particulate/bounce = style name of this surface collision model
* pstick = probability that an impacting particulate sticks and is
  removed (0 to 1)
* pvel = fraction of the impact speed kept on reflection (0 to 1)
* pdiff = probability that a reflection is diffuse, otherwise it is
  specular (0 to 1)
* zero or more keyword/value pairs may be appended
* keyword = *pmass* or *ptemp*

  * *pmass* value = fraction of the radius kept on reflection, in
    (0,1].  The mass is scaled by its cube.
  * *ptemp* value = factor applied to the particulate temperature on
    reflection (> 0)

Examples
--------

.. code-block:: text

   surf_collide particulate_wall particulate/bounce 0.7 0.4 0.5 &
                pmass 0.9 ptemp 0.8
   surf_modify wall collide particulate_wall

Description
-----------

Define a collision model for particulates that strike a surface.  The
model is assigned to surface elements with the
:doc:`surf_modify <surf_modify>` command.

Symbols are defined on the :ref:`notation <notation>` page.  Write
:math:`p_\mathrm{s}`, :math:`e_v`, :math:`p_\mathrm{d}`,
:math:`f_r` and :math:`f_T` for the values of *pstick*, *pvel*,
*pdiff*, *pmass* and *ptemp*, and :math:`\hat{\mathbf{n}}` for the
surface normal.  For each impact of a particle with positive radius
:math:`r_d` the model does the following.

1. With probability :math:`p_\mathrm{s}` the particulate sticks and is
   removed from the simulation.

2. Otherwise it is reflected.  With probability
   :math:`1 - p_\mathrm{d}` the reflection is specular with the
   restitution coefficient :math:`e_v`,

   .. math::
      :label: pbounce-specular

      \mathbf{v}' = e_v\left[\mathbf{v}
         - 2\,(\mathbf{v}\cdot\hat{\mathbf{n}})\,\hat{\mathbf{n}}\right] .

   With probability :math:`p_\mathrm{d}` it is diffuse.  The speed is
   :math:`v' = e_v\,v` and the direction follows the cosine law about
   the normal,

   .. math::
      :label: pbounce-diffuse

      \mathbf{v}' = e_v\, v\left[\cos\vartheta\,\hat{\mathbf{n}}
         + \sin\vartheta\left(\cos\varphi\,\hat{\mathbf{t}}_1
         + \sin\varphi\,\hat{\mathbf{t}}_2\right)\right] ,
      \qquad
      \cos\vartheta = \sqrt{U_1} , \quad \varphi = 2\pi U_2 ,

   where :math:`\hat{\mathbf{t}}_1` is along the tangential part of
   the incident velocity and
   :math:`\hat{\mathbf{t}}_2 = \hat{\mathbf{n}}\times\hat{\mathbf{t}}_1`.

3. A reflected particulate loses material and heat to the wall,

   .. math::
      :label: pbounce-loss

      r_d' = f_r\, r_d , \qquad
      m_d' = f_r^3\, m_d , \qquad
      T_d' = f_T\, T_d .

The coefficients are constants.  They do not depend on the impact
speed, the impact angle or the temperature of the particulate.  The
mass removed by :math:`f_r < 1` and by sticking is not added to the
surface by this model.

.. note::

   A particle without a positive radius is deleted when it strikes a
   surface that has this model.  Assign the model only to surfaces
   where atoms and ions are meant to be absorbed, or where only
   particulate species arrive.

Restrictions
------------

This command is part of the OPENEDGE package.

*pstick*, *pvel* and *pdiff* must lie in [0,1], *pmass* in (0,1], and
*ptemp* must be positive.

Related commands
----------------

:doc:`surf_modify <surf_modify>`, :doc:`surf_collide <surf_collide>`,
:doc:`fix particulate/emit <fix_particulate_emit>`

Default
-------

The option defaults are pmass = 1.0 and ptemp = 1.0.
