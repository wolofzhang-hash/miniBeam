from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass
class SectionProps:
    A: float
    Iy: float
    Iz: float
    J: float
    c_y: float
    c_z: float
    Zp_y: float
    Zp_z: float
    shape_factor_y: float
    shape_factor_z: float
    shape_factor_t: float

def rect_solid(b: float, h: float) -> SectionProps:
    # b along z, h along y (for a beam along x; bending about z uses Iz with y distances)
    A = b*h
    Iy = b*h**3/12.0
    Iz = h*b**3/12.0
    # Saint-Venant torsion const approx for rectangle (thin formula not used); simple conservative:
    # Use Roark approx
    a = max(b, h)
    c = min(b, h)
    beta = c/a
    J = a*c**3*(1/3 - 0.21*beta*(1 - beta**4/12))
    c_y = b/2.0
    c_z = h/2.0
    Ze_y = Iy / max(c_y, 1e-12)
    Ze_z = Iz / max(c_z, 1e-12)
    Zp_y_raw = b*h*h/4.0
    Zp_z_raw = b*h*h/4.0
    shape_factor_y = max(Zp_y_raw / max(Ze_y, 1e-12), 1.0)
    shape_factor_z = max(shape_factor_y, Zp_z_raw / max(Ze_z, 1e-12), 1.0)
    # Rectangular sections do not have a simple robust single closed-form torsional plastic
    # correction for all aspect ratios in this lightweight solver; keep conservative.
    shape_factor_t = 1.0
    Zp_y = shape_factor_y*Ze_y
    Zp_z = shape_factor_z*Ze_z
    return SectionProps(A, Iy, Iz, J, c_y, c_z, Zp_y, Zp_z, shape_factor_y, shape_factor_z, shape_factor_t)

def circle_solid(d: float) -> SectionProps:
    r = d/2.0
    A = math.pi*r*r
    I = math.pi*r**4/4.0
    J = math.pi*r**4/2.0
    c_y = r
    c_z = r
    Ze = I / max(r, 1e-12)
    Zp_raw = 4.0*r**3/3.0
    shape_factor = max(Zp_raw / max(Ze, 1e-12), 1.0)
    # Circular solid fully plastic torsion factor Tp/Ty = 4/3.
    shape_factor_t = 4.0/3.0
    Zp = shape_factor*Ze
    return SectionProps(A, I, I, J, c_y, c_z, Zp, Zp, shape_factor, shape_factor, shape_factor_t)

def i_section(h: float, bf: float, tf: float, tw: float) -> SectionProps:
    # Very simplified I-beam about z (bending in XY): use Iz for strong axis with depth h
    # Coordinate: y vertical.
    A = 2*bf*tf + (h-2*tf)*tw
    # Iz about z axis (out of plane) uses widths; but for bending about z, we need Iz about z with y distances:
    # Actually bending moment Mz causes stress from Iz about z? In PyNite local axes: 'Mz' is about local z axis, causing bending in local y.
    # We'll treat Iz as second moment about z (through section, axis out of plane) = sum(Iz parts) with y parallel axis.
    # For rectangle width b, height t: I_about_z = b*t^3/12 where t is thickness in y.
    # Here for flange: b=bf, t=tf.
    Iz_flange = bf*tf**3/12.0
    # web: b=tw, t=(h-2tf)
    Iz_web = tw*(h-2*tf)**3/12.0
    # parallel axis for flanges:
    y = (h/2 - tf/2)
    Iz = 2*(Iz_flange + bf*tf*y*y) + Iz_web
    # Iy (about y) roughly:
    Iy = 2*(tf*bf**3/12.0) + (h-2*tf)*tw**3/12.0
    # torsion J: very rough thin-wall approx
    J = 2*(bf*tf**3/3.0) + (h-2*tf)*tw**3/3.0
    c_y = bf/2.0
    c_z = h/2.0
    Ze_z = Iz / max(c_z, 1e-12)
    Ze_y = Iy / max(c_y, 1e-12)
    a_flange = bf*tf
    y_flange = h/2.0 - tf/2.0
    h_web_half = max(h/2.0 - tf, 0.0)
    a_web_half = tw*h_web_half
    y_web_half = h_web_half/2.0
    Zp_raw = 2.0*(a_flange*y_flange + a_web_half*y_web_half)
    shape_factor_z = max(Zp_raw / max(Ze_z, 1e-12), 1.0)
    # Minor-axis plastic modulus approximation by flange+web split about y-axis.
    Zp_y_raw = tf*bf*bf/2.0 + (h-2*tf)*tw*tw/4.0
    shape_factor_y = max(Zp_y_raw / max(Ze_y, 1e-12), 1.0)
    shape_factor_t = 1.0
    Zp_y = shape_factor_y*Ze_y
    Zp_z = shape_factor_z*Ze_z
    return SectionProps(A, Iy, Iz, J, c_y, c_z, Zp_y, Zp_z, shape_factor_y, shape_factor_z, shape_factor_t)


def rect_hollow(b: float, h: float, t: float) -> SectionProps:
    """Hollow rectangular tube (outer b x h, wall thickness t).

    Geometry convention follows :func:`rect_solid`:
    - b along z
    - h along y
    """
    b = float(b); h=float(h); t=float(t)
    if t <= 0:
        raise ValueError("t must be > 0")
    if 2*t >= min(b, h):
        raise ValueError("t too large for hollow section")
    bi = b - 2*t
    hi = h - 2*t
    A = b*h - bi*hi
    Iy = (b*h**3 - bi*hi**3)/12.0
    Iz = (h*b**3 - hi*bi**3)/12.0

    # Approx torsion const: Roark-type for rectangular tube (thin wall approximation).
    # Use: J ≈ 4*A_m^2 / Σ (s/t) ; where A_m is area enclosed by median lines.
    bm = b - t
    hm = h - t
    Am = bm*hm
    J = 4*(Am**2) / (2*(bm/t) + 2*(hm/t))

    c_y = b/2.0
    c_z = h/2.0
    Ze_y = Iy / max(c_y, 1e-12)
    Ze_z = Iz / max(c_z, 1e-12)
    Zp_y_raw = (b*h*h - bi*hi*hi)/4.0
    Zp_z_raw = (b*h*h - bi*hi*hi)/4.0
    shape_factor_y = max(Zp_y_raw / max(Ze_y, 1e-12), 1.0)
    shape_factor_z = max(shape_factor_y, Zp_z_raw / max(Ze_z, 1e-12), 1.0)
    shape_factor_t = 1.0
    Zp_y = shape_factor_y*Ze_y
    Zp_z = shape_factor_z*Ze_z
    return SectionProps(A, Iy, Iz, J, c_y, c_z, Zp_y, Zp_z, shape_factor_y, shape_factor_z, shape_factor_t)

def circle_hollow(D: float, t: float) -> SectionProps:
    """Hollow circular tube (outer diameter D, wall thickness t)."""
    D=float(D); t=float(t)
    if t <= 0:
        raise ValueError("t must be > 0")
    if 2*t >= D:
        raise ValueError("t too large for hollow circle")
    R = D/2.0
    r = R - t
    A = math.pi*(R**2 - r**2)
    I = math.pi*(R**4 - r**4)/4.0
    J = math.pi*(R**4 - r**4)/2.0
    c_y = R
    c_z = R
    Ze = I / max(c_z, 1e-12)
    Zp_raw = 4.0*(R**3 - r**3)/3.0
    shape_factor = max(Zp_raw / max(Ze, 1e-12), 1.0)
    # Hollow circular tube torsion factor Tp/Ty.
    shape_factor_t = (4.0/3.0) * (R*(R**3 - r**3) / max(R**4 - r**4, 1e-12))
    Zp = shape_factor*Ze
    return SectionProps(A, I, I, J, c_y, c_z, Zp, Zp, shape_factor, shape_factor, max(shape_factor_t, 1.0))
