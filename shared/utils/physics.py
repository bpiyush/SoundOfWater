import numpy as np
import torch


# Universal constants
C = 340. * 100.  # Speed of sound in air (cm/s)


def compute_length_of_air_column_cylindrical(
        timestamps, duration, height, b, **kwargs,
    ):
    """
    Randomly chooses a l(t) curve satisfying the two point equations.
    """
    L = height * ( (1 - np.exp(b * (duration - timestamps))) / (1 - np.exp(b * duration)) )
    return L


def compute_axial_frequency_cylindrical(
        lengths, radius, beta=0.62, mode=1, **kwargs,
    ):
    """
    Computes axial resonance frequency for cylindrical container at given timestamps.
    """
    if mode == 1:
        harmonic_weight = 1.
    elif mode == 2:
        harmonic_weight = 3.
    elif mode == 3:
        harmonic_weight = 5.
    else:
        raise ValueError

    # Compute fundamental frequency curve
    F0 = harmonic_weight * (0.25 * C) * (1. / (lengths + (beta * radius)))

    return F0


def compute_axial_frequency_bottleneck(
        lengths, radius, height, Rn, Hn, beta_bottle=(0.6 + 8/np.pi), **kwargs,
    ):
    # Here, R and H are base radius and height of the bottleneck
    eps = 1e-6
    kappa = (0.5 * C / np.pi) * (Rn/radius) * np.sqrt(1 / (Hn + beta_bottle * Rn))
    frequencies = kappa * np.sqrt(1 / (lengths + eps))
    return frequencies


def compute_f0_cylindrical(Y, rho_g, a, R, H, mode=1, **kwargs,):

    if mode == 1:
        m = 1.875
        n = 2
    elif mode == 2:
        m = 4.694
        n = 3
    elif mode == 3:
        m = 7.855
        n = 4
    else:
        raise ValueError

    term = ( ((n**2 - 1)**2) + ((m * R/H)**4) ) / (1 + (1./n**2))
    f0 = (1. / (12 * np.pi)) * np.sqrt(3 * Y / rho_g) * (a / (R**2)) * np.sqrt(term)
    return f0


def compute_xi_cylindrical(rho_l, rho_g, R, a, **kwargs,):
    """
    Different papers use different multipliers.
    For us, using 12. * (4./9.) works best empirically.
    """
    xi = 12. * (4. / 9.) * (rho_l/rho_g) * (R/a)
    return xi


def compute_radial_frequency_cylindrical(
        heights, R, H, Y, rho_g, a, rho_l, power=3, mode=1, **kwargs,
    ):
    """
    Computes radial resonance frequency for cylindrical.

    Args:
        heights (np.ndarray): height of liquid at pre-defined time stamps
    """
    # Only f0 changes for higher modes
    f0 = compute_f0_cylindrical(Y, rho_g, a, R, H, mode=mode)
    xi = compute_xi_cylindrical(rho_l, rho_g, R, a)
    frequencies = f0 / np.sqrt(1 + xi * ((heights/H) ** power) )
    return frequencies


def compute_slant_lengths_semiconical(
        timestamps, duration, r_top, r_bot, height, **kwargs,
    ):

    # Top radius / base radius
    rf =  r_bot / r_top

    # Time fraction
    tf = timestamps/duration
    
    # Height fractions: h(t) / H
    height_fractions = (1. / (rf - 1)) * (np.cbrt(((rf**3 - 1) * (tf)) + 1) - 1)
    
    # Slant air column lengths
    heights = height_fractions * height
    slant_lengths = np.sqrt(1 - ((r_top - r_bot) / height)**2) * (height - heights)

    return slant_lengths


def compute_axial_frequency_semiconical(slant_lengths, r_top, r_bot, beta=1.28, **kwargs):
    """
    Computes axial resonance frequency for cylinder.

    Args:
        slant_lengths (np.ndarray): slant length of air column
        r_top (float): top radius
        r_bot (float): base radius
        beta (float): end correction coefficient
    """
    frequencies_axial = (C / 2) * (1 / (slant_lengths + (beta * (r_bot + r_top))))
    return frequencies_axial


def get_frequencies(
        t,
        params,
        container_shape="cylindrical",
        harmonic=None,
        vibration_type="axial",
        semiconical_as_cylinder=False,
    ):
    """
    Computes requires frequency f(t) for given t.
    """

    if container_shape == "semiconical":
        # Makes an assumption that semiconical shape is similar to cylindrical
        if semiconical_as_cylinder:
            container_shape = "cylindrical"
    
    if (container_shape == "cylindrical") or (container_shape == "bottleneck_as_cylindrical"):

        # Compute length of air column first
        lengths = compute_length_of_air_column_cylindrical(t, **params)

        if vibration_type == "axial":
            frequencies = compute_axial_frequency_cylindrical(lengths, **params)

            if harmonic is not None:
                assert harmonic > 0 and isinstance(harmonic, int)
                frequencies = frequencies * harmonic

        elif vibration_type == "radial":
            if harmonic is None:
                mode = 1
            else:
                assert isinstance(harmonic, int)
                assert harmonic in [1, 2]
                mode = harmonic + 1
            frequencies = compute_radial_frequency_cylindrical(
                lengths, mode=mode, **params,
            )

        else:
            raise NotImplementedError

    elif container_shape == "semiconical":

            # Compute length of air column first
            slant_lengths = compute_slant_lengths_semiconical(t, **params)

            if vibration_type == "axial":
                frequencies = compute_axial_frequency_semiconical(
                    slant_lengths, **params,
                )
    
                if harmonic is not None:
                    assert harmonic > 0 and isinstance(harmonic, int)
                    frequencies = frequencies * harmonic

            else:
                raise NotImplementedError

    elif container_shape == "bottleneck":

        # Compute length of air column first assuming 
        # base of the bottle is a cylindrical
        lengths = compute_length_of_air_column_cylindrical(t, **params)

        if vibration_type == "axial":
            frequencies = compute_axial_frequency_bottleneck(
                lengths, **params,
            )

            if harmonic is not None:
                assert harmonic > 0 and isinstance(harmonic, int)
                frequencies = frequencies * harmonic
        else:
            raise NotImplementedError

    else:
        raise ValueError

    return frequencies


def get_params(row, semiconical_as_cylinder=False):
    m = row["measurements"]
    duration = row["end_time"] - row["start_time"]
    params = dict(duration=duration)
    if row["shape"] == "cylindrical":
        radius = 0.25 * (m["diameter_top"] + m["diameter_bottom"])
        height = m["net_height"]
        params.update(
            height=height,
            radius=radius,
            beta=row.get("beta", 0.62),
            # Constant flow
            b=0.01,
        )
    elif row["shape"] == "semiconical":

        if semiconical_as_cylinder:
            # Assume semiconical shape as cylindrical
            radius = 0.25 * (m["diameter_top"] + m["diameter_bottom"])
            height = m["net_height"]
            params.update(
                height=height,
                radius=radius,
                beta=0.62,
                # Constant flow
                b=0.01,
            )
        else:
            r_top = 0.5 * m["diameter_top"]
            r_bot = 0.5 * m["diameter_bottom"]
            height = m["net_height"]
            beta = 1.28
            params.update(
                r_top=r_top,
                r_bot=r_bot,
                height=height,
                beta=beta,
            )
    elif row["shape"] == "bottleneck":
        radius = 0.5 * m["diameter_bottom"]
        Rn = 0.5 * m["diameter_top"]
        Hn = m["neck_height"] 
        height = m["net_height"] - Hn
        params.update(
            height=height,
            radius=radius,
            Rn=Rn,
            Hn=Hn,
            # Constant flow
            b=0.01,
        )
    elif row["shape"] == "bottleneck_as_cylindrical":
        # Approximates bottleneck as cylindrical
        radius = 0.5 * m["diameter_bottom"]
        height = m["net_height"] + m["neck_height"]
        params.update(
            height=height,
            radius=radius,
            beta=row.get("beta", 0.62),
            # Constant flow
            b=0.01,
        )
    else:
        raise ValueError
    return params

def frequency_to_wavelength(f):
    """
    Converts frequency to wavelength.

    Args:
        f (float): frequency
    """
    return C / f


def wavelength_to_frequency(l):
    """
    Converts wavelength to frequency.

    Args:
        l (float): wavelength
    """
    return C / l


def get_cylinder_radius(m):
    return 0.25 * (m['diameter_top'] + m['diameter_bottom'])


def get_cylinder_height(m):
    return m['net_height']


def get_flow_rate(m, duration):
    r = get_cylinder_radius(m)
    h = get_cylinder_height(m)
    volume = np.pi * (r**2) * h
    q = volume / duration
    return q


def get_length_of_air_column(m, duration, timestamps):
    h = get_cylinder_height(m)
    l = (-h/duration) * timestamps + h
    l = torch.from_numpy(l)
    return l


def estimate_cylinder_radius(wavelengths, timestamps=None, beta=0.62):
    radius_pred = ((1. / beta) * (wavelengths[-1] / 4.)).item()
    return radius_pred


def estimate_cylinder_height(wavelengths, timestamps=None, beta=0.62):
    height_pred = wavelengths[0] / 4. - wavelengths[-1] / 4.
    return height_pred.item()


def estimate_flow_rate(wavelengths, timestamps=None, output_fps=49.):
    radius = estimate_cylinder_radius(wavelengths)
    l_pred = (wavelengths - wavelengths[-1]) / 4.
    slope = np.gradient(l_pred).mean() * output_fps
    Q_pred = -np.pi * (radius**2) * slope
    return Q_pred


def estimate_length_of_air_column(wavelengths, timestamps=None):
    l_pred = (wavelengths - wavelengths[-1]) / 4.
    return l_pred

