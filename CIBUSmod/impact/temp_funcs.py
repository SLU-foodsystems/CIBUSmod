import numpy as np

def generate_temp_responses(n=100, vers="AR5", model="C2012"):
    """
        Generate temperature impulse response functions (deltaT/kg) for CO2, CH4, and N2O.

        This function computes temperature response curves (per kg emission) over 'n' years,
        based on a selected IPCC methodology and impulse response model. These curves can
        be linearly scaled and convolved with emissions timeseries to produce temperature effects.

        Parameters
        ----------
        n : int
            Number of years to calculate the temperature response over (length of output vectors).
        vers : str, optional
            IPCC version for constants. One of:
                - "AR5" (default)
                - "AR4"
        model : str, optional
            Temperature response model to use. One of:
                - "C2012" (Collins et al. 2012, AGTP formulation)
                - "E2013" (Ericsson et al. 2013, convolution method)
                - "E2014" (Ericsson et al. 2014, empirical constants)
        verbose : bool, optional
            If True, plots the resulting response curves (scaled by GWP100 for CH4/N2O).

        Returns
        -------
        dict of np.ndarray
            Dictionary with keys 'co2', 'ch4', 'n2o' and values being 1D NumPy arrays
            of temperature response per kg of emission over `n+1` time steps.
        """
    from copy import deepcopy

    # Base constants for all versions
    common = {
        "c1": 0.631,
        "c2": 0.429,
        "d1": 8.4,
        "d2": 409.5,
    }

    # Version-specific constants
    cs = deepcopy(common)
    if vers == "AR5":
        cs.update({
            "a0": 0.2173, "a1": 0.2240, "a2": 0.2824, "a3": 0.2763,
            "t1co2": 394.4, "t2co2": 36.54, "t3co2": 4.304,
            "tch4": 12.4, "tn2o": 121.0,
            "f1": 0.5, "f2": 0.15,
            "M0": 1803, "M": 1804, "N0": 324, "N": 325, "C0": 391, "C": 392,
            "AlphaM": 0.036, "AlphaN": 0.12, "AlphaC": 5.35,
            "Ma": 28.97, "MxM": 16.04276, "MxN": 44.01288, "MxC": 44.0098,
            "Tm": 5.1352e18
        })
    elif vers == "AR4":
        cs.update({
            "a0": 0.217, "a1": 0.259, "a2": 0.338, "a3": 0.186,
            "t1co2": 172.9, "t2co2": 18.51, "t3co2": 1.186,
            "tch4": 12.0, "tn2o": 114.0,
            "f1": 0.25, "f2": 0.15,
            "M0": 1774, "M": 1775, "N0": 319, "N": 320, "C0": 378, "C": 379,
            "AlphaM": 0.036, "AlphaN": 0.12, "AlphaC": 5.35,
            "Ma": 28.96, "MxM": 16.04276, "MxN": 44.01288, "MxC": 44.0098,
            "Tm": 5.12e18
        })
    else:
        raise ValueError("Unsupported version specified")

    # Time axis
    t = np.arange(0, n + 1)

    # Radiative efficiency adjustment for CH4/N2O (log terms from IPCC TAR)
    f_mn0 = 0.47 * np.log(
        1 + 2.01e-5 * (cs["M"] * cs["N0"]) ** 0.75 + 5.31e-15 * cs["M"] * (cs["M"] * cs["N0"]) ** 1.52)
    f_m0n0 = 0.47 * np.log(
        1 + 2.01e-5 * (cs["M0"] * cs["N0"]) ** 0.75 + 5.31e-15 * cs["M0"] * (cs["M0"] * cs["N0"]) ** 1.52)
    f_m0n = 0.47 * np.log(
        1 + 2.01e-5 * (cs["M0"] * cs["N"]) ** 0.75 + 5.31e-15 * cs["M0"] * (cs["M0"] * cs["N"]) ** 1.52)

    # Radiative efficiencies (W/m2) for the current concentration increase
    re_ch4v = cs["AlphaM"] * (np.sqrt(cs["M"]) - np.sqrt(cs["M0"])) - (f_mn0 - f_m0n0)
    re_n2ov = cs["AlphaN"] * (np.sqrt(cs["N"]) - np.sqrt(cs["N0"])) - (f_m0n - f_m0n0)
    re_co2v = cs["AlphaC"] * np.log(cs["C"] / cs["C0"])

    # Convert to W/m2 per kg emitted
    f_ch4 = cs["Tm"] / 1e9 / cs["Ma"] * cs["MxM"] * (cs["M"] - cs["M0"])
    f_n2o = cs["Tm"] / 1e9 / cs["Ma"] * cs["MxN"] * (cs["N"] - cs["N0"])
    f_co2 = cs["Tm"] / 1e6 / cs["Ma"] * cs["MxC"] * (cs["C"] - cs["C0"])

    re_ch4 = re_ch4v * (1 + cs["f1"] + cs["f2"]) / f_ch4
    re_n2o = re_n2ov / f_n2o
    re_co2 = re_co2v / f_co2

    # Generate deltaT/kg impulse response functions
    if model == "E2013":
        # Convolution model using decay kernel
        c_co2 = cs["a0"] + cs["a1"] * np.exp(-t / cs["t1co2"]) + cs["a2"] * np.exp(-t / cs["t2co2"]) + cs[
            "a3"] * np.exp(-t / cs["t3co2"])
        c_ch4 = np.exp(-t / cs["tch4"])
        c_n2o = np.exp(-t / cs["tn2o"])

        rf_co2 = re_co2 * c_co2
        rf_ch4 = re_ch4 * c_ch4
        rf_n2o = re_n2o * c_n2o

        rt = cs["c1"] / cs["d1"] * np.exp(-t / cs["d1"]) + cs["c2"] / cs["d2"] * np.exp(-t / cs["d2"])

        dt_co2 = np.convolve(rt, rf_co2)[:n + 1]
        dt_ch4 = np.convolve(rt, rf_ch4)[:n + 1]
        dt_n2o = np.convolve(rt, rf_n2o)[:n + 1]

    elif model == "E2014":
        # Empirical response coefficients
        dt_co2 = re_co2 * (0.23 - 0.02 * np.exp(-t / 1.186) - 0.68 * np.exp(-t / 8.4) + 0.384 * np.exp(
            -t / 18.51) + 0.091 * np.exp(-t / 172.9) - 0.005 * np.exp(-t / 409.5))
        dt_ch4 = re_ch4 * (-1.984 * np.exp(-t / 8.4) + 1.972 * np.exp(-t / 12) + 0.012 * np.exp(-t / 409.5))
        dt_n2o = re_n2o * (-0.643 * np.exp(-t / 8.4) + 0.487 * np.exp(-t / 114) + 0.156 * np.exp(-t / 409.5))

    elif model == "C2012":
        # AGTP model using integrals over exponential decay
        def agtp(a, d):
            return a * cs["c1"] * (1 - np.exp(-t / d))

        dt_co2 = re_co2 * (
                agtp(cs["a0"], cs["d1"]) +
                cs["a1"] * cs["t1co2"] * cs["c1"] / (cs["t1co2"] - cs["d1"]) * (
                            np.exp(-t / cs["t1co2"]) - np.exp(-t / cs["d1"])) +
                cs["a2"] * cs["t2co2"] * cs["c1"] / (cs["t2co2"] - cs["d1"]) * (
                            np.exp(-t / cs["t2co2"]) - np.exp(-t / cs["d1"])) +
                cs["a3"] * cs["t3co2"] * cs["c1"] / (cs["t3co2"] - cs["d1"]) * (
                            np.exp(-t / cs["t3co2"]) - np.exp(-t / cs["d1"])) +
                agtp(cs["a0"], cs["d2"]) +
                cs["a1"] * cs["t1co2"] * cs["c2"] / (cs["t1co2"] - cs["d2"]) * (
                            np.exp(-t / cs["t1co2"]) - np.exp(-t / cs["d2"])) +
                cs["a2"] * cs["t2co2"] * cs["c2"] / (cs["t2co2"] - cs["d2"]) * (
                            np.exp(-t / cs["t2co2"]) - np.exp(-t / cs["d2"])) +
                cs["a3"] * cs["t3co2"] * cs["c2"] / (cs["t3co2"] - cs["d2"]) * (
                            np.exp(-t / cs["t3co2"]) - np.exp(-t / cs["d2"]))
        )

        dt_ch4 = re_ch4 * (
                cs["tch4"] * cs["c1"] / (cs["tch4"] - cs["d1"]) * (
                    np.exp(-t / cs["tch4"]) - np.exp(-t / cs["d1"])) +
                cs["tch4"] * cs["c2"] / (cs["tch4"] - cs["d2"]) * (np.exp(-t / cs["tch4"]) - np.exp(-t / cs["d2"]))
        )

        dt_n2o = re_n2o * (
                cs["tn2o"] * cs["c1"] / (cs["tn2o"] - cs["d1"]) * (
                    np.exp(-t / cs["tn2o"]) - np.exp(-t / cs["d1"])) +
                cs["tn2o"] * cs["c2"] / (cs["tn2o"] - cs["d2"]) * (np.exp(-t / cs["tn2o"]) - np.exp(-t / cs["d2"]))
        )
    else:
        raise ValueError("Unsupported model")

    return {
        "co2": dt_co2,
        "ch4": dt_ch4,
        "n2o": dt_n2o
    }