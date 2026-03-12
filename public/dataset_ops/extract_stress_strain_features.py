import numpy as np
from simstack.core.context import context

from models.stress_strain_model import StrainStressModel


async def extract_stress_strain_features(stress: np.ndarray, strain: np.ndarray, curve_number:int) -> StrainStressModel:
    """
    Extract key features from a stress-strain curve.
    
    Features extracted:
    - linear_region: Young's modulus (estimated slope) and proportional limit.
    - yield_strength: The first local maximum of the curve.
    - ultimate_strength: The second (highest) local maximum of the curve.
    - fracture: The last point of the stress-strain data.
    
    Parameters
    ----------
    stress : np.ndarray
        1D array of engineering stress values.
    strain : np.ndarray
        1D array of engineering strain values.
    curve_number : int
    Returns
    -------
    dict
        Dictionary containing the extracted features.

    """
    # Remove NaNs if any (common at the end of data if fixed-size arrays were used)
    mask = ~np.isnan(stress) & ~np.isnan(strain)
    stress = stress[mask]
    strain = strain[mask]

    if len(stress) < 2:
        return {}

    # 1. Linear region / Young's modulus
    # Use only very initial points where strain is very small to avoid Lüders/Plastic effects
    # In this dataset, E is around 210,000 MPa, so stress should be E * strain
    # The first point is (0,0). Let's take points where strain < 0.001
    initial_mask = strain < 0.001
    if np.sum(initial_mask) >= 2:
        e_modulus, b = np.polyfit(strain[initial_mask], stress[initial_mask], 1)
    else:
        # Fallback to first few points
        e_modulus, b = np.polyfit(strain[:5], stress[:5], 1)
    
    # Proportional limit is where it deviates from linearity
    # Let's find the first point where stress < (E * strain - tolerance)
    # or just use the first point where slope starts to decrease significantly.
    deviation = np.abs(stress - (e_modulus * strain + b))
    # Threshold could be 1% of max stress
    threshold = 0.01 * np.max(stress)
    indices_above_threshold = np.where(deviation > threshold)[0]
    if len(indices_above_threshold) > 0:
        idx_prop = indices_above_threshold[0]
    else:
        idx_prop = int(len(stress) * 0.05)
    
    linear_region = {
        "youngs_modulus_MPa": float(e_modulus),
        "proportional_limit_index": int(idx_prop),
        "proportional_limit_stress": float(stress[idx_prop]),
        "proportional_limit_strain": float(strain[idx_prop])
    }

    # 2. Local maxima (Yield Strength and Ultimate Tensile Strength)
    # We use a simple difference-based peak detection or look for where slope changes sign
    # Since find_peaks is in scipy, and we might want to avoid extra dependencies if possible
    # But scipy is likely available in this env.
    
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(stress)
    except ImportError:
        # Fallback: simple peak detection
        peaks = np.where((stress[1:-1] > stress[:-2]) & (stress[1:-1] > stress[2:]))[0] + 1

    yield_strength = {}
    ultimate_strength = {}

    if len(peaks) >= 1:
        # First maximum is yield strength (especially if upper yield exists)
        idx_yield = peaks[0]
        yield_strength = {
            "index": int(idx_yield),
            "stress_MPa": float(stress[idx_yield]),
            "strain": float(strain[idx_yield])
        }
    
    if len(peaks) >= 2:
        # Second (usually highest) maximum is UTS
        # Actually UTS is the global maximum in engineering stress before necking
        # But if we have multiple peaks, we take the one after the yield plateau
        idx_uts = peaks[-1] # Usually the last peak before fracture
        ultimate_strength = {
            "index": int(idx_uts),
            "stress_MPa": float(stress[idx_uts]),
            "strain": float(strain[idx_uts])
        }
    elif len(peaks) == 1:
        # If only one peak, it might be UTS if there's no distinct yield point
        # Or it might be yield if the data cut off early.
        # Let's assume global max is UTS if it's not the same as yield.
        idx_max = np.argmax(stress)
        if idx_max != peaks[0]:
            ultimate_strength = {
                "index": int(idx_max),
                "stress_MPa": float(stress[idx_max]),
                "strain": float(strain[idx_max])
            }
    else:
        # No peaks found, use argmax
        idx_max = np.argmax(stress)
        ultimate_strength = {
            "index": int(idx_max),
            "stress_MPa": float(stress[idx_max]),
            "strain": float(strain[idx_max])
        }

    # 3. Fracture (last point)
    fracture = {
        "index": int(len(stress) - 1),
        "stress_MPa": float(stress[-1]),
        "strain": float(strain[-1])
    }

    analysis_result = {
        "linear_region": linear_region,
        "yield_strength": yield_strength,
        "ultimate_strength": ultimate_strength,
        "fracture": fracture
    }

    # Pack analysis results into StrainStressModel
    strain_stress_model = StrainStressModel(
        curve_index=curve_number,
        linear_region=analysis_result["linear_region"],
        yield_strength=analysis_result.get("yield_strength"),
        ultimate_strength=analysis_result["ultimate_strength"],
        fracture=analysis_result["fracture"]
    )
    await context.db.save(strain_stress_model)
    return strain_stress_model