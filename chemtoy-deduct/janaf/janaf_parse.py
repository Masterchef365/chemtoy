# WARNING: ChatGPT
import re
from collections import defaultdict

# Default atomic masses (standard atomic weights or commonly-used isotopic masses).
# Tweak any entry as needed. This dictionary covers elements 1 (H) through 118 (Og).
ELEMENT_MASSES = {
    "H": 1.00784, "He": 4.002602, "Li": 6.938, "Be": 9.0121831, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224,
    "Nb": 92.90637, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.90550,
    "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196,
    "Ba": 137.327, "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535,
    "Dy": 162.500, "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422, "Yb": 173.045,
    "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.592,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98040, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.0377,
    "Pa": 231.03588, "U": 238.02891, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0,
    "Md": 258.0, "No": 259.0, "Lr": 266.0, "Rf": 267.0, "Db": 268.0,
    "Sg": 269.0, "Bh": 270.0, "Hs": 277.0, "Mt": 278.0, "Ds": 281.0,
    "Rg": 282.0, "Cn": 285.0, "Nh": 286.0, "Fl": 289.0, "Mc": 290.0,
    "Lv": 293.0, "Ts": 294.0, "Og": 294.0,
}

# ---- utility: parse charge (sign before optional digits) ----
def get_charge(formula: str) -> int:
    """
    Parse trailing JANAF-style charge: sign first, optional digits after sign.
    Examples:
      "Al"    -> 0
      "Al+"   -> +1
      "Al-2"  -> -2
      "HAlO2-"-> -1
    """
    f = formula.strip()
    m = re.search(r'([+-])(\d*)$', f)
    if not m:
        return 0
    sign, digits = m.groups()
    magnitude = int(digits) if digits else 1
    return magnitude if sign == "+" else -magnitude

# ---- utility: parse chemical formula (without charge) ----
def parse_formula(formula: str) -> dict:
    """
    Parse a molecular formula (without a trailing charge) into element counts.
    Supports nested parentheses and integer multipliers.
    Returns a dict: {element_symbol: count, ...}
    Raises ValueError on invalid token or unknown element format.
    """
    s = formula.strip()
    if not s:
        return {}

    # We'll implement a recursive parser using a position index.
    token_elem = re.compile(r'[A-Z][a-z]?')    # element symbol (1 or 2 letters)
    token_num = re.compile(r'\d+')

    def parse_segment(i):
        counts = defaultdict(int)
        while i < len(s):
            ch = s[i]
            if ch == '(':
                # parse subgroup
                subgroup_counts, i = parse_segment(i + 1)
                # expect closing ')'
                if i >= len(s) or s[i] != ')':
                    raise ValueError(f"Unmatched '(' in formula at position {i}: {formula}")
                i += 1  # skip ')'
                # optional multiplier after ')'
                m = token_num.match(s, i)
                if m:
                    mul = int(m.group(0))
                    i = m.end()
                else:
                    mul = 1
                for el, cnt in subgroup_counts.items():
                    counts[el] += cnt * mul
                continue
            if ch == ')':
                # end of this segment; caller will handle ')'
                break

            # element token
            m_el = token_elem.match(s, i)
            if not m_el:
                raise ValueError(f"Invalid token at position {i} in formula: {formula}")
            el = m_el.group(0)
            i = m_el.end()
            # optional integer count
            m_num = token_num.match(s, i)
            if m_num:
                cnt = int(m_num.group(0))
                i = m_num.end()
            else:
                cnt = 1
            counts[el] += cnt
        return counts, i

    counts, pos = parse_segment(0)
    if pos != len(s):
        # if leftover characters (like unmatched ')'), that's an error
        raise ValueError(f"Unexpected trailing characters at position {pos} in formula: {formula}")
    return dict(counts)

# ---- main function: compute mass given formula string ----
def formula_mass(formula: str, masses: dict = None) -> float:
    """
    Compute molecular mass for a JANAF-style formula string.
    - Strips a trailing charge (like '+', '-2', '+3') and ignores electron mass.
    - Parses the remaining chemical formula and sums element masses * counts.
    - `masses` is an element-to-mass dictionary (defaults to ELEMENT_MASSES).
    Returns the mass as a float.
    Raises KeyError if an element symbol is not found in masses.
    """
    if masses is None:
        masses = ELEMENT_MASSES

    f = formula.strip()
    # strip charge suffix (sign + optional digits)
    m_charge = re.search(r'([+-]\d*?)$', f)
    if m_charge:
        f_no_charge = f[: m_charge.start()]
    else:
        f_no_charge = f

    f_no_charge = f_no_charge.strip()
    if f_no_charge == "":
        # formula was purely a charge? treat as zero-mass (or you could raise)
        return 0.0

    counts = parse_formula(f_no_charge)
    total = 0.0
    for el, cnt in counts.items():
        if el not in masses:
            raise KeyError(f"Element '{el}' not found in mass lookup table. "
                           "Add it to the ELEMENT_MASSES dict to proceed.")
        total += masses[el] * cnt
    return total

# ---- convenience: combined return (mass, charge, counts) ----
def analyze_formula(formula: str, masses: dict = None):
    """
    Return a tuple (mass, charge, counts_dict) for convenience.
    """
    ch = get_charge(formula)
    m = formula_mass(formula, masses=masses)
    counts = parse_formula(re.sub(r'([+-]\d*?)$', '', formula).strip())
    return m, ch, counts

# ---- Example usage ----
if __name__ == "__main__":
    examples = [
        "Al", "Al+", "Al+3", "Al-2", "AlCl2-", "AlCl3+3",
        "Al2O", "Al2O2+", "BBr2Cl", "HBCl2", "Al2Cl6", "HBF2O"
    ]
    for ex in examples:
        mass, charge, counts = analyze_formula(ex)
        print(f"{ex:12s}  mass={mass:.6f}  charge={charge:+d}  counts={counts}")
