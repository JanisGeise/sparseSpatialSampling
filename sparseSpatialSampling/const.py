"""
    stores shared constants used in the dataloader as well as the datawriter class
"""
from enum import IntEnum

# path to the const. attributes in HDF5 file
CONST = "constant"

# path to the grid in HDF5 file
GRID = "grid"

# path to the temporal data in HDF5 file
DATA = "data"

# keys for identifying the cell centers, vertices and faces in the grid group of the HDF5 file
FACES = "faces"
CENTERS = "centers"
VERTICES = "vertices"

"""
Note:
        each cell has 8 neighbors (nb) in 2D case: 1 at each side, 1 at each corner of the cell. In 3D, there
        are additionally 9 neighbors of the plane above and below (in z-direction) present. The neighbors are
        assigned in clockwise direction starting at the left neighbor. For 3D, first the neighbors in the same plane are
        assigned, then the neighbors in lower plane and finally the neighbors in upper plane. The indices of the
        neighbors list refer to the corresponding neighbors as:

        2D:
            0 = left nb, 1 = upper left nb, 2 = upper nb, 3 = upper right nb, 4 = right nb, 5 = lower right nb,
            6 = lower nb, 7 = lower left nb

        additionally for 3D:

            8 = left nb lower plane, 9 = upper left nb lower plane,  10 = upper nb lower plane,
            11 = upper right nb lower plane, 12 = right nb lower plane, 13 = lower right nb lower plane,
            14 = lower nb lower plane, 15 = lower left nb lower plane, 16 = center nb lower plane

            17 = left nb upper plane, 18 = upper left nb upper plane, 19 = upper nb upper plane,
            20 = upper right nb upper plane, 21 = right nb upper plane, 22 = lower right nb upper plane,
            23 = lower nb upper plane, 24 = lower left nb upper plane, 25 = center nb upper plane

        for readability, actual positions (indices) are replaced with cardinal directions as:
            n = north, e = east, s = south, w = west, l = lower plane, u = upper plane, c = center

        e.g., 'swu' := south-west neighbor cell in the plane above the current cell
"""

class Neighbor(IntEnum):
    # 2D neighbors
    W = 0
    NW = 1
    N = 2
    NE = 3
    E = 4
    SE = 5
    S = 6
    SW = 7
    # 3D: lower plane
    WL = 8
    NWL = 9
    NL = 10
    NEL = 11
    EL = 12
    SEL = 13
    SL = 14
    SWL = 15
    CL = 16
    # 3D: upper plane
    WU = 17
    NWU = 18
    NU = 19
    NEU = 20
    EU = 21
    SEU = 22
    SU = 23
    SWU = 24
    CU = 25


class Child(IntEnum):
    # 2D = quadtree (suffix U = upper plane, L = lower plane)
    SWU = 0
    NWU = 1
    NEU = 2
    SEU = 3
    SWL = 4
    NWL = 5
    NEL = 6
    SEL = 7


# -------------------------
# Neighbor relations in 2D
# -------------------------
NEIGHBOR_RELATIONS_2D = {
    Child.SWU: [
        (Neighbor.W,  Neighbor.W,  Child.SEU),
        (Neighbor.NW, Neighbor.W,  Child.NEU),
        (Neighbor.N,  None,        Child.NWU),
        (Neighbor.NE, None,        Child.NEU),
        (Neighbor.E,  None,        Child.SEU),
        (Neighbor.SE, Neighbor.S,  Child.NEU),
        (Neighbor.S,  Neighbor.S,  Child.NWU),
        (Neighbor.SW, Neighbor.SW, Child.NEU),
    ],
    Child.NWU: [
        (Neighbor.W,  Neighbor.W,  Child.NEU),
        (Neighbor.NW, Neighbor.NW, Child.SEU),
        (Neighbor.N,  Neighbor.N,  Child.SWU),
        (Neighbor.NE, Neighbor.N,  Child.SEU),
        (Neighbor.E,  None,        Child.NEU),
        (Neighbor.SE, None,        Child.SEU),
        (Neighbor.S,  None,        Child.SWU),
        (Neighbor.SW, Neighbor.W,  Child.SEU),
    ],
    Child.NEU: [
        (Neighbor.W,  None,        Child.NWU),
        (Neighbor.NW, Neighbor.N,  Child.SWU),
        (Neighbor.N,  Neighbor.N,  Child.SEU),
        (Neighbor.NE, Neighbor.NE, Child.SWU),
        (Neighbor.E,  Neighbor.E,  Child.NWU),
        (Neighbor.SE, Neighbor.E,  Child.SWU),
        (Neighbor.S,  None,        Child.SEU),
        (Neighbor.SW, None,        Child.SWU),
    ],
    Child.SEU: [
        (Neighbor.W,  None,        Child.SWU),
        (Neighbor.NW, None,        Child.NWU),
        (Neighbor.N,  None,        Child.NEU),
        (Neighbor.NE, Neighbor.E,  Child.NWU),
        (Neighbor.E,  Neighbor.E,  Child.SWU),
        (Neighbor.SE, Neighbor.SE, Child.NWU),
        (Neighbor.S,  Neighbor.S,  Child.NEU),
        (Neighbor.SW, Neighbor.SW, Child.NWU),
    ],
}


# -------------------------
# Neighbor relations in 3D
# -------------------------
NEIGHBOR_RELATIONS_3D = {
    Child.SWL: [
        # lower-left child, lower plane (= south-west lower)
        (Neighbor.W, Neighbor.W, Child.SEL),
        (Neighbor.NW, Neighbor.W, Child.NEL),
        (Neighbor.N, None, Child.NWL),
        (Neighbor.NE, None, Child.NEL),
        (Neighbor.E, None, Child.SEL),
        (Neighbor.SE, Neighbor.S, Child.NEL),
        (Neighbor.S, Neighbor.S, Child.NWL),
        (Neighbor.SW, Neighbor.SW, Child.NEL),

        (Neighbor.WL, Neighbor.WL, Child.SEU),
        (Neighbor.NWL, Neighbor.WL, Child.NEU),
        (Neighbor.NL, Neighbor.CL, Child.NWU),
        (Neighbor.NEL, Neighbor.CL, Child.NEU),
        (Neighbor.EL, Neighbor.CL, Child.SEU),
        (Neighbor.SEL, Neighbor.SL, Child.NEU),
        (Neighbor.SL, Neighbor.SL, Child.NWU),
        (Neighbor.SWL, Neighbor.SWL, Child.NEU),
        (Neighbor.CL, Neighbor.CL, Child.SWU),

        (Neighbor.WU, Neighbor.W, Child.SEU),
        (Neighbor.NWU, Neighbor.W, Child.NEU),
        (Neighbor.NU, None, Child.NWU),
        (Neighbor.NEU, None, Child.NEU),
        (Neighbor.EU, None, Child.SEU),
        (Neighbor.SEU, Neighbor.S, Child.NEU),
        (Neighbor.SU, Neighbor.S, Child.NWU),
        (Neighbor.SWU, Neighbor.SW, Child.NEU),
        (Neighbor.CU, None, Child.SWU),
    ],

    Child.NWL: [
        # upper-left child, lower plane (= north-west lower)
        (Neighbor.W, Neighbor.W, Child.NEL),
        (Neighbor.NW, Neighbor.NW, Child.SEL),
        (Neighbor.N, Neighbor.N, Child.SWL),
        (Neighbor.NE, Neighbor.N, Child.SEL),
        (Neighbor.E, None, Child.NEL),
        (Neighbor.SE, None, Child.SEL),
        (Neighbor.S, None, Child.SWL),
        (Neighbor.SW, Neighbor.W, Child.SEL),

        (Neighbor.WL, Neighbor.WL, Child.NEU),
        (Neighbor.NWL, Neighbor.NWL, Child.NEU),
        (Neighbor.NL, Neighbor.NL, Child.SWU),
        (Neighbor.NEL, Neighbor.NL, Child.SEU),
        (Neighbor.EL, Neighbor.CL, Child.NEU),
        (Neighbor.SEL, Neighbor.CL, Child.SEU),
        (Neighbor.SL, Neighbor.CL, Child.SWU),
        (Neighbor.SWL, Neighbor.WL, Child.SEU),
        (Neighbor.CL, Neighbor.CL, Child.NWU),

        (Neighbor.WU, Neighbor.W, Child.NEU),
        (Neighbor.NWU, Neighbor.NW, Child.SEU),
        (Neighbor.NU, Neighbor.N, Child.SWU),
        (Neighbor.NEU, Neighbor.N, Child.SEU),
        (Neighbor.EU, None, Child.NEU),
        (Neighbor.SEU, None, Child.SEU),
        (Neighbor.SU, None, Child.SWU),
        (Neighbor.SWU, Neighbor.W, Child.SEU),
        (Neighbor.CU, None, Child.NWU),
    ],

    Child.NEL: [
        # upper-right child, lower plane (= north-east lower)
        (Neighbor.W, None, Child.NWL),
        (Neighbor.NW, Neighbor.N, Child.SWL),
        (Neighbor.N, Neighbor.N, Child.SEL),
        (Neighbor.NE, Neighbor.NE, Child.SWL),
        (Neighbor.E,  Neighbor.E, Child.NWL),
        (Neighbor.SE, Neighbor.E, Child.SWL),
        (Neighbor.S, None, Child.SEL),
        (Neighbor.SW, None, Child.SWL),

        (Neighbor.WL, Neighbor.CL, Child.NWU),
        (Neighbor.NWL, Neighbor.NL, Child.SWU),
        (Neighbor.NL, Neighbor.NL, Child.SEU),
        (Neighbor.NEL, Neighbor.NEL, Child.SWU),
        (Neighbor.EL, Neighbor.EL, Child.NWU),
        (Neighbor.SEL, Neighbor.EL, Child.SWU),
        (Neighbor.SL, Neighbor.CL, Child.SEU),
        (Neighbor.SWL, Neighbor.CL, Child.SWU),
        (Neighbor.CL, Neighbor.CL, Child.NEU),

        (Neighbor.WU, None, Child.NWU),
        (Neighbor.NWU, Neighbor.N, Child.SWU),
        (Neighbor.NU, Neighbor.N, Child.SEU),
        (Neighbor.NEU, Neighbor.NE, Child.SWU),
        (Neighbor.EU, Neighbor.E, Child.NWU),
        (Neighbor.SEU, Neighbor.E, Child.SWU),
        (Neighbor.SU, None, Child.SEU),
        (Neighbor.SWU, None, Child.SWU),
        (Neighbor.CU, None, Child.NEU),
    ],

    Child.SEL: [
        # lower-right child, lower plane (= south-east lower)
        (Neighbor.W, None, Child.SWL),
        (Neighbor.NW, None, Child.NWL),
        (Neighbor.N, None, Child.NEL),
        (Neighbor.NE, Neighbor.E, Child.NWL),
        (Neighbor.E, Neighbor.E, Child.SWL),
        (Neighbor.SE, Neighbor.SE, Child.NWL),
        (Neighbor.S, Neighbor.S, Child.NEL),
        (Neighbor.SW, Neighbor.S, Child.NWL),

        (Neighbor.WL, Neighbor.CL, Child.SWU),
        (Neighbor.NWL, Neighbor.CL, Child.NWU),
        (Neighbor.NL, Neighbor.CL, Child.NEU),
        (Neighbor.NEL, Neighbor.EL, Child.NWU),
        (Neighbor.EL, Neighbor.EL, Child.SWU),
        (Neighbor.SEL, Neighbor.SEL, Child.NWU),
        (Neighbor.SL, Neighbor.SL, Child.NEU),
        (Neighbor.SWL, Neighbor.SL, Child.NWU),
        (Neighbor.CL, Neighbor.CL, Child.SEU),

        (Neighbor.WU, None, Child.SWU),
        (Neighbor.NWU, None, Child.NWU),
        (Neighbor.NU, None, Child.NEU),
        (Neighbor.NEU, Neighbor.E, Child.NWU),
        (Neighbor.EU, Neighbor.E, Child.SWU),
        (Neighbor.SEU, Neighbor.SE, Child.NWU),
        (Neighbor.SU, Neighbor.S, Child.NEU),
        (Neighbor.SWU, Neighbor.S, Child.NWU),
        (Neighbor.CU, None, Child.SEU),
    ],

    Child.SWU: [
        # lower-left child, upper plane (= south-west upper)
        (Neighbor.WL, Neighbor.W, Child.SEL),
        (Neighbor.NWL, Neighbor.W, Child.NEL),
        (Neighbor.NL, None, Child.NWL),
        (Neighbor.NEL, None, Child.NEL),
        (Neighbor.EL, None, Child.SEL),
        (Neighbor.SEL, Neighbor.S, Child.NEL),
        (Neighbor.SL, Neighbor.S, Child.NWL),
        (Neighbor.SWL, Neighbor.SW, Child.NEL),
        (Neighbor.CL, None, Child.SWL),

        (Neighbor.WU, Neighbor.WU, Child.SEL),
        (Neighbor.NWU, Neighbor.WU, Child.NEL),
        (Neighbor.NU, Neighbor.CU, Child.NWL),
        (Neighbor.NEU, Neighbor.CU, Child.NEL),
        (Neighbor.EU, Neighbor.CU, Child.SEL),
        (Neighbor.SEU, Neighbor.SU, Child.NEL),
        (Neighbor.SU, Neighbor.SU, Child.NWL),
        (Neighbor.SWU, Neighbor.SWU, Child.NEL),
        (Neighbor.CU, Neighbor.CU, Child.SWL),
    ],

    Child.NWU: [
        # upper-left child, upper plane (= north-west upper)
        (Neighbor.WL, Neighbor.W, Child.NEL),
        (Neighbor.NWL, Neighbor.NW, Child.SEL),
        (Neighbor.NL, Neighbor.N, Child.SWL),
        (Neighbor.NEL, Neighbor.N, Child.SEL),
        (Neighbor.EL, None, Child.NEL),
        (Neighbor.SEL, None, Child.SEL),
        (Neighbor.SL, None, Child.SWL),
        (Neighbor.SWL, Neighbor.W, Child.SEL),
        (Neighbor.CL, None, Child.NWL),

        (Neighbor.WU, Neighbor.WU, Child.NEL),
        (Neighbor.NWU, Neighbor.NWU, Child.SEL),
        (Neighbor.NU, Neighbor.NU, Child.SWL),
        (Neighbor.NEU, Neighbor.NU, Child.SEL),
        (Neighbor.EU, Neighbor.CU, Child.NEL),
        (Neighbor.SEU, Neighbor.CU, Child.SEL),
        (Neighbor.SU, Neighbor.CU, Child.SWL),
        (Neighbor.SWU, Neighbor.WU, Child.SEL),
        (Neighbor.CU, Neighbor.CU, Child.NWL),
    ],

    Child.NEU: [
        # upper right child, upper plane (= north-east upper)
        (Neighbor.WL, None, Child.NWL),
        (Neighbor.NWL, Neighbor.N, Child.SWL),
        (Neighbor.NL, Neighbor.N, Child.SEL),
        (Neighbor.NEL, Neighbor.NE, Child.SWL),
        (Neighbor.EL, Neighbor.E, Child.NWL),
        (Neighbor.SEL, Neighbor.E, Child.SWL),
        (Neighbor.SL, None, Child.SEL),
        (Neighbor.SWL, None, Child.SWL),
        (Neighbor.CL, None, Child.NEL),

        (Neighbor.WU, Neighbor.CU, Child.NWL),
        (Neighbor.NWU, Neighbor.NU, Child.SWL),
        (Neighbor.NU, Neighbor.NU, Child.SEL),
        (Neighbor.NEU, Neighbor.NEU, Child.SWL),
        (Neighbor.EU, Neighbor.EU, Child.NWL),
        (Neighbor.SEU, Neighbor.EU, Child.SWL),
        (Neighbor.SU, Neighbor.CU, Child.SEL),
        (Neighbor.SWU, Neighbor.CU, Child.SWL),
        (Neighbor.CU, Neighbor.CU, Child.NEL),
    ],

    # lower right child, upper plane (= south-east upper)
    Child.SEU: [
        (Neighbor.WL,  None,        Child.SWL),
        (Neighbor.NWL, None,        Child.NWL),
        (Neighbor.NL,  None,        Child.NEL),
        (Neighbor.NEL, Neighbor.E,  Child.NWL),
        (Neighbor.EL,  Neighbor.E,  Child.SWL),
        (Neighbor.SEL, Neighbor.SE, Child.NWL),
        (Neighbor.SL,  Neighbor.S,  Child.NEL),
        (Neighbor.SWL, Neighbor.S,  Child.NWL),
        (Neighbor.CL, None,         Child.SEL),

        (Neighbor.WU,  Neighbor.CU,  Child.SWL),
        (Neighbor.NWU, Neighbor.CU,  Child.NWL),
        (Neighbor.NU,  Neighbor.CU,  Child.NEL),
        (Neighbor.NEU, Neighbor.EU,  Child.NWL),
        (Neighbor.EU,  Neighbor.EU,  Child.SWL),
        (Neighbor.SEU, Neighbor.SEU, Child.NWL),
        (Neighbor.SU,  Neighbor.SU,  Child.NEL),
        (Neighbor.SWU, Neighbor.SU,  Child.NWL),
        (Neighbor.CU,  Neighbor.CU,  Child.SEL),
    ],
}

if __name__ == "__main__":
    pass
