
import jax
import jax.numpy as np

def compute_normal(edge):
    tangent = edge[1]-edge[0]
    normal = np.array([tangent[1], -tangent[0]])
    return normal / np.linalg.norm(normal)


def average_normals(normal0, normal1):
    normalDiff = normal0 - normal1
    return normalDiff / np.linalg.norm(normalDiff)



def eval_linear_field_on_edge(field, xi):
    return field[0] * (1.0 - xi) + field[1] * xi


def compute_intersection(edgeA, edgeB, normal):

    def compute_xi(xa, edgeB, normal):
        # solve xa - xb(xi) + g * normal = 0
        # xb = edgeB[0] * (1-xi) + edgeB[1] * xi
        M = np.array([edgeB[0]-edgeB[1], normal]).T
        r = np.array(edgeB[0]-xa)
        xig = np.linalg.solve(M,r)
        return xig[0], xig[1]

    xiBs1, gs1 = jax.vmap(compute_xi, (0,None,None))(edgeA, edgeB, normal)
    xiAs2, gs2 = jax.vmap(compute_xi, (0,None,None))(edgeB, edgeA,-normal)

    xiAs = np.hstack((np.arange(2), xiAs2))
    xiBs = np.hstack((xiBs1, np.arange(2)))
    gs = np.hstack((gs1, gs2))

    xiAgood = jax.vmap(lambda xia, xib: np.where((xia >= 0.0) & (xia <= 1.0) & (xib >= 0.0) & (xib <= 1.0), xia, np.nan))(xiAs, xiBs)
    argsMinMax = np.array([np.nanargmin(xiAgood), np.nanargmax(xiAgood)])

    return xiAs[argsMinMax], xiBs[argsMinMax], gs[argsMinMax]


def length(edge):
    return np.linalg.norm(edge[0]-edge[1])


@jax.jit
def integrate_gap(edge0, edge1):
    n0 = compute_normal(edge0)
    n1 = compute_normal(edge1)
    n = average_normals(n0, n1)
    xi0, xi1, g = compute_intersection(edge0, edge1, n)
    integralLength = (xi0[1]-xi0[0]) * length(edge0) * (n0 @ n)
    return 0.5 * (g[0]+g[1]) * integralLength


@jax.jit
def integrate_gap_against_shape(edge0, edge1):
    n0 = compute_normal(edge0)
    n1 = compute_normal(edge1)
    n = average_normals(n0, n1)
    xi0, xi1, g = compute_intersection(edge0, edge1, n)
    integralLength = (xi0[1]-xi0[0]) * length(edge0) * (n0 @ n)

    xi0t = np.array([xi0[0], 0.5*(xi0[0]+xi0[1]), xi0[1]])
    gt = np.array([g[0]/6, (g[0]+g[1])/3, g[1]/6])
    wght = np.array([1./6, 2./3, 1./6])

    Nl0 = 1.0 - xi0t
    Nr0 = xi0t
    areaGapLeft0 = gt @ Nl0 * integralLength
    areaGapRight0 = gt @ Nr0 * integralLength
    areaLeft0 = wght @ Nl0 * integralLength
    areaRight0 = wght @ Nr0 * integralLength

    xi1t = np.array([xi1[0], 0.5*(xi1[0]+xi1[1]), xi1[1]])

    Nl1 = 1.0 - xi1t
    Nr1 = xi1t
    areaGapLeft1 = gt @ Nl1 * integralLength
    areaGapRight1 = gt @ Nr1 * integralLength
    areaLeft1 = wght @ Nl1 * integralLength
    areaRight1 = wght @ Nr1 * integralLength

    return np.array([areaGapLeft0, areaGapRight0]), np.array([areaGapLeft1, areaGapRight1]), \
           np.array([areaLeft0, areaRight0]), np.array([areaLeft1, areaRight1])


@jax.jit
def get_cut_coordinates(edge0, edge1):
    n0 = compute_normal(edge0)
    n1 = compute_normal(edge1)
    n = average_normals(n0, n1)
    xi0, xi1, g = compute_intersection(edge0, edge1, n)

    left0 = eval_linear_field_on_edge(edge0, xi0[0])
    right0 = eval_linear_field_on_edge(edge0, xi0[1])

    left1 = eval_linear_field_on_edge(edge1, xi1[0])
    right1 = eval_linear_field_on_edge(edge1, xi1[1])

    return np.array([left0, right0]), np.array([left1, right1])
