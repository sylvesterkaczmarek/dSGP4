import torch

from .util import initialize_tle, propagate


TLE_ELEMENT_ORDER = (
    "bstar",
    "ndot",
    "nddot",
    "ecco",
    "argpo",
    "inclo",
    "mo",
    "no_kozai",
    "nodeo",
)


def state_jacobian(
    tle,
    tsince,
    gravity_constant_name="wgs-84",
    create_graph=False,
):
    """
    Propagate a TLE and compute the Cartesian state Jacobian with respect to
    the nine internal SGP4/TLE parameters.

    The parameter order is given by :data:`TLE_ELEMENT_ORDER`. Angular
    quantities and mean motion use dSGP4's internal units (radians and
    radians/minute), rather than the degree and revolution/day units printed
    in a TLE.

    Parameters
    ----------
    tle : dsgp4.tle.TLE
        Single TLE (or compatible OMM) to propagate. The input object is not
        modified; differentiation is performed on a copy.
    tsince : torch.Tensor
        Minutes since the element epoch. Scalar and vector times are supported.
    gravity_constant_name : str
        Gravity model passed to ``initialize_tle``.
    create_graph : bool
        Preserve the derivative graph so higher-order derivatives can be
        computed from the returned Jacobian.

    Returns
    -------
    state : torch.Tensor
        Propagated TEME state in km and km/s.
    jacobian : torch.Tensor
        ``state.shape + (9,)`` tensor containing derivatives of every state
        component with respect to the parameters in ``TLE_ELEMENT_ORDER``.
    """
    if isinstance(tle, list):
        raise TypeError("state_jacobian expects a single TLE or OMM object")

    working_tle = tle.copy()
    elements = initialize_tle(
        working_tle,
        gravity_constant_name=gravity_constant_name,
        with_grad=True,
    )
    state = propagate(working_tle, tsince, initialized=True)
    flat_state = state.reshape(-1)

    rows = []
    for index, component in enumerate(flat_state):
        retain_graph = index < flat_state.numel() - 1 or create_graph
        gradient = torch.autograd.grad(
            component,
            elements,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )[0]
        rows.append(gradient)

    jacobian = torch.stack(rows).reshape(*state.shape, len(TLE_ELEMENT_ORDER))
    return state, jacobian


def propagate_covariance(
    tle,
    tsince,
    covariance,
    gravity_constant_name="wgs-84",
):
    """
    Propagate a covariance in dSGP4's nine-parameter TLE space to Cartesian
    TEME state covariance using the first-order map ``J P J.T``.

    ``covariance`` must correspond to :data:`TLE_ELEMENT_ORDER` and therefore
    use the internal units returned by ``initialize_tle(with_grad=True)``.

    Parameters
    ----------
    tle : dsgp4.tle.TLE
        Single TLE (or compatible OMM).
    tsince : torch.Tensor
        Minutes since epoch. Scalar and vector times are supported.
    covariance : array-like or torch.Tensor
        9 x 9 covariance of the internal TLE parameters.
    gravity_constant_name : str
        Gravity model passed to ``initialize_tle``.

    Returns
    -------
    state : torch.Tensor
        Propagated TEME state in km and km/s.
    state_covariance : torch.Tensor
        6 x 6 covariance for scalar ``tsince`` or N x 6 x 6 for N times.
    jacobian : torch.Tensor
        State Jacobian returned by :func:`state_jacobian`.
    """
    state, jacobian = state_jacobian(
        tle,
        tsince,
        gravity_constant_name=gravity_constant_name,
        create_graph=False,
    )
    covariance = torch.as_tensor(
        covariance,
        dtype=jacobian.dtype,
        device=jacobian.device,
    )
    if covariance.shape != (len(TLE_ELEMENT_ORDER), len(TLE_ELEMENT_ORDER)):
        raise ValueError(
            "covariance must have shape (9, 9) in TLE_ELEMENT_ORDER"
        )

    jacobian_2d = jacobian.reshape(-1, 6, len(TLE_ELEMENT_ORDER))
    state_covariance = (
        jacobian_2d
        @ covariance
        @ jacobian_2d.transpose(-1, -2)
    )

    if state.ndim == 2:
        state_covariance = state_covariance[0]
    else:
        state_covariance = state_covariance.reshape(
            *state.shape[:-2],
            6,
            6,
        )

    return state, state_covariance, jacobian
