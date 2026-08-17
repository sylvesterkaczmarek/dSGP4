import torch


def rtn_rotation_matrix(state):
    """
    Construct the instantaneous Cartesian-to-RTN direction cosine matrix.

    The returned matrix has radial, transverse, and normal unit vectors as
    rows. It can therefore left-multiply Cartesian vector components to express
    them in the local RTN frame.

    Parameters
    ----------
    state : array-like or torch.Tensor
        Cartesian state with trailing shape ``(2, 3)``. The first row is
        position and the second is velocity. Leading batch dimensions are
        supported.

    Returns
    -------
    torch.Tensor
        Rotation matrix with trailing shape ``(3, 3)``.

    Notes
    -----
    The frame is undefined for zero position or zero angular momentum
    (collinear position and velocity), in which case a ``ValueError`` is
    raised.
    """
    state = torch.as_tensor(state)
    if not torch.is_floating_point(state):
        state = state.to(torch.get_default_dtype())
    if state.shape[-2:] != (2, 3):
        raise ValueError("state must have trailing shape (2, 3)")

    position = state[..., 0, :]
    velocity = state[..., 1, :]

    position_norm = torch.linalg.vector_norm(
        position,
        dim=-1,
        keepdim=True,
    )
    angular_momentum = torch.linalg.cross(
        position,
        velocity,
        dim=-1,
    )
    angular_momentum_norm = torch.linalg.vector_norm(
        angular_momentum,
        dim=-1,
        keepdim=True,
    )

    threshold = 10.0 * torch.finfo(state.dtype).eps
    if torch.any(position_norm <= threshold).detach().item():
        raise ValueError("RTN frame is undefined for zero position")
    if torch.any(angular_momentum_norm <= threshold).detach().item():
        raise ValueError(
            "RTN frame is undefined for collinear position and velocity"
        )

    radial = position / position_norm
    normal = angular_momentum / angular_momentum_norm
    transverse = torch.linalg.cross(normal, radial, dim=-1)

    return torch.stack((radial, transverse, normal), dim=-2)


def cartesian_covariance_to_rtn(state, covariance):
    """
    Rotate a Cartesian 6 x 6 position/velocity covariance into RTN components.

    A block-diagonal matrix ``diag(R, R)`` is applied, where ``R`` is the
    instantaneous Cartesian-to-RTN rotation from :func:`rtn_rotation_matrix`.
    This is the component-frame covariance transform commonly used for
    orbital-state uncertainty at a fixed epoch.

    Parameters
    ----------
    state : array-like or torch.Tensor
        Cartesian state with trailing shape ``(2, 3)``.
    covariance : array-like or torch.Tensor
        Cartesian covariance with trailing shape ``(6, 6)``. Leading
        dimensions may be broadcast against those of ``state``.

    Returns
    -------
    covariance_rtn : torch.Tensor
        Covariance expressed in RTN components.
    rotation : torch.Tensor
        The 3 x 3 Cartesian-to-RTN rotation used in the transformation.

    Notes
    -----
    This rotates position and velocity *components* at one epoch. It does not
    add rotating-frame kinematic terms associated with differentiating RTN
    coordinates in time.
    """
    rotation = rtn_rotation_matrix(state)
    covariance = torch.as_tensor(
        covariance,
        dtype=rotation.dtype,
        device=rotation.device,
    )
    if covariance.shape[-2:] != (6, 6):
        raise ValueError("covariance must have trailing shape (6, 6)")

    zero = torch.zeros_like(rotation)
    upper = torch.cat((rotation, zero), dim=-1)
    lower = torch.cat((zero, rotation), dim=-1)
    transform = torch.cat((upper, lower), dim=-2)

    covariance_rtn = (
        transform
        @ covariance
        @ transform.transpose(-1, -2)
    )
    return covariance_rtn, rotation
