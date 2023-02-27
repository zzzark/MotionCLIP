import torch


# NOTE:
#   the code below will be removed in the future
#
class _SymbolWrapper:
    """
    Avoid allocating `zero` tensors in memory, which is replaced by `None` object
    """
    def __init__(self, obj):
        self.obj = obj

    def __mul__(self, other):
        if self.obj is None: return _SymbolWrapper(None)
        if other.obj is None: return _SymbolWrapper(None)
        return _SymbolWrapper(self.obj * other.obj)

    def __truediv__(self, other):
        if self.obj is None: return _SymbolWrapper(None)
        if other.obj is None: raise ZeroDivisionError('Error: divided by Zero')
        return _SymbolWrapper(self.obj / other.obj)

    def __floordiv__(self, other):
        if self.obj is None: return _SymbolWrapper(None)
        if other.obj is None: raise ZeroDivisionError('Error: divided by Zero')
        return _SymbolWrapper(self.obj // other.obj)

    def __add__(self, other):
        if self.obj is None: return _SymbolWrapper(other.obj)
        if other.obj is None: return _SymbolWrapper(self.obj)
        return _SymbolWrapper(self.obj + other.obj)

    def __sub__(self, other):
        if self.obj is None: return _SymbolWrapper(None if other.obj is None else -other.obj)
        if other.obj is None: return _SymbolWrapper(self.obj)
        return _SymbolWrapper(self.obj - other.obj)


def _warped_mul_two_quaternions(qa: tuple, qb: tuple) -> tuple:
    """
    perform quaternion multiplication qa * qb
    e.g.
        (w, 0, y, 0) * (w, 0, 0, z)
        ==> _mul_quaternion((w, None, y, None), (w, None, None, z))
        where `None` stands for `zero` in the quaternion
    :param qa: quaternion a
    :param qb: quaternion b
    :return: qa * qb
    """
    if len(qa) != len(qb) or len(qa) != 4:
        raise ValueError(f"Length should be the same and equals to 4, but got qa={len(qa)} while qb={len(qb)}.")

    w1, x1, y1, z1 = tuple(list(_SymbolWrapper(e) for e in qa))
    w2, x2, y2, z2 = tuple(list(_SymbolWrapper(e) for e in qb))
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = x1*w2 + w1*x2 - z1*y2 + y1*z2
    y = y1*w2 + z1*x2 + w1*y2 - x1*z2
    z = z1*w2 - y1*x2 + x1*y2 + w1*z2
    return w.obj, x.obj, y.obj, z.obj


def euler_to_quaternion(eul: torch.Tensor, to_rad, order="ZYX", intrinsic=True) -> torch.Tensor:
    """
    euler rotation -> quaternion
    :param order: rotation order, default is "ZYX"
    :param to_rad: degree to radius (3.14159265 / 180.0)
    :param eul: [(B), J, 3, T] (rad)
    :param intrinsic: intrinsic or extrinsic rotation
    :return: [(B), J, 4, T]
    """
    batch, eul = (True, eul) if len(eul.shape) == 4 else (False, eul[None, ...])

    if len(eul.shape) != 4 or eul.shape[2] != 3:
        raise ValueError('Input tensor should be in the shape of BxJx3xF.')

    if to_rad != 1.0:
        eul = eul * to_rad
    half_eul = eul * 0.5
    s = [torch.sin(half_eul[..., 0:1, :]),
         torch.sin(half_eul[..., 1:2, :]),
         torch.sin(half_eul[..., 2:3, :])]

    c = [torch.cos(half_eul[..., 0:1, :]),
         torch.cos(half_eul[..., 1:2, :]),
         torch.cos(half_eul[..., 2:3, :])]
    r = []
    for i, od in enumerate(order):
        if od == "X": r.append((c[i], s[i], None, None))
        if od == "Y": r.append((c[i], None, s[i], None))
        if od == "Z": r.append((c[i], None, None, s[i]))
    if len(r) != 3:
        raise ValueError(f'Error: Unknown order {order}')

    if intrinsic:
        w, x, y, z = _warped_mul_two_quaternions(r[0], _warped_mul_two_quaternions(r[1], r[2]))
    else:
        w, x, y, z = _warped_mul_two_quaternions(r[2], _warped_mul_two_quaternions(r[1], r[0]))

    ret = torch.cat((w, x, y, z), dim=2)
    return ret if batch else ret[0]


def matrix_to_euler(mtx: torch.Tensor) -> torch.Tensor:
    """
    matrix -> euler
    :param mtx: [(B), J, 3, 3, T]
    :return: euler, [(B), J, 3, T]
    """
    batch, qua = (True, mtx) if len(mtx.shape) == 5 else (False, mtx[None, ...])

    if len(mtx.shape) != 5 or mtx.shape[2] != 3 or mtx.shape[3] != 3:
        raise ValueError('Input tensor should be in the shape of BxJx3x3xF.')

    # reference: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

    r11 = mtx[..., 0:1, 0, :]
    r21 = mtx[..., 1:2, 0, :]
    r31 = mtx[..., 2:3, 0, :]
    r32 = mtx[..., 2:3, 1, :]
    r33 = mtx[..., 2:3, 2, :]

    the1 = -torch.asin(torch.clip(r31, min=-0.9999, max=0.9999))
    cos1 = torch.cos(the1)
    # pai1 = torch.atan2((r32 / cos1), (r33 / cos1))
    # phi1 = torch.atan2((r21 / cos1), (r11 / cos1))
    # -- avoid division by zero
    pai1 = torch.atan2((r32 * cos1), (r33 * cos1))
    phi1 = torch.atan2((r21 * cos1), (r11 * cos1))

    ret = torch.cat((phi1, the1, pai1), dim=2)
    return ret if batch else ret[0]


def quaternion_to_euler(qua: torch.Tensor, order='XYZ', intrinsic=True) -> torch.Tensor:
    """
    quaternion to euler
    :param qua:  [(B), J, 4, T]
    :param order: rotation order of euler angles
    :param intrinsic: intrinsic rotation or extrinsic rotation
    :return: [(B), J, 3, T]
    """

    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        roll = atan2(2xw + 2yz, 1 - 2xx - 2yy)
        pitch = asin(2yw - 2xz)
        yaw = atan2(2zw + 2xy, 1 - 2yy - 2zz)
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor must be in the shape of BxJx4xF.')

    # extrinsic to intrinsic
    if not intrinsic:
        intrinsic = True
        order = order[::-1]

    w = qua[..., 0:1, :]
    x = qua[..., 1:2, :]
    y = qua[..., 2:3, :]
    z = qua[..., 3:4, :]

    if order == "XYZ":
        xx = 2 * x*x
        yy = 2 * y*y
        zz = 2 * z*z
        xy = 2 * x*y
        xz = 2 * x*z
        xw = 2 * x*w
        yz = 2 * y*z
        yw = 2 * y*w
        zw = 2 * z*w

        roll  = torch.arctan2(xw + yz, 1 - xx - yy)
        pitch = torch.arcsin(torch.clip(yw - xz, min=-0.9999, max=0.9999))
        yaw   = torch.arctan2(zw + xy, 1 - yy - zz)

        # first roll, then pitch, then yaw
        # yaw * pitch * roll * V
        ret = torch.cat((yaw, pitch, roll), dim=2)

    elif order == "YZX":
        xx, yy, zz, ww = x * x, y * y, z * z, w * w
        ex = torch.atan2(2 * (x * w - y * z), -xx + yy - zz + ww)
        ey = torch.atan2(2 * (y * w - x * z), xx - yy - zz + ww)
        ez = torch.asin(torch.clamp(2 * (x * y + z * w), min=-0.9999, max=0.9999))
        ret = torch.cat((ex, ez, ey), dim=2)
    else:
        raise NotImplementedError

    return ret if batch else ret[0]


def quaternion_to_euler_2(qua: torch.Tensor, order, intrinsic) -> torch.Tensor:
    """
    quaternion to euler
    :param qua:  [(B), J, 4, T]
    :param order: rotation order of euler angles
    :param intrinsic: intrinsic rotation or extrinsic rotation
    :return: [(B), J, 3, T]
    """
    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor must be in the shape of BxJx4xF.')

    if order != "ZYX" or intrinsic:  # only export "ZYX" euler angles, in extrinsic rotation
        raise NotImplementedError

    mtx = quaternion_to_matrix(qua)
    eul = matrix_to_euler(mtx)
    return eul


def normalize_quaternion(qua: torch.Tensor) -> torch.Tensor:
    """
    euler rotation -> quaternion
    :param qua: [(B), J, 4, T]
    :return: [(B), J, 4, T]
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor should be in the shape of BxJx4xF.')

    ret = torch.nn.functional.normalize(qua, p=2.0, dim=2)
    return ret if batch else ret[0]

    # s = torch.norm(qua, dim=2, keepdim=True)
    # # s = torch.sqrt(torch.sum(qua**2, dim=2, keepdim=True))
    # s = torch.broadcast_to(s, qua.shape)
    # return torch.div(qua, s)


def quaternion_to_matrix(qua: torch.Tensor) -> torch.Tensor:
    """
    quaternion -> matrix
    :param qua: [(B), J, 4, T]
    :return: [(B), J, 3, 3, T]
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor should be in the shape of BxJx4xF.')

    w = qua[..., 0:1, :][..., None, :]
    x = qua[..., 1:2, :][..., None, :]
    y = qua[..., 2:3, :][..., None, :]
    z = qua[..., 3:4, :][..., None, :]
    xx = 2 * x*x
    yy = 2 * y*y
    zz = 2 * z*z
    xy = 2 * x*y
    xz = 2 * x*z
    xw = 2 * x*w
    yz = 2 * y*z
    yw = 2 * y*w
    zw = 2 * z*w

    r11, r12, r13 = 1 - yy - zz,      xy - zw,      xz + yw
    r21, r22, r23 =     xy + zw,  1 - xx - zz,      yz - xw
    r31, r32, r33 =     xz - yw,      yz + xw,  1 - xx - yy

    r1 = torch.cat((r11, r12, r13), dim=3)
    r2 = torch.cat((r21, r22, r23), dim=3)
    r3 = torch.cat((r31, r32, r33), dim=3)
    ret = torch.cat((r1, r2, r3), dim=2)
    return ret if batch else ret[0]


def quaternion_from_two_vectors(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    :param v0: [..., 3, F], start
    :param v1: [..., 3, F], end
    :return:
    """
    # Quaternion q;
    # vector a = crossproduct(v1, v2);
    # q.xyz = a;
    # q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
    a = torch.cross(v0, v1, dim=-2)
    l0 = torch.norm(v0, dim=-2, keepdim=True)
    l1 = torch.norm(v1, dim=-2, keepdim=True)
    dot = torch.sum(v0 * v1, dim=-2, keepdim=True)
    w = l0*l1 + dot  # fix bug: since torch.norm <==> torch.sqrt(v ** 2) there is no need to use torch.sqrt any more
    qua = torch.cat([w, a], dim=-2)
    qua = torch.nn.functional.normalize(qua, p=2.0, dim=-2)
    return qua


def conjugate_quaternion(q) -> torch.Tensor:
    """
    :param q: [..., 4, F]
    :return: [..., 4, F]
    """
    qc = q.clone()
    qc[..., 1:, :] = -q[..., 1:, :]
    return qc


def norm_of_quaternion(q) -> torch.Tensor:
    """
    :param q: [..., 4, F]
    :return: [..., 1, F]
    """
    qn = torch.norm(q, dim=-2, keepdim=True)
    return qn


def inverse_quaternion(q) -> torch.Tensor:
    """
    :param q: [..., 4, F]
    :return: [..., 4, F]
    """
    return conjugate_quaternion(q) / norm_of_quaternion(q)


def mul_two_quaternions(q0, q1) -> torch.Tensor:
    """
    perform quaternion multiplication qa * qb
    :param q0: [..., 4, F], quaternion 0
    :param q1: [..., 4, F], quaternion 0
    :return: q0 * q1
    """
    qa = (q0[..., 0:1, :], q0[..., 1:2, :], q0[..., 2:3, :], q0[..., 3:4, :])
    qb = (q1[..., 0:1, :], q1[..., 1:2, :], q1[..., 2:3, :], q1[..., 3:4, :])
    return torch.cat(_warped_mul_two_quaternions(qa, qb), dim=-2)


def rectify_w_of_quaternion(qua: torch.Tensor, inplace=False) -> torch.Tensor:
    """
    quaternion[w < 0] --> quaternion[w < 0]
    :param qua: [(B), J, 4, T]
    :param inplace: inplace operator or not
    :return: [(B), J, 4, T]
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor should be in the shape of [(B), J, 4, T].')

    w_lt = (qua[:, :, [0], :] < 0.0).expand(-1, -1, 4, -1)  # w less than 0.0
    w_ge = torch.logical_not(w_lt)  # w greater equal than 0.0

    if inplace:
        qua[w_lt] *= -1
    else:
        new = torch.empty_like(qua, dtype=qua.dtype, device=qua.device)
        new[w_ge] = qua[w_ge]
        new[w_lt] = qua[w_lt] * (-1)
        qua = new

    return qua if batch else qua[0]


def pad_position_to_quaternion(_xyz: torch.Tensor) -> torch.Tensor:
    """
    :param _xyz: [..., 3, (F)]
    :return:
    """
    no_frame = len(_xyz.shape) == 1
    if no_frame: _xyz = _xyz[:, None]
    assert _xyz.shape[-2] == 3, "input tensor shape should be [..., 3, (F)]"

    zero = torch.zeros_like(_xyz, dtype=_xyz.dtype, device=_xyz.device)[..., [0], :]
    wxyz = torch.concat([zero, _xyz], dim=-2)

    if no_frame: wxyz = wxyz[..., 0]

    return wxyz


# def simple_test():
#     v0 = torch.zeros((2, 3, 1))
#     v1 = torch.zeros((2, 3, 1))
#     v0[..., 0, :] = 1.0
#     v1[..., 1, :] = 1.0
#     print(quaternion_from_two_vectors(v0, v1))
#
#     v0 = torch.zeros((1, 2, 3, 1))
#     v1 = torch.zeros((1, 2, 3, 1))
#     v0[..., 1, :] = 1.0
#     v1[..., 0, :] = 1.0
#     print(quaternion_from_two_vectors(v0, v1))
#
#
# if __name__ == '__main__':
#     simple_test()
