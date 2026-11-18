import numpy as np

# the search for the flat surface uses the median of this number of frames, at every SEARCH_STEP pixel
REFERENCE_FRAMES = 9
SEARCH_STEP = 2
# the planes through 3 random pixels that the search compares; the random numbers have a fixed seed, so the result does not change
CANDIDATE_PLANES = 200
# a pixel of the median image is on the surface when its disparity is this near to the plane
SURFACE_TOLERANCE_PX = 0.5
# the surface pixels nearer than this number of search pixels to a pixel that is not on the surface are removed
EDGE_MARGIN = 4
MIN_SURFACE_PIXELS = 1000
# the plane fit of each frame uses at most this number of surface pixels
MAX_FIT_PIXELS = 1024


# the part of the pixels of each frame that has a depth value
def fill_rates(depths: list[np.ndarray]) -> np.ndarray:
    return np.array([np.count_nonzero(depth) / depth.size for depth in depths])


def erode(mask: np.ndarray, margin: int) -> np.ndarray:
    for axis in (0, 1):
        padded = np.pad(mask, [(margin, margin) if a == axis else (0, 0) for a in (0, 1)])
        mask = np.all([np.take(padded, range(shift, shift + mask.shape[axis]), axis=axis) for shift in range(2 * margin + 1)], axis=0)
    return mask


# the pixels (u, v) of the largest flat surface of the scene and the part of the image that the surface covers, or None
# a flat surface is a plane in disparity; disparity_scale / raw depth value = disparity in pixels
def flat_surface(depths: list[np.ndarray], disparity_scale: float) -> tuple[np.ndarray, np.ndarray, float] | None:
    sample = [depth[::SEARCH_STEP, ::SEARCH_STEP] for depth in depths[:: max(1, len(depths) // REFERENCE_FRAMES)]]
    reference = np.median(np.stack(sample), axis=0)
    valid = reference > 0
    if np.count_nonzero(valid) < MIN_SURFACE_PIXELS:
        return None
    v, u = np.nonzero(valid)
    points = np.stack([u * SEARCH_STEP, v * SEARCH_STEP, disparity_scale / reference[valid]], axis=1)
    triples = points[np.random.default_rng(0).integers(0, len(points), (CANDIDATE_PLANES, 3))]
    normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
    # 3 pixels on one line give no plane
    triples, normals = triples[np.abs(normals[:, 2]) > 1e-9], normals[np.abs(normals[:, 2]) > 1e-9]
    if len(triples) == 0:
        return None
    slope_u, slope_v = -normals[:, 0] / normals[:, 2], -normals[:, 1] / normals[:, 2]
    offset = triples[:, 0, 2] - slope_u * triples[:, 0, 0] - slope_v * triples[:, 0, 1]
    test = points[:: max(1, len(points) // 5000)]
    residuals = test[:, 2] - (slope_u[:, None] * test[:, 0] + slope_v[:, None] * test[:, 1] + offset[:, None])
    best = np.argmax(np.count_nonzero(np.abs(residuals) < SURFACE_TOLERANCE_PX, axis=1))
    plane = np.array([slope_u[best], slope_v[best], offset[best]])
    design = np.stack([points[:, 0], points[:, 1], np.ones(len(points))], axis=1)
    for _ in range(2):
        on_plane = np.abs(points[:, 2] - design @ plane) < SURFACE_TOLERANCE_PX
        if np.count_nonzero(on_plane) < MIN_SURFACE_PIXELS:
            return None
        plane, *_ = np.linalg.lstsq(design[on_plane], points[on_plane, 2], rcond=None)
    mask = np.zeros(valid.shape, dtype=bool)
    mask[v, u] = np.abs(points[:, 2] - design @ plane) < SURFACE_TOLERANCE_PX
    mask = erode(mask, EDGE_MARGIN)
    count = np.count_nonzero(mask)
    if count < MIN_SURFACE_PIXELS:
        return None
    v, u = np.nonzero(mask)
    step = -(-count // MAX_FIT_PIXELS)
    return u[::step] * SEARCH_STEP, v[::step] * SEARCH_STEP, count / mask.size


# the median of the values of each row where keep is True, as a column; each row has a value to keep
# one sort of all the rows is faster than np.nanmedian, which makes one call for each row
def row_medians(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.where(keep, values, np.inf), axis=1)
    count = np.count_nonzero(keep, axis=1)[:, None]
    return (np.take_along_axis(ordered, (count - 1) // 2, axis=1) + np.take_along_axis(ordered, count // 2, axis=1)) / 2


# noise: the RMS distance in disparity pixels from the surface pixels of each frame to the plane that fits them best; NaN for a
# frame with a depth value on less than half of the surface pixels
# the second value is the median raw depth value of the surface pixels
def surface_noise(depths: list[np.ndarray], u: np.ndarray, v: np.ndarray, disparity_scale: float) -> tuple[np.ndarray, float]:
    raw = np.stack([depth[v, u] for depth in depths])
    valid = raw > 0
    noise = np.full(len(depths), np.nan)
    rows = np.count_nonzero(valid, axis=1) >= len(u) // 2
    if not rows.any():
        return noise, float("nan")
    valid = valid[rows]
    disparity = disparity_scale / np.where(valid, raw[rows], 1)
    # the plane of each frame comes from the normal equations of all the frames in one step
    design = np.stack([(u - u.mean()) / 100, (v - v.mean()) / 100, np.ones(len(u))], axis=1)
    products = (design[:, :, None] * design[:, None, :]).reshape(len(u), 9)
    keep = valid
    # 2 fits: the second fit has no pixel further than 3 sigma from the median residual of the first fit (a hand or an arm in front
    # of the surface); an object in front of the surface pulls the first plane, so the residuals of the surface are not centered on 0,
    # and a limit around 0 can keep no pixel; a limit around the median keeps at least half of the pixels of the fit
    for _ in range(2):
        normal = (keep @ products).reshape(-1, 3, 3)
        moment = (keep * disparity) @ design
        planes = np.linalg.solve(normal, moment[:, :, None])[:, :, 0]
        residual = disparity - planes @ design.T
        center = row_medians(residual, keep)
        sigma = 1.4826 * row_medians(np.abs(residual - center), keep)
        keep = valid & (np.abs(residual - center) < 3 * sigma + 1e-9)
    noise[rows] = np.sqrt(np.nanmean(np.where(keep, residual**2, np.nan), axis=1))
    return noise, float(np.median(raw[rows][valid]))
