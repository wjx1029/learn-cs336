import math

def lr_cosine_schedule(t: int, lr_min: float, lr_max: float, warm_up: int, cosine_end: int):
    if t < warm_up:
        return t / warm_up * lr_max
    elif warm_up <= t and t <= cosine_end:
        return lr_min + 0.5 * (1 + math.cos((t - warm_up) / (cosine_end - warm_up) * math.pi)) * (lr_max - lr_min)
    elif t > cosine_end:
        return lr_min